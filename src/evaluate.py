"""
Model Evaluation — Score predictions against known tournament outcomes.

Evaluates:
  1. Overall Log Loss (the Kaggle metric)
  2. Per-Season Log Loss
  3. Per-Seed-Matchup accuracy & calibration
  4. Calibration plot (reliability diagram)
  5. Upset detection analysis
  6. Base model comparison
"""

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path

from src.config import (
    RAW_DIR, PROCESSED_DIR, SUBMISSIONS_DIR,
    FIRST_DETAILED_SEASON, LAST_HISTORICAL_SEASON,
)
from src.data_loader import load_tourney_compact, make_game_id
from src.tabular.seeds import build_seed_lookup


# ═══════════════════════════════════════════════════════════════════════════
# 1.  LOAD ACTUAL RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def load_actual_results(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Build a DataFrame of actual tournament outcomes with canonical GameIDs.

    Returns: GameID, Season, TeamA, TeamB, Result (1 = TeamA won)
    """
    dfs = []
    for prefix, is_w in [("M", 0), ("W", 1)]:
        comp = load_tourney_compact(prefix)
        if seasons:
            comp = comp[comp.Season.isin(seasons)]
        for _, r in comp.iterrows():
            s = int(r.Season)
            w, l = int(r.WTeamID), int(r.LTeamID)
            lo, hi = sorted([w, l])
            dfs.append({
                "GameID": make_game_id(s, w, l),
                "Season": s,
                "TeamA": lo,
                "TeamB": hi,
                "Result": 1 if w == lo else 0,
                "is_women": is_w,
                "WScore": int(r.WScore),
                "LScore": int(r.LScore),
            })

    actuals = pd.DataFrame(dfs)

    # Add seeds
    seeds = build_seed_lookup()
    actuals = actuals.merge(
        seeds[["Season", "TeamID", "SeedNum"]].rename(
            columns={"TeamID": "TeamA", "SeedNum": "SeedA"}
        ), on=["Season", "TeamA"], how="left"
    )
    actuals = actuals.merge(
        seeds[["Season", "TeamID", "SeedNum"]].rename(
            columns={"TeamID": "TeamB", "SeedNum": "SeedB"}
        ), on=["Season", "TeamB"], how="left"
    )
    # Determine which seed was higher (lower number = better)
    actuals["HighSeed"] = actuals[["SeedA", "SeedB"]].min(axis=1)
    actuals["LowSeed"]  = actuals[["SeedA", "SeedB"]].max(axis=1)
    actuals["Upset"] = (
        (actuals.Result == 1) & (actuals.SeedA > actuals.SeedB) |
        (actuals.Result == 0) & (actuals.SeedB > actuals.SeedA)
    ).astype(int)

    return actuals


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SCORE PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_submission(stage: int = 1,
                        seasons: list[int] | None = None) -> dict:
    """
    Score a submission against actual tournament results.

    Returns a dict with all metrics.
    """
    if seasons is None:
        if stage == 1:
            seasons = [2022, 2023, 2024, 2025]
        else:
            raise ValueError("Stage 2 is 2026 — no actuals available yet!")

    # Load submission
    sub_path = SUBMISSIONS_DIR / f"submission_stage{stage}.csv"
    sub = pd.read_csv(sub_path)
    print(f"Loaded submission: {sub_path} ({len(sub):,} rows)")

    # Load actuals
    actuals = load_actual_results(seasons)
    print(f"Loaded {len(actuals):,} actual tournament games for {seasons}")

    # Merge predictions with actuals
    merged = actuals.merge(sub, left_on="GameID", right_on="ID", how="inner")
    print(f"Matched {len(merged):,} games with predictions")

    results = {}

    # ── Overall Metrics ──────────────────────────────────────────────
    ll = log_loss(merged.Result, merged.Pred)
    bs = brier_score_loss(merged.Result, merged.Pred)
    results["overall_log_loss"] = ll
    results["overall_brier"] = bs

    print(f"\n{'='*50}")
    print(f"  OVERALL LOG LOSS:   {ll:.4f}")
    print(f"  OVERALL BRIER:      {bs:.4f}")
    print(f"{'='*50}")

    # ── Per-Season ───────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  PER-SEASON LOG LOSS")
    print(f"{'─'*50}")
    season_results = {}
    for s in sorted(merged.Season.unique()):
        mask = merged.Season == s
        n = mask.sum()
        sl = log_loss(merged.loc[mask, "Result"], merged.loc[mask, "Pred"])
        m_mask = mask & (merged.is_women == 0)
        w_mask = mask & (merged.is_women == 1)
        m_ll = log_loss(merged.loc[m_mask, "Result"], merged.loc[m_mask, "Pred"]) if m_mask.sum() > 0 else None
        w_ll = log_loss(merged.loc[w_mask, "Result"], merged.loc[w_mask, "Pred"]) if w_mask.sum() > 0 else None
        season_results[s] = {"log_loss": sl, "n_games": n, "men_ll": m_ll, "women_ll": w_ll}
        print(f"  {s}: LL={sl:.4f}  ({n} games)  "
              f"M={f'{m_ll:.4f}' if m_ll else 'N/A'}  "
              f"W={f'{w_ll:.4f}' if w_ll else 'N/A'}")
    results["per_season"] = season_results

    # ── Per-Seed Matchup Analysis ────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  SEED MATCHUP ANALYSIS (Favorite Win Rate)")
    print(f"{'─'*50}")
    merged["SeedDiff"] = merged.LowSeed - merged.HighSeed   # always ≥ 0
    merged["FavWin"] = 1 - merged.Upset
    matchup_groups = merged.groupby(["HighSeed", "LowSeed"]).agg(
        n_games=("Result", "count"),
        actual_fav_win=("FavWin", "mean"),
    ).reset_index()
    # Compute average predicted probability for the favorite in each matchup
    # (Higher seed = favorite; if Result-aligned, use Pred directly is tricky
    #  so we compute both directions)

    # Classic matchups
    classics = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12),
                (6, 11), (7, 10), (8, 9)]
    for hi, lo in classics:
        mask = (merged.HighSeed == hi) & (merged.LowSeed == lo)
        if mask.sum() == 0:
            continue
        sub_m = merged[mask]
        actual_fav = sub_m.FavWin.mean()
        # Get the predicted prob for the favorite
        # Favorite is the team with the lower seed number
        fav_pred = []
        for _, row in sub_m.iterrows():
            if row.SeedA <= row.SeedB:
                fav_pred.append(row.Pred)      # TeamA is fav, Pred = P(TeamA wins)
            else:
                fav_pred.append(1 - row.Pred)  # TeamB is fav, Pred = P(TeamA wins)
        avg_fav_pred = np.mean(fav_pred)
        print(f"  {hi:2d} vs {lo:2d}:  "
              f"Actual={actual_fav:.1%}  Predicted={avg_fav_pred:.1%}  "
              f"({mask.sum()} games)  "
              f"{'✓' if abs(actual_fav - avg_fav_pred) < 0.1 else '⚠ MISCALIBRATED'}")

    # ── Calibration (Reliability Diagram) ────────────────────────────
    print(f"\n{'─'*50}")
    print("  CALIBRATION (Reliability Diagram)")
    print(f"{'─'*50}")
    bins = np.linspace(0, 1, 11)
    bin_labels = []
    bin_pred_means = []
    bin_actual_means = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = (merged.Pred >= bins[i]) & (merged.Pred < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_labels.append(f"{bins[i]:.1f}–{bins[i+1]:.1f}")
        bin_pred_means.append(merged.loc[mask, "Pred"].mean())
        bin_actual_means.append(merged.loc[mask, "Result"].mean())
        bin_counts.append(mask.sum())
        print(f"  Pred {bins[i]:.1f}–{bins[i+1]:.1f}: "
              f"avg_pred={bin_pred_means[-1]:.3f}  actual={bin_actual_means[-1]:.3f}  "
              f"n={bin_counts[-1]}")

    # Save calibration plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.scatter(bin_pred_means, bin_actual_means, s=[c*3 for c in bin_counts],
               alpha=0.7, c='#2196F3', edgecolors='white', linewidth=1)
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_title("Calibration (Reliability Diagram)", fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Prediction distribution
    ax = axes[1]
    ax.hist(merged.Pred, bins=50, alpha=0.7, color='#4CAF50', edgecolor='white')
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Distribution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = SUBMISSIONS_DIR / "calibration_stage1.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Calibration plot saved → {plot_path}")

    # ── Worst Predictions (most overconfident misses) ────────────────
    print(f"\n{'─'*50}")
    print("  TOP 10 WORST PREDICTIONS (most costly log-loss)")
    print(f"{'─'*50}")
    merged["game_ll"] = -(
        merged.Result * np.log(merged.Pred.clip(1e-15)) +
        (1 - merged.Result) * np.log((1 - merged.Pred).clip(1e-15))
    )
    worst = merged.nlargest(10, "game_ll")
    for _, row in worst.iterrows():
        teams_info = f"({int(row.SeedA):2d}) vs ({int(row.SeedB):2d})"
        score = f"{int(row.WScore)}–{int(row.LScore)}"
        outcome = "TeamA" if row.Result == 1 else "TeamB"
        print(f"  {row.GameID}  {teams_info}  Score: {score}  "
              f"Pred: {row.Pred:.3f}  Actual: {outcome}  "
              f"LL: {row.game_ll:.3f}  {'UPSET' if row.Upset else ''}")

    results["merged"] = merged
    return results


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = evaluate_submission(stage=1)
