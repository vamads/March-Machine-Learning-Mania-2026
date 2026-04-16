"""
Raddar-Style Baseline — Standalone XGBoost regressor for March Madness.

A single-model baseline following the Raddar feature engineering approach:
  - 29 delta features: seeds, box-score averages, Elo, quality
  - XGBoost regressor with 'winning' hyperparams
  - Temporal holdout: train 2003-2021, test 2022-2025

Usage:
    cd /path/to/march_machine_learning_mania
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n madness python baseline_raddar/raddar_baseline.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import brier_score_loss, log_loss

# ── Project paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "march-machine-learning-mania-2026"
OUTPUT_DIR = Path(__file__).resolve().parent
REGULAR_SEASON_CUTOFF = 132

HOLDOUT_SEASONS = [2022, 2023, 2024, 2025]
TRAIN_SEASONS = list(range(2003, 2022))  # 2003-2021
ALL_SEASONS = list(range(2003, 2026))


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load all required data files."""
    data = {}
    for prefix, is_w in [("M", 0), ("W", 1)]:
        det = pd.read_csv(RAW_DIR / f"{prefix}RegularSeasonDetailedResults.csv")
        det["is_women"] = is_w
        comp = pd.read_csv(RAW_DIR / f"{prefix}RegularSeasonCompactResults.csv")
        comp["is_women"] = is_w
        tourn = pd.read_csv(RAW_DIR / f"{prefix}NCAATourneyCompactResults.csv")
        tourn["is_women"] = is_w
        seeds = pd.read_csv(RAW_DIR / f"{prefix}NCAATourneySeeds.csv")
        seeds["is_women"] = is_w
        data[prefix] = {
            "detailed": det, "compact": comp,
            "tourney": tourn, "seeds": seeds,
        }
    return data


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SEED PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_seeds(data):
    """Parse seed strings into numeric values."""
    all_seeds = []
    for prefix in ["M", "W"]:
        seeds = data[prefix]["seeds"].copy()
        seeds["SeedNum"] = seeds["Seed"].str[1:3].astype(int)
        all_seeds.append(seeds[["Season", "TeamID", "SeedNum"]])
    return pd.concat(all_seeds, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  ELO RATINGS
# ═══════════════════════════════════════════════════════════════════════════

def compute_elo(data, k=20, init=1500, regress=0.6):
    """Compute Elo ratings from compact results (M + W combined)."""
    elo = defaultdict(lambda: init)
    all_rows = []

    for prefix in ["M", "W"]:
        compact = data[prefix]["compact"]
        for season in sorted(compact.Season.unique()):
            # Season regression
            for tid in list(elo.keys()):
                elo[tid] = init + regress * (elo[tid] - init)

            games = compact[
                (compact.Season == season) &
                (compact.DayNum <= REGULAR_SEASON_CUTOFF)
            ].sort_values("DayNum")

            for _, row in games.iterrows():
                w, l = int(row.WTeamID), int(row.LTeamID)
                margin = int(row.WScore - row.LScore)
                exp_w = 1.0 / (1.0 + 10.0 ** ((elo[l] - elo[w]) / 400.0))
                mult = np.log1p(abs(margin)) / (exp_w + 0.5)
                k_adj = k * mult
                elo[w] += k_adj * (1.0 - exp_w)
                elo[l] += k_adj * (0.0 - (1.0 - exp_w))

            active = set(games.WTeamID.unique()) | set(games.LTeamID.unique())
            for tid in active:
                all_rows.append({
                    "Season": season, "TeamID": int(tid),
                    "Elo": elo[int(tid)],
                })

    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  RADDAR-STYLE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def build_team_stats(data):
    """
    Build per-team per-season averages from detailed box scores.

    Returns DataFrame with columns:
      Season, TeamID, avg_Score, avg_FGA, avg_OR, avg_DR, avg_Blk, avg_PF,
      avg_opponent_FGA, avg_opponent_Blk, avg_opponent_PF, avg_PointDiff,
      Quality (win rate as a simple quality metric)
    """
    all_stats = []

    for prefix in ["M", "W"]:
        det = data[prefix]["detailed"]
        det = det[det.DayNum <= REGULAR_SEASON_CUTOFF].copy()

        stat_cols = ["Score", "FGA", "OR", "DR", "Blk", "PF"]
        opp_cols = ["FGA", "Blk", "PF"]

        # Winner perspective
        w = det.copy()
        w_rename = {f"W{c}": c for c in stat_cols}
        w_opp_rename = {f"L{c}": f"opponent_{c}" for c in opp_cols}
        w = w.rename(columns={
            "WTeamID": "TeamID", "WScore": "Score", "LScore": "OppScore",
            **{k: v for k, v in w_rename.items() if k != "WScore"},
            **w_opp_rename,
        })
        w["Win"] = 1
        w["PointDiff"] = w["Score"] - w["OppScore"]

        # Loser perspective
        l = det.copy()
        l_rename = {f"L{c}": c for c in stat_cols}
        l_opp_rename = {f"W{c}": f"opponent_{c}" for c in opp_cols}
        l = l.rename(columns={
            "LTeamID": "TeamID", "LScore": "Score", "WScore": "OppScore",
            **{k: v for k, v in l_rename.items() if k != "LScore"},
            **l_opp_rename,
        })
        l["Win"] = 0
        l["PointDiff"] = l["Score"] - l["OppScore"]

        keep = ["Season", "TeamID", "Score", "FGA", "OR", "DR", "Blk", "PF",
                "opponent_FGA", "opponent_Blk", "opponent_PF", "PointDiff", "Win"]
        games = pd.concat([w[keep], l[keep]], ignore_index=True)

        # Aggregate to season averages
        agg = games.groupby(["Season", "TeamID"]).agg(
            avg_Score        = ("Score",        "mean"),
            avg_FGA          = ("FGA",          "mean"),
            avg_OR           = ("OR",           "mean"),
            avg_DR           = ("DR",           "mean"),
            avg_Blk          = ("Blk",          "mean"),
            avg_PF           = ("PF",           "mean"),
            avg_opponent_FGA = ("opponent_FGA", "mean"),
            avg_opponent_Blk = ("opponent_Blk", "mean"),
            avg_opponent_PF  = ("opponent_PF",  "mean"),
            avg_PointDiff    = ("PointDiff",    "mean"),
            Quality          = ("Win",          "mean"),  # Win rate as quality
            GP               = ("Win",          "count"),
        ).reset_index()

        all_stats.append(agg)

    return pd.concat(all_stats, ignore_index=True)


def build_matchups(data):
    """Build canonical tournament matchups with Result column."""
    all_matchups = []
    for prefix in ["M", "W"]:
        tourn = data[prefix]["tourney"].copy()
        tourn["is_women"] = 1 if prefix == "W" else 0
        for _, row in tourn.iterrows():
            a, b = sorted([int(row.WTeamID), int(row.LTeamID)])
            result = 1.0 if int(row.WTeamID) == a else 0.0
            all_matchups.append({
                "Season": row.Season, "TeamA": a, "TeamB": b,
                "Result": result, "is_women": row.is_women,
            })
    return pd.DataFrame(all_matchups)


# Raddar feature columns (per team)
_TEAM_STATS = [
    "avg_Score", "avg_FGA", "avg_OR", "avg_DR", "avg_Blk", "avg_PF",
    "avg_opponent_FGA", "avg_opponent_Blk", "avg_opponent_PF", "avg_PointDiff",
]

# Full 29 delta feature names
FEATURE_NAMES = (
    ["Seed_diff"]
    + [f"T1_{c}" for c in _TEAM_STATS]
    + [f"T2_{c}" for c in _TEAM_STATS]
    + ["elo_diff", "T1_quality", "T2_quality"]
)


def build_features(matchups, team_stats, seeds, elo_ratings):
    """
    Build the 29-feature Raddar-style feature matrix.

    Features:
      1. Seed_diff
      2-11. T1 (TeamA) box-score averages (10 features)
      12-21. T2 (TeamB) box-score averages (10 features)
      22. elo_diff
      23. T1_quality (TeamA win rate)
      24. T2_quality (TeamB win rate)
    """
    # Merge seeds
    df = matchups.merge(
        seeds.rename(columns={"TeamID": "TeamA", "SeedNum": "SeedA"}),
        on=["Season", "TeamA"], how="left",
    ).merge(
        seeds.rename(columns={"TeamID": "TeamB", "SeedNum": "SeedB"}),
        on=["Season", "TeamB"], how="left",
    )
    df["Seed_diff"] = df["SeedA"] - df["SeedB"]

    # Merge team stats for TeamA
    t1_rename = {c: f"T1_{c}" for c in _TEAM_STATS + ["Quality"]}
    df = df.merge(
        team_stats[["Season", "TeamID"] + _TEAM_STATS + ["Quality"]].rename(
            columns={"TeamID": "TeamA", **t1_rename}
        ), on=["Season", "TeamA"], how="left",
    )

    # Merge team stats for TeamB
    t2_rename = {c: f"T2_{c}" for c in _TEAM_STATS + ["Quality"]}
    df = df.merge(
        team_stats[["Season", "TeamID"] + _TEAM_STATS + ["Quality"]].rename(
            columns={"TeamID": "TeamB", **t2_rename}
        ), on=["Season", "TeamB"], how="left",
    )

    # Merge Elo
    df = df.merge(
        elo_ratings.rename(columns={"TeamID": "TeamA", "Elo": "EloA"}),
        on=["Season", "TeamA"], how="left",
    ).merge(
        elo_ratings.rename(columns={"TeamID": "TeamB", "Elo": "EloB"}),
        on=["Season", "TeamB"], how="left",
    )
    df["elo_diff"] = df["EloA"] - df["EloB"]
    df["T1_quality"] = df["T1_Quality"]
    df["T2_quality"] = df["T2_Quality"]

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5.  MODEL: XGBoost Regressor with Raddar Params
# ═══════════════════════════════════════════════════════════════════════════

def raddar_xgb_model():
    """XGBoost regressor with Raddar's 'winning' hyperparameters."""
    return xgb.XGBRegressor(
        eta=0.01,
        subsample=0.6,
        max_depth=4,
        num_parallel_tree=2,
        objective="reg:squarederror",
        n_estimators=1000,
        verbosity=0,
        random_state=42,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6.  HOLDOUT EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def run_holdout(df, train_seasons, holdout_seasons):
    """
    Train on train_seasons, predict on holdout_seasons.
    Returns holdout DataFrame with predictions.
    """
    train = df[df.Season.isin(train_seasons)].dropna(subset=FEATURE_NAMES)
    holdout = df[df.Season.isin(holdout_seasons)].dropna(subset=FEATURE_NAMES)

    print(f"  Train: {len(train):,} games ({min(train_seasons)}–{max(train_seasons)})")
    print(f"  Holdout: {len(holdout):,} games ({holdout_seasons})")

    X_train = train[FEATURE_NAMES].values
    y_train = train["Result"].values
    X_holdout = holdout[FEATURE_NAMES].values

    model = raddar_xgb_model()
    model.fit(X_train, y_train)

    holdout = holdout.copy()
    holdout["Pred"] = model.predict(X_holdout).clip(0.025, 0.975)

    return holdout, model


# ═══════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  RADDAR BASELINE — XGBoost Regressor")
    print("  29 features, train 2003-2021, test 2022-2025")
    print("=" * 60)

    # Load data
    print("\n── Loading data ──")
    data = load_data()

    # Parse seeds
    print("── Parsing seeds ──")
    seeds = parse_seeds(data)
    print(f"  {len(seeds):,} seed entries")

    # Compute Elo
    print("── Computing Elo ratings ──")
    elo = compute_elo(data)
    print(f"  {len(elo):,} team-season Elo ratings")

    # Build team stats
    print("── Building team stats ──")
    team_stats = build_team_stats(data)
    print(f"  {len(team_stats):,} team-seasons")

    # Build matchups
    print("── Building tournament matchups ──")
    matchups = build_matchups(data)
    matchups = matchups[matchups.Season.isin(ALL_SEASONS)]
    print(f"  {len(matchups):,} matchups")

    # Build features
    print("── Building 29-feature matrix ──")
    df = build_features(matchups, team_stats, seeds, elo)
    n_missing = df[FEATURE_NAMES].isna().any(axis=1).sum()
    print(f"  {len(df):,} matchups, {n_missing} with missing features")
    print(f"  Features: {FEATURE_NAMES}")

    # Holdout evaluation
    print("\n── Running holdout evaluation ──")
    holdout, model = run_holdout(df, TRAIN_SEASONS, HOLDOUT_SEASONS)

    # Score
    brier = brier_score_loss(holdout.Result, holdout.Pred)
    ll = log_loss(holdout.Result, holdout.Pred)
    holdout["Correct"] = ((holdout.Pred > 0.5).astype(int) == holdout.Result).astype(int)
    acc = holdout.Correct.mean()

    print("\n" + "=" * 60)
    print("  RADDAR BASELINE HOLDOUT RESULTS")
    print("=" * 60)
    print(f"\n  Overall Brier:      {brier:.4f}")
    print(f"  Overall Log Loss:   {ll:.4f}")
    print(f"  Overall Accuracy:   {acc:.1%} ({holdout.Correct.sum()}/{len(holdout)})")

    # Per season
    print(f"\n{'─'*55}")
    print(f"  {'Season':<8} {'Brier':>10} {'Log Loss':>10} {'Accuracy':>10} {'N':>6}")
    print(f"{'─'*55}")
    for s in sorted(holdout.Season.unique()):
        m = holdout[holdout.Season == s]
        s_bs = brier_score_loss(m.Result, m.Pred)
        s_ll = log_loss(m.Result, m.Pred)
        s_acc = m.Correct.mean()
        print(f"  {s:<8} {s_bs:>10.4f} {s_ll:>10.4f} {s_acc:>10.1%} {len(m):>6}")

    # By gender
    print(f"\n{'─'*55}")
    for label, is_w in [("Men's", 0), ("Women's", 1)]:
        m = holdout[holdout.is_women == is_w]
        if len(m) > 0:
            g_bs = brier_score_loss(m.Result, m.Pred)
            g_ll = log_loss(m.Result, m.Pred)
            g_acc = m.Correct.mean()
            print(f"  {label:<10} Brier={g_bs:.4f}  LL={g_ll:.4f}  "
                  f"Acc={g_acc:.1%}  ({len(m)} games)")

    # Comparison
    print(f"\n{'═'*60}")
    print(f"  COMPARISON")
    print(f"{'═'*60}")
    print(f"  GNN Ensemble:    Brier=0.1711  LL=0.5077  Acc=74.3%")
    print(f"  Raddar Baseline: Brier={brier:.4f}  LL={ll:.4f}  Acc={acc:.1%}")
    bs_delta = brier - 0.1711
    print(f"  Delta:           Brier={bs_delta:+.4f}")

    # Save results to text file
    results_path = OUTPUT_DIR / "holdout_results.txt"
    with open(results_path, "w") as f:
        f.write("RADDAR BASELINE — HOLDOUT RESULTS (2022-2025)\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Brier Score:  {brier:.6f}\n")
        f.write(f"Log Loss:     {ll:.6f}\n")
        f.write(f"Accuracy:     {acc:.4f} ({holdout.Correct.sum()}/{len(holdout)})\n\n")
        f.write(f"Per-Season:\n")
        for s in sorted(holdout.Season.unique()):
            m = holdout[holdout.Season == s]
            f.write(f"  {s}: Brier={brier_score_loss(m.Result, m.Pred):.4f}  "
                    f"LL={log_loss(m.Result, m.Pred):.4f}  "
                    f"Acc={m.Correct.mean():.1%}  (N={len(m)})\n")
        f.write(f"\nComparison:\n")
        f.write(f"  GNN Ensemble:    Brier=0.1711  LL=0.5077  Acc=74.3%\n")
        f.write(f"  Raddar Baseline: Brier={brier:.4f}  LL={ll:.4f}  Acc={acc:.1%}\n")
        f.write(f"  Delta:           Brier={bs_delta:+.4f}\n")

    print(f"\n  ✓ Results saved → {results_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
