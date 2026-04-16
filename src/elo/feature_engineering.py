"""
Elo Feature Engineering — MOVDA (Margin of Victory Differential Analysis).

Advanced Elo rating system with:
  - Expected Margin of Victory (EMOV) via scaled tanh with home-court advantage
  - Rating updates driven by TMOV − EMOV differential
  - Dynamic K-factor: high early-season for rapid convergence, low late-season
  - α, β, δ parameters fit via scipy curve_fit on historical data
  - Season-to-season regression toward the mean

Per-team feature: Elo_Final (end-of-season MOV-adjusted Elo)
Delta feature for matchups: Delta_Elo_Final

TEMPORAL FIREWALL: Only regular-season games (Day ≤ 132) are used.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit

from src.config import (
    RAW_DIR, PROCESSED_DIR, ELO_FEATURES_FILE,
    REGULAR_SEASON_CUTOFF,
    FIRST_DETAILED_SEASON, LAST_HISTORICAL_SEASON, PREDICTION_SEASON,
    TRAIN_SEASONS,
    ELO_INIT, ELO_K, ELO_K_DECAY, ELO_REGRESS, ELO_LAMBDA,
    ELO_MOVDA_ALPHA, ELO_MOVDA_BETA, ELO_MOVDA_DELTA,
    FIRST_COMPACT_SEASON_M, FIRST_COMPACT_SEASON_W,
)
from src.data_loader import load_regular_season_compact, load_tourney_labels


# ═══════════════════════════════════════════════════════════════════════════
# 1.  MOVDA PARAMETER FITTING
# ═══════════════════════════════════════════════════════════════════════════

def _emov_model(X, alpha, beta, delta_hca):
    """
    Expected Margin of Victory model for curve_fit.

    Parameters
    ----------
    X : array of shape (N, 2) — [delta_elo, home_indicator]
    alpha : asymptotic max margin from skill differential
    beta : steepness near Δelo=0
    delta_hca : home-court advantage in points

    Returns
    -------
    array of expected margins
    """
    delta_elo = X[0]
    home_ind = X[1]
    return alpha * np.tanh(beta * delta_elo) + delta_hca * home_ind


def fit_movda_params(
    compact: pd.DataFrame,
    init: float = ELO_INIT,
    k: float = 20.0,
    regress: float = 0.6,
) -> tuple[float, float, float]:
    """
    Fit MOVDA parameters (α, β, δ) via non-linear least squares.

    Runs a simple Elo pass (no MOV) to get rating gaps, then fits
    the EMOV model to predict actual margins from (Δelo, home_indicator).

    Parameters
    ----------
    compact : compact regular-season results (must have WLoc column)
    init, k, regress : Elo parameters for the preliminary rating pass

    Returns
    -------
    (alpha, beta, delta_hca) — fitted MOVDA parameters
    """
    # Step 1: Run a basic Elo pass to get pre-game rating diffs
    elo = defaultdict(lambda: init)
    records = []

    games = compact[compact.DayNum <= REGULAR_SEASON_CUTOFF].copy()
    games = games.sort_values(["Season", "DayNum"]).reset_index(drop=True)

    prev_season = None
    for _, row in games.iterrows():
        season = int(row.Season)
        if season != prev_season and prev_season is not None:
            for tid in list(elo.keys()):
                elo[tid] = init + regress * (elo[tid] - init)
        prev_season = season

        w, l = int(row.WTeamID), int(row.LTeamID)
        margin = int(row.WScore - row.LScore)
        wloc = row.WLoc if "WLoc" in row.index else "N"

        # Home indicator: +1 if winner is home, -1 if away, 0 if neutral
        home_ind = 1.0 if wloc == "H" else (-1.0 if wloc == "A" else 0.0)

        # Record pre-game state (from winner's perspective)
        delta_elo = elo[w] - elo[l]
        records.append((delta_elo, home_ind, margin))

        # Simple Elo update for this pass
        exp_w = 1.0 / (1.0 + 10.0 ** ((elo[l] - elo[w]) / 400.0))
        elo[w] += k * (1.0 - exp_w)
        elo[l] += k * (0.0 - (1.0 - exp_w))

    # Step 2: Fit EMOV model via curve_fit
    data = np.array(records)
    X_fit = np.array([data[:, 0], data[:, 1]])  # (delta_elo, home_ind)
    y_fit = data[:, 2]  # actual margins

    # Initial guesses and bounds
    p0 = [12.0, 0.003, 3.5]
    bounds = ([2.0, 0.0005, 0.5], [80.0, 0.02, 10.0])

    popt, _ = curve_fit(_emov_model, X_fit, y_fit, p0=p0, bounds=bounds,
                        maxfev=10000)
    alpha, beta, delta_hca = popt

    print(f"  MOVDA curve_fit: α={alpha:.2f}, β={beta:.5f}, δ={delta_hca:.2f}")

    return alpha, beta, delta_hca


# ═══════════════════════════════════════════════════════════════════════════
# 2.  ELO ENGINE WITH MOVDA
# ═══════════════════════════════════════════════════════════════════════════

def _expected_score(elo_a: float, elo_b: float) -> float:
    """Expected win probability for team A given Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def compute_elo_ratings(
    compact: pd.DataFrame,
    seasons: list[int],
    k: float = ELO_K,
    k_decay: float = ELO_K_DECAY,
    init: float = ELO_INIT,
    regress: float = ELO_REGRESS,
    lambda_mov: float = ELO_LAMBDA,
    alpha: float = ELO_MOVDA_ALPHA,
    beta: float = ELO_MOVDA_BETA,
    delta_hca: float = ELO_MOVDA_DELTA,
) -> pd.DataFrame:
    """
    Compute iterative Elo ratings with MOVDA across multiple seasons.

    Parameters
    ----------
    compact : DataFrame of compact regular-season results
    seasons : list of seasons to process (must be sorted ascending)
    k : base K-factor (decays over the season)
    k_decay : exponential decay rate for K over the season
    init : initial Elo rating
    regress : season-to-season regression factor toward init
    lambda_mov : weight for the margin-of-victory differential term
    alpha, beta, delta_hca : MOVDA parameters (from fit_movda_params)

    Returns
    -------
    DataFrame with columns: Season, TeamID, Elo_Final
    """
    elo = defaultdict(lambda: init)
    results = []

    for season in sorted(seasons):
        # ── Season-to-season regression ───────────────────────────────
        for team_id in list(elo.keys()):
            elo[team_id] = init + regress * (elo[team_id] - init)

        # Get this season's games, sorted by day
        games = compact[
            (compact.Season == season) &
            (compact.DayNum <= REGULAR_SEASON_CUTOFF)
        ].sort_values("DayNum")

        if len(games) == 0:
            continue

        # Find max day for K-decay normalization
        max_day = games.DayNum.max()

        for _, row in games.iterrows():
            w, l = int(row.WTeamID), int(row.LTeamID)
            margin = int(row.WScore - row.LScore)
            day = int(row.DayNum)
            wloc = row.WLoc if "WLoc" in games.columns else "N"

            # Dynamic K-factor: decays exponentially over the season
            day_frac = day / max_day if max_day > 0 else 1.0
            k_game = k * np.exp(-k_decay * day_frac)

            # Standard Elo expected score (winner's perspective)
            exp_w = _expected_score(elo[w], elo[l])

            # MOVDA: Expected Margin of Victory
            delta_elo_wl = elo[w] - elo[l]
            home_ind = 1.0 if wloc == "H" else (-1.0 if wloc == "A" else 0.0)
            emov = alpha * np.tanh(beta * delta_elo_wl) + delta_hca * home_ind

            # True Margin of Victory
            tmov = float(margin)

            # Margin differential
            margin_diff = tmov - emov

            # Combined update: standard Elo + MOV differential
            # Winner update
            elo_change = k_game * (1.0 - exp_w) + lambda_mov * margin_diff
            elo[w] += elo_change
            elo[l] -= elo_change

        # Record final Elo for all teams active this season
        active_teams = set(games.WTeamID.unique()) | set(games.LTeamID.unique())
        for team_id in active_teams:
            team_id = int(team_id)
            results.append({
                "Season": season,
                "TeamID": team_id,
                "Elo_Final": elo[team_id],
            })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  SEASON-LEVEL PROFILES
# ═══════════════════════════════════════════════════════════════════════════

def build_elo_profiles(
    seasons: list[int] | None = None,
    k: float = ELO_K,
    k_decay: float = ELO_K_DECAY,
    init: float = ELO_INIT,
    regress: float = ELO_REGRESS,
    lambda_mov: float = ELO_LAMBDA,
    alpha: float | None = None,
    beta: float | None = None,
    delta_hca: float | None = None,
) -> pd.DataFrame:
    """
    Build Elo-based features for all teams across seasons (M + W).
    Uses compact results going back to 1985 (Men) / 1998 (Women).

    If alpha/beta/delta_hca are None, will fit MOVDA params from data.
    """
    if seasons is None:
        seasons = TRAIN_SEASONS + [PREDICTION_SEASON]

    all_profiles = []

    for prefix, is_w, first_season in [
        ("M", 0, FIRST_COMPACT_SEASON_M),
        ("W", 1, FIRST_COMPACT_SEASON_W),
    ]:
        compact = load_regular_season_compact(prefix)

        # Fit MOVDA params if not provided
        movda_alpha = alpha
        movda_beta = beta
        movda_delta = delta_hca
        if movda_alpha is None or movda_beta is None or movda_delta is None:
            print(f"  Fitting MOVDA params for {prefix} data …")
            movda_alpha, movda_beta, movda_delta = fit_movda_params(compact)

        # Compute Elo from the earliest available season to build up ratings
        all_seasons = sorted(compact.Season.unique())
        warmup_seasons = [s for s in all_seasons if s >= first_season]

        print(f"  Computing {prefix} MOVDA-Elo ({len(warmup_seasons)} seasons, "
              f"{first_season}–{max(warmup_seasons)}) …")
        elo_df = compute_elo_ratings(
            compact, warmup_seasons,
            k=k, k_decay=k_decay, init=init, regress=regress,
            lambda_mov=lambda_mov,
            alpha=movda_alpha, beta=movda_beta, delta_hca=movda_delta,
        )

        # Filter to only requested seasons
        elo_df = elo_df[elo_df.Season.isin(seasons)]
        elo_df["is_women"] = is_w
        all_profiles.append(elo_df)

    profiles = pd.concat(all_profiles, ignore_index=True)
    return profiles


# ═══════════════════════════════════════════════════════════════════════════
# 4.  DELTA FEATURES
# ═══════════════════════════════════════════════════════════════════════════

_ELO_COLS = ["Elo_Final"]

ELO_FEATURE_NAMES = [f"Delta_{c}" for c in _ELO_COLS]


def _merge_elo_to_matchups(
    matchups: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Elo profiles → matchups and compute deltas."""
    # Team A
    a_rename = {c: f"A_{c}" for c in _ELO_COLS}
    matchups = matchups.merge(
        profiles[["Season", "TeamID"] + _ELO_COLS].rename(
            columns={"TeamID": "TeamA", **a_rename}
        ),
        on=["Season", "TeamA"],
        how="left",
    )

    # Team B
    b_rename = {c: f"B_{c}" for c in _ELO_COLS}
    matchups = matchups.merge(
        profiles[["Season", "TeamID"] + _ELO_COLS].rename(
            columns={"TeamID": "TeamB", **b_rename}
        ),
        on=["Season", "TeamB"],
        how="left",
    )

    # Deltas
    for col in _ELO_COLS:
        matchups[f"Delta_{col}"] = matchups[f"A_{col}"] - matchups[f"B_{col}"]

    return matchups


# ═══════════════════════════════════════════════════════════════════════════
# 5.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def build_elo_features() -> pd.DataFrame:
    """
    End-to-end Elo pipeline: fit MOVDA params, compute ratings,
    merge with tournament labels, compute deltas, save to parquet.
    """
    print("▶ Building MOVDA-Elo profiles …")
    profiles = build_elo_profiles()
    print(f"  {len(profiles):,} team-seasons with Elo ratings")

    print("▶ Loading tournament labels …")
    labels = load_tourney_labels()

    print("▶ Computing Elo delta features …")
    features = _merge_elo_to_matchups(labels, profiles)

    n_missing = features[ELO_FEATURE_NAMES].isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"  ⚠ {n_missing} matchups missing Elo features. Dropping.")
        features = features.dropna(subset=ELO_FEATURE_NAMES)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(ELO_FEATURES_FILE, index=False)
    print(f"  Saved {len(features):,} matchups → {ELO_FEATURES_FILE}")

    return features


# ── Quick Sanity Check ───────────────────────────────────────────────────

if __name__ == "__main__":
    features = build_elo_features()
    print(f"\nShape: {features.shape}")
    print(f"\nDelta feature ranges:")
    for col in ELO_FEATURE_NAMES:
        if col in features.columns:
            print(f"  {col:25s}  "
                  f"mean={features[col].mean():+.1f}  "
                  f"std={features[col].std():.1f}  "
                  f"[{features[col].min():+.0f}, {features[col].max():+.0f}]")
    print(f"\nResult balance: {features.Result.mean():.3f}")
    print(f"Coverage: {features.Season.min()}–{features.Season.max()} "
          f"({features.Season.nunique()} seasons)")
    print(f"Men's: {(features.is_women == 0).sum():,} | "
          f"Women's: {(features.is_women == 1).sum():,}")
