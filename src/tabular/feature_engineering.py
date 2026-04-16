"""
Tabular Feature Engineering — Opponent-Adjusted Tempo-Free Efficiencies

  1. Aggregate regular-season detailed box scores into per-team season profiles
     (ONLY days ≤ 132 — the temporal firewall).
  2. Compute tempo-free per-game metrics (pts/100 possessions).
  3. Iterative opponent adjustment (KenPom-style):
     adjust each team's efficiency for the quality of opponents faced.
  4. Output: AdjOE, AdjDE, AdjEM + variance & recent form.
  5. Join seeds, produce symmetrical delta features for every matchup.
"""

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
from tqdm import tqdm

from src.config import (
    RAW_DIR, PROCESSED_DIR, TABULAR_FEATURES_FILE,
    REGULAR_SEASON_CUTOFF, FIRST_DETAILED_SEASON, PREDICTION_SEASON,
    TRAIN_SEASONS,
)
from src.data_loader import load_regular_season_detailed, load_tourney_labels
from src.tabular.seeds import build_seed_lookup


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PER-GAME STATS  (one row per team per game, both W & L perspectives)
# ═══════════════════════════════════════════════════════════════════════════

def _possession_estimate(fga, fta, orb, to, opp_drb):
    """
    Simple possession estimate (per-game, one side).
    Poss ≈ FGA + 0.44 * FTA − ORB + TO
    """
    return fga + 0.44 * fta - orb + to


def _unstack_games(detailed: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the winner/loser format into a team-centric format.

    Each game produces TWO rows: one from the winner's perspective and
    one from the loser's perspective (Symmetry requirement §1).

    Returns
    -------
    DataFrame with columns:
        Season, DayNum, TeamID, OppID,
        FGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk, PF,
        Opp_FGM, Opp_FGA, ..., Score, OppScore, Win
    """
    # Enforce the temporal firewall
    detailed = detailed[detailed["DayNum"] <= REGULAR_SEASON_CUTOFF].copy()

    stat_cols = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                 "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

    w_cols = {f"W{c}": c for c in stat_cols}
    l_cols = {f"L{c}": c for c in stat_cols}
    w_opp  = {f"L{c}": f"Opp_{c}" for c in stat_cols}
    l_opp  = {f"W{c}": f"Opp_{c}" for c in stat_cols}

    base = ["Season", "DayNum"]

    # Winner's row
    has_wloc = "WLoc" in detailed.columns
    win_extra = ["WLoc"] if has_wloc else []
    win_df = detailed[base + ["WTeamID", "LTeamID", "WScore", "LScore"]
                      + list(w_cols.keys()) + list(w_opp.keys()) + win_extra].copy()
    win_df = win_df.rename(columns={
        "WTeamID": "TeamID", "LTeamID": "OppID",
        "WScore": "Score", "LScore": "OppScore",
        **w_cols, **w_opp
    })
    win_df["Win"] = 1
    # Home indicator: +1 if this team played at home, -1 away, 0 neutral
    if has_wloc:
        win_df["Home"] = win_df["WLoc"].map({"H": 1.0, "A": -1.0, "N": 0.0}).fillna(0.0)
        win_df = win_df.drop(columns=["WLoc"])
    else:
        win_df["Home"] = 0.0

    # Loser's row
    loss_df = detailed[base + ["LTeamID", "WTeamID", "LScore", "WScore"]
                       + list(l_cols.keys()) + list(l_opp.keys()) + win_extra].copy()
    loss_df = loss_df.rename(columns={
        "LTeamID": "TeamID", "WTeamID": "OppID",
        "LScore": "Score", "WScore": "OppScore",
        **l_cols, **l_opp
    })
    loss_df["Win"] = 0
    # Loser's home indicator is inverted from WLoc
    if has_wloc:
        loss_df["Home"] = loss_df["WLoc"].map({"H": -1.0, "A": 1.0, "N": 0.0}).fillna(0.0)
        loss_df = loss_df.drop(columns=["WLoc"])
    else:
        loss_df["Home"] = 0.0

    games = pd.concat([win_df, loss_df], ignore_index=True)
    games = games.sort_values(["Season", "TeamID", "DayNum"]).reset_index(drop=True)
    return games


def _add_per_game_metrics(games: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game advanced metrics on the unstacked DataFrame."""
    g = games

    # Possessions (for offensive efficiency)
    g["Poss"] = _possession_estimate(g.FGA, g.FTA, g.OR, g.TO, g.Opp_DR)
    g["Opp_Poss"] = _possession_estimate(g.Opp_FGA, g.Opp_FTA, g.Opp_OR, g.Opp_TO, g.DR)

    # Offensive efficiency (points per 100 possessions) — used for variance
    g["Off_Eff"] = np.where(g.Poss > 0, g.Score / g.Poss * 100, 0)

    # Four Factors (per-game)
    g["eFG"]  = np.where(g.FGA > 0, (g.FGM + 0.5 * g.FGM3) / g.FGA, 0)
    g["TOV"]  = np.where(g.Poss > 0, g.TO / g.Poss, 0)
    g["ORB"]  = np.where((g.OR + g.Opp_DR) > 0, g.OR / (g.OR + g.Opp_DR), 0)
    g["FTR"]  = np.where(g.FGA > 0, g.FTM / g.FGA, 0)

    # Madness metrics (per-game)
    g["ThreePAr"]  = np.where(g.FGA > 0, g.FGA3 / g.FGA, 0)          # 3-point attempt rate
    g["Def_TOV"]   = np.where(g.Opp_Poss > 0, g.Opp_TO / g.Opp_Poss, 0)  # Chaos creation

    # Advanced rate metrics (per-game)
    g["AST_Rate"]  = np.where(g.FGM > 0, g.Ast / g.FGM, 0)           # Assist rate
    g["BLK_Rate"]  = np.where(g.Opp_FGA > 0, g.Blk / g.Opp_FGA, 0)  # Block rate
    g["STL_Rate"]  = np.where(g.Opp_Poss > 0, g.Stl / g.Opp_Poss, 0) # Steal rate
    g["Foul_Rate"] = np.where(g.Opp_Poss > 0, g.PF / g.Opp_Poss, 0)  # Foul rate

    # Defensive efficiency (per-game)
    g["Def_Eff"] = np.where(g.Opp_Poss > 0, g.OppScore / g.Opp_Poss * 100, 0)

    # 3-point shooting percentage (per game)
    g["FGP3"] = np.where(g.FGA3 > 0, g.FGM3 / g.FGA3, 0)

    # Margin
    g["Margin"] = g.Score - g.OppScore

    return g


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SEASON AGGREGATION  (per team per season)
# ═══════════════════════════════════════════════════════════════════════════

def _aggregate_season_profiles(games: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-game stats → season-level team profiles.

    Computes raw Off_Eff/Def_Eff means (later opponent-adjusted),
    plus variance and recent form metrics.
    """
    agg = games.groupby(["Season", "TeamID"]).agg(
        # Counts
        GP        = ("Win",    "count"),
        Wins      = ("Win",    "sum"),

        # Raw tempo-free efficiencies (pts/100 poss)
        Off_Eff_mean = ("Off_Eff", "mean"),
        Def_Eff_mean = ("Def_Eff", "mean"),

        # Variance (game-to-game consistency)
        Off_Eff_std  = ("Off_Eff",  "std"),
    ).reset_index()

    agg["WinPct"] = agg.Wins / agg.GP
    agg["Off_Eff_std"] = agg["Off_Eff_std"].fillna(0)

    # 3-point shooting volatility (game-to-game std of 3P%)
    threep_std = games.groupby(["Season", "TeamID"]).agg(
        ThreeP_Volatility = ("FGP3", "std"),
    ).reset_index()
    threep_std["ThreeP_Volatility"] = threep_std["ThreeP_Volatility"].fillna(0)
    agg = agg.merge(threep_std, on=["Season", "TeamID"], how="left")

    # Late-season Net Efficiency Margin (last 30 days of regular season)
    # DayNum ≤ 132 is the cutoff; last 30 days = DayNum > 102
    late = games[games.DayNum > 102].copy()
    if len(late) > 0:
        late_agg = late.groupby(["Season", "TeamID"]).agg(
            Late_Off_Eff = ("Off_Eff", "mean"),
            Late_Def_Eff = ("Def_Eff", "mean"),
        ).reset_index()
        late_agg["Late_NetEM"] = late_agg["Late_Off_Eff"] - late_agg["Late_Def_Eff"]
        agg = agg.merge(
            late_agg[["Season", "TeamID", "Late_NetEM"]],
            on=["Season", "TeamID"], how="left"
        )
    else:
        agg["Late_NetEM"] = np.nan
    agg["Late_NetEM"] = agg["Late_NetEM"].fillna(agg.get("Off_Eff_mean", 0) - agg.get("Def_Eff_mean", 0))

    # ── WAB: Wins Above Bubble ──
    # For each team-season, calculate expected wins a bubble team (50th %-ile)
    # would get against that team's exact schedule, then WAB = actual - expected
    all_wab = []
    for (season,), grp in games.groupby(["Season"]):
        # Estimate each team's strength as WinPct (simple proxy)
        team_wp = grp.groupby("TeamID")["Win"].mean()
        # Bubble team expected win prob = 1 - opponent_strength
        # (a .500 team beats a .400 team ~60% of the time, etc.)
        for team_id, team_games in grp.groupby("TeamID"):
            opponents = team_games["OppID"].values
            opp_strengths = team_wp.reindex(opponents).fillna(0.5).values
            # Log5 formula: P(bubble wins) where bubble = 0.5
            # P = 0.5 * (1-opp) / (0.5*(1-opp) + opp*0.5) = (1-opp)
            # Simplified: expected wins for .500 team = sum(1 - opp_wp)
            expected_wins = (1 - opp_strengths).sum()
            actual_wins = team_games["Win"].sum()
            all_wab.append({
                "Season": season,
                "TeamID": team_id,
                "WAB": actual_wins - expected_wins,
            })
    wab_df = pd.DataFrame(all_wab)
    agg = agg.merge(wab_df, on=["Season", "TeamID"], how="left")
    agg["WAB"] = agg["WAB"].fillna(0)

    # Recent form: last 10 games per team per season
    recent = (games.sort_values(["Season", "TeamID", "DayNum"])
              .groupby(["Season", "TeamID"]).tail(10))
    recent_agg = recent.groupby(["Season", "TeamID"]).agg(
        Recent_WinPct  = ("Win",    "mean"),
        Recent_Margin  = ("Margin", "mean"),
    ).reset_index()
    agg = agg.merge(recent_agg, on=["Season", "TeamID"], how="left")

    return agg


# ═══════════════════════════════════════════════════════════════════════════
# 2b.  RIDGE REGRESSION OPPONENT ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════════

# Metrics to opponent-adjust via Ridge: (game_col, off_coef_name, def_coef_name)
_RIDGE_METRICS = [
    ("Off_Eff",  "AdjO",        "AdjD"),         # Offensive/defensive efficiency
    ("eFG",      "Adj_eFG_O",   "Adj_eFG_D"),    # Effective FG%
    ("TOV",      "Adj_TOV_O",   "Adj_TOV_D"),    # Turnover rate
    ("ORB",      "Adj_ORB_O",   "Adj_ORB_D"),    # Offensive rebound rate
    ("FTR",      "Adj_FTR_O",   "Adj_FTR_D"),    # Free throw rate
]


def _ridge_adjust_single_metric(
    games_season: pd.DataFrame,
    metric_col: str,
    team_ids: np.ndarray,
    alpha: float = 1.0,
) -> tuple[dict, dict, float]:
    """
    Solve: y = μ + α_i (offense) + β_j (defense) + δ*home + ε
    via Ridge Regression for a single season and metric.

    Returns (off_coefs, def_coefs, hca) where:
    - off_coefs[team_id] = team's offensive contribution relative to mean
    - def_coefs[team_id] = team's defensive concession relative to mean
    - hca = home-court advantage
    """
    n_teams = len(team_ids)
    team_to_idx = {tid: i for i, tid in enumerate(team_ids)}

    n_games = len(games_season)
    if n_games == 0:
        empty = {tid: 0.0 for tid in team_ids}
        return empty, empty, 0.0

    # Build sparse design matrix: [off_team_dummies | def_team_dummies | home]
    # Shape: (n_games, 2*n_teams + 1)
    rows, cols, vals = [], [], []

    y = games_season[metric_col].values.astype(float)
    team_col = games_season["TeamID"].values
    opp_col = games_season["OppID"].values

    # Determine home indicator from original game data
    # In our unstacked format: for the winner row, WLoc applies directly
    # For the loser row, WLoc is from winner's perspective (inverted)
    # We stored Win=1/0, so we can reconstruct:
    # If Win=1 and we're the winner: home_ind from WLoc column if available
    # Simplified: use a home column if it exists, else 0
    has_home = "Home" in games_season.columns
    home_vals = games_season["Home"].values if has_home else np.zeros(n_games)

    for i in range(n_games):
        off_idx = team_to_idx.get(int(team_col[i]))
        def_idx = team_to_idx.get(int(opp_col[i]))
        if off_idx is None or def_idx is None:
            continue

        # Offense dummy
        rows.append(i); cols.append(off_idx); vals.append(1.0)
        # Defense dummy
        rows.append(i); cols.append(n_teams + def_idx); vals.append(1.0)
        # Home indicator
        rows.append(i); cols.append(2 * n_teams); vals.append(float(home_vals[i]))

    X = sparse.csr_matrix((vals, (rows, cols)), shape=(n_games, 2 * n_teams + 1))

    # Fit Ridge (fit_intercept=True captures μ, the league mean)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X, y)

    coefs = ridge.coef_
    off_coefs = {tid: float(coefs[team_to_idx[tid]]) for tid in team_ids}
    def_coefs = {tid: float(coefs[n_teams + team_to_idx[tid]]) for tid in team_ids}
    hca = float(coefs[2 * n_teams])

    return off_coefs, def_coefs, hca


def _ridge_opponent_adjust(
    games: pd.DataFrame,
    profiles: pd.DataFrame,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Ridge Regression opponent adjustment for multiple metrics.

    For each season and each metric, builds a sparse linear model:
      metric_value = μ + α_i (offense) + β_j (defense) + δ*home
    and solves via Ridge to get true adjusted offensive and defensive
    coefficients for every team.

    Parameters
    ----------
    games : per-game DataFrame (from _add_per_game_metrics)
    profiles : season profiles (from _aggregate_season_profiles)
    alpha : Ridge regularization strength (L2 penalty)

    Returns
    -------
    profiles with new columns: AdjO, AdjD, NetEM, plus Four Factor adjustments
    """
    profiles = profiles.copy()

    # Initialize all output columns
    for _, off_name, def_name in _RIDGE_METRICS:
        profiles[off_name] = 0.0
        profiles[def_name] = 0.0

    for season in sorted(profiles.Season.unique()):
        season_games = games[games.Season == season].copy()
        season_teams = profiles[profiles.Season == season]["TeamID"].values
        mask = profiles.Season == season

        if len(season_games) == 0:
            continue

        for metric_col, off_name, def_name in _RIDGE_METRICS:
            off_c, def_c, hca = _ridge_adjust_single_metric(
                season_games, metric_col, season_teams, alpha=alpha
            )

            # Store coefficients: intercept (league mean) + team adjustment
            # For offense: higher = better (more pts/100 poss scored)
            # For defense: higher = worse (more pts/100 poss allowed)
            profiles.loc[mask, off_name] = profiles.loc[mask, "TeamID"].map(off_c)
            profiles.loc[mask, def_name] = profiles.loc[mask, "TeamID"].map(def_c)

    # Net Efficiency Margin = AdjO - AdjD (higher = better)
    profiles["NetEM"] = profiles["AdjO"] - profiles["AdjD"]

    return profiles


# ═══════════════════════════════════════════════════════════════════════════
# 3.  MATCHUP FEATURES  (Cross-matched offense vs defense interactions)
# ═══════════════════════════════════════════════════════════════════════════

# Per-team columns needed from profiles for matchup feature construction
_PROFILE_COLS = [
    "AdjO", "AdjD", "NetEM",
    "Adj_eFG_O", "Adj_eFG_D",
    "Adj_TOV_O", "Adj_TOV_D",
    "Adj_ORB_O", "Adj_ORB_D",
    "Adj_FTR_O", "Adj_FTR_D",
    "Off_Eff_std",
]


def _merge_profiles_to_matchups(
    matchups: pd.DataFrame,
    profiles: pd.DataFrame,
    seed_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each (Season, TeamA, TeamB) matchup, merge team profiles and seeds,
    then compute cross-matched matchup interaction features.

    Instead of flat deltas (A_stat - B_stat), we cross-match:
      - Team A's offense vs Team B's defense (and vice versa)
      - Multiplication-based vulnerability/mismatch signals
    """
    # Merge Team A profile
    a_cols = {c: f"A_{c}" for c in _PROFILE_COLS}
    matchups = matchups.merge(
        profiles[["Season", "TeamID"] + _PROFILE_COLS].rename(
            columns={"TeamID": "TeamA", **a_cols}
        ),
        on=["Season", "TeamA"],
        how="left",
    )

    # Merge Team B profile
    b_cols = {c: f"B_{c}" for c in _PROFILE_COLS}
    matchups = matchups.merge(
        profiles[["Season", "TeamID"] + _PROFILE_COLS].rename(
            columns={"TeamID": "TeamB", **b_cols}
        ),
        on=["Season", "TeamB"],
        how="left",
    )

    # Merge seeds
    matchups = matchups.merge(
        seed_lookup[["Season", "TeamID", "SeedNum"]].rename(
            columns={"TeamID": "TeamA", "SeedNum": "A_Seed"}
        ),
        on=["Season", "TeamA"],
        how="left",
    )
    matchups = matchups.merge(
        seed_lookup[["Season", "TeamID", "SeedNum"]].rename(
            columns={"TeamID": "TeamB", "SeedNum": "B_Seed"}
        ),
        on=["Season", "TeamB"],
        how="left",
    )

    # ── Macro delta features (kept as-is) ─────────────────────────────
    matchups["Delta_NetEM"] = matchups["A_NetEM"] - matchups["B_NetEM"]
    matchups["Delta_Off_Eff_std"] = matchups["A_Off_Eff_std"] - matchups["B_Off_Eff_std"]
    matchups["Delta_Seed"] = matchups["A_Seed"] - matchups["B_Seed"]
    
    # ── Group A: Advantage Deltas (offense vs opposing defense) ────────
    # TeamA's offense vs TeamB's defense
    matchups["TeamA_Off_Advantage"] = matchups["A_AdjO"] - matchups["B_AdjD"]
    matchups["TeamB_Off_Advantage"] = matchups["B_AdjO"] - matchups["A_AdjD"]

    matchups["TeamA_eFG_Advantage"] = matchups["A_Adj_eFG_O"] - matchups["B_Adj_eFG_D"]
    matchups["TeamB_eFG_Advantage"] = matchups["B_Adj_eFG_O"] - matchups["A_Adj_eFG_D"]

    matchups["TeamA_FTR_Advantage"] = matchups["A_Adj_FTR_O"] - matchups["B_Adj_FTR_D"]
    matchups["TeamB_FTR_Advantage"] = matchups["B_Adj_FTR_O"] - matchups["A_Adj_FTR_D"]

    # ── Group B: Vulnerability Multipliers (mismatch signals) ─────────
    # Does an elite crashing team face a weak rebounding block-out team?
    matchups["TeamA_ORB_Mismatch"] = matchups["A_Adj_ORB_O"] * matchups["B_Adj_ORB_D"]
    matchups["TeamB_ORB_Mismatch"] = matchups["B_Adj_ORB_O"] * matchups["A_Adj_ORB_D"]

    # Does an elite pressure team face a turnover-prone opponent?
    matchups["TeamA_TOV_Mismatch"] = matchups["A_Adj_TOV_D"] * matchups["B_Adj_TOV_O"]
    matchups["TeamB_TOV_Mismatch"] = matchups["B_Adj_TOV_D"] * matchups["A_Adj_TOV_O"]

    return matchups


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

    dfs = []
    for prefix, is_w in [("M", 0), ("W", 1)]:
        detailed = load_regular_season_detailed(prefix)
        detailed = detailed[detailed["Season"].isin(seasons)]
        games = _unstack_games(detailed)
        games = _add_per_game_metrics(games)
        profile = _aggregate_season_profiles(games)
        profile = _ridge_opponent_adjust(games, profile)
        print(f"  {prefix}: AdjO range=[{profile.AdjO.min():.2f}, {profile.AdjO.max():.2f}], "
              f"AdjD range=[{profile.AdjD.min():.2f}, {profile.AdjD.max():.2f}], "
              f"NetEM range=[{profile.NetEM.min():.1f}, {profile.NetEM.max():.1f}]")
        profile["is_women"] = is_w
        dfs.append(profile)

    profiles = pd.concat(dfs, ignore_index=True)

    return profiles


TABULAR_FEATURE_NAMES = [
    # Macro features
    "Delta_Seed",
    "Delta_NetEM",
    "Delta_Off_Eff_std",
    # Group A: Advantage Deltas (offense vs opposing defense)
    "TeamA_Off_Advantage",
    "TeamB_Off_Advantage",
    "TeamA_eFG_Advantage",
    "TeamB_eFG_Advantage",
    "TeamA_FTR_Advantage",
    "TeamB_FTR_Advantage",
    # Group B: Vulnerability Multipliers
    "TeamA_ORB_Mismatch",
    "TeamB_ORB_Mismatch",
    "TeamA_TOV_Mismatch",
    "TeamB_TOV_Mismatch",
]


def build_season_profiles(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Build per-team season profiles from detailed box scores (M + W).

    Parameters
    ----------
    seasons : list of int, optional
        Seasons to process.  Defaults to TRAIN_SEASONS + [PREDICTION_SEASON].

    Returns
    -------
    pd.DataFrame  (Season, TeamID, eFG_mean, …, WinPct, is_women)
    """
    if seasons is None:
        seasons = TRAIN_SEASONS + [PREDICTION_SEASON]

    dfs = []
    for prefix, is_w in [("M", 0), ("W", 1)]:
        detailed = load_regular_season_detailed(prefix)
        detailed = detailed[detailed["Season"].isin(seasons)]
        games = _unstack_games(detailed)
        games = _add_per_game_metrics(games)
        profile = _aggregate_season_profiles(games)
        profile = _ridge_opponent_adjust(games, profile)
        print(f"  {prefix}: AdjO range=[{profile.AdjO.min():.2f}, {profile.AdjO.max():.2f}], "
              f"AdjD range=[{profile.AdjD.min():.2f}, {profile.AdjD.max():.2f}], "
              f"NetEM range=[{profile.NetEM.min():.1f}, {profile.NetEM.max():.1f}]")
        profile["is_women"] = is_w
        dfs.append(profile)

    profiles = pd.concat(dfs, ignore_index=True)

    return profiles


def build_tabular_features() -> pd.DataFrame:
    """
    End-to-end: build season profiles, merge with tournament labels, compute
    delta features, and save to parquet.

    Returns the training-ready DataFrame with columns:
      Season, TeamA, TeamB, GameID, Result, is_women,
      Delta_eFG_mean, Delta_TOV_mean, …, Delta_Seed
    """
    print("▶ Building season profiles …")
    profiles = build_season_profiles()
    print(f"  {len(profiles):,} team-seasons computed")

    print("▶ Loading tournament labels …")
    labels = load_tourney_labels()

    print("▶ Loading seed lookup …")
    seeds = build_seed_lookup()

    print("▶ Computing delta features for tournament matchups …")
    features = _merge_profiles_to_matchups(labels, profiles, seeds)

    # Report missing features (teams without detailed stats in some seasons)
    n_missing = features[TABULAR_FEATURE_NAMES].isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"  ⚠ {n_missing} matchups have missing features "
              f"(likely pre-2003 women's or non-D1 teams).  Dropping them.")
        features = features.dropna(subset=TABULAR_FEATURE_NAMES)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(TABULAR_FEATURES_FILE, index=False)
    print(f"  Saved {len(features):,} matchups → {TABULAR_FEATURES_FILE}")

    return features


# ── Quick Sanity Check ───────────────────────────────────────────────────

if __name__ == "__main__":
    features = build_tabular_features()
    print(f"\nShape: {features.shape}")
    print(f"Columns: {list(features.columns)}")
    print(f"\nDelta feature ranges:")
    for col in TABULAR_FEATURE_NAMES:
        print(f"  {col:25s}  "
              f"mean={features[col].mean():+.4f}  "
              f"std={features[col].std():.4f}  "
              f"[{features[col].min():+.4f}, {features[col].max():+.4f}]")
    print(f"\nResult balance: {features.Result.mean():.3f}")
