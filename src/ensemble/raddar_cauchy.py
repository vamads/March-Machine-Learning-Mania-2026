"""
Raddar/Vilnius NCAA Cauchy Leaf-Node Stacker.

Replicates the 4th-place 2025 March Machine Learning Mania solution:
  1. Builds ~25 Raddar-style features (box scores, Elo, GLM quality, SOS).
  2. Adds 3 Laplace-smoothed features (Prior Matchup, Away Wins, L14 Win Ratio).
  3. Trains Cauchy-loss XGBoost regressor on Point Differential.
  4. Extracts leaf indices → OHE → L1 Logistic Regression meta-learner.
  5. Evaluates on 2022-2025 holdout.

Usage:
    conda run -n madness python -m src.ensemble.raddar_cauchy
"""

import pandas as pd
import numpy as np
import xgboost as xgb_lib
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
from scipy.sparse import hstack, csr_matrix
from scipy import sparse
import warnings

from src.config import (
    RAW_DIR, HOLDOUT_SEASONS, TRAIN_ONLY_SEASONS, TRAIN_SEASONS,
    CLIP_LOW, CLIP_HIGH, REGULAR_SEASON_CUTOFF,
    FIRST_DETAILED_SEASON, LAST_HISTORICAL_SEASON,
)
from src.data_loader import (
    load_regular_season_detailed, load_regular_season_compact,
    load_tourney_compact, load_tourney_seeds, make_game_id,
)

warnings.filterwarnings("ignore")

CAUCHY_C = 10.0
SIGMOID_K = 0.175


# ═══════════════════════════════════════════════════════════════════════
# CAUCHY OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════

def cauchy_objective(y_true, y_pred):
    x = y_pred - y_true
    c2 = CAUCHY_C ** 2
    x2 = x ** 2
    grad = x / ((x2 / c2) + 1.0)
    hess = c2 * (c2 - x2) / ((x2 + c2) ** 2)
    hess = np.maximum(hess, 1e-5)
    return grad, hess


# ═══════════════════════════════════════════════════════════════════════
# TASK 1: RADDAR FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def _parse_seed_num(seed_str: str) -> int:
    """Parse 'W01' → 1, 'X16a' → 16, etc."""
    return int("".join(c for c in seed_str[1:] if c.isdigit()))


def _build_seed_lookup(seasons):
    """Build (Season, TeamID) → SeedNum lookup for M+W."""
    dfs = []
    for prefix in ["M", "W"]:
        seeds = load_tourney_seeds(prefix)
        seeds = seeds[seeds.Season.isin(seasons)]
        seeds["SeedNum"] = seeds["Seed"].apply(_parse_seed_num)
        seeds["is_women"] = 1 if prefix == "W" else 0
        dfs.append(seeds[["Season", "TeamID", "SeedNum", "is_women"]])
    return pd.concat(dfs, ignore_index=True)


def _build_box_score_profiles(seasons):
    """
    Build per-team per-season box score averages from detailed results.
    Only uses DayNum <= 132 (temporal firewall).
    """
    all_profiles = []

    for prefix, is_w in [("M", 0), ("W", 1)]:
        det = load_regular_season_detailed(prefix)
        det = det[det.Season.isin(seasons)]
        det = det[det.DayNum <= REGULAR_SEASON_CUTOFF]

        stat_cols = ["FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
                     "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF"]

        # Winner rows
        w = det[["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"] +
                [f"W{c}" for c in stat_cols] + [f"L{c}" for c in stat_cols]].copy()
        w = w.rename(columns={
            "WTeamID": "TeamID", "LTeamID": "OppID",
            "WScore": "Score", "LScore": "OppScore",
            **{f"W{c}": c for c in stat_cols},
            **{f"L{c}": f"Opp_{c}" for c in stat_cols},
        })
        w["Win"] = 1
        w["Home"] = w["WLoc"].map({"H": 1.0, "A": -1.0, "N": 0.0}).fillna(0.0)
        w = w.drop(columns=["WLoc"])

        # Loser rows
        l = det[["Season", "DayNum", "LTeamID", "WTeamID", "LScore", "WScore", "WLoc"] +
                [f"L{c}" for c in stat_cols] + [f"W{c}" for c in stat_cols]].copy()
        l = l.rename(columns={
            "LTeamID": "TeamID", "WTeamID": "OppID",
            "LScore": "Score", "WScore": "OppScore",
            **{f"L{c}": c for c in stat_cols},
            **{f"W{c}": f"Opp_{c}" for c in stat_cols},
        })
        l["Win"] = 0
        l["Home"] = l["WLoc"].map({"H": -1.0, "A": 1.0, "N": 0.0}).fillna(0.0)
        l = l.drop(columns=["WLoc"])

        games = pd.concat([w, l], ignore_index=True)
        games["Poss"] = games.FGA + 0.44 * games.FTA - games.OR + games.TO
        games["Opp_Poss"] = games.Opp_FGA + 0.44 * games.Opp_FTA - games.Opp_OR + games.Opp_TO
        games["OE"] = np.where(games.Poss > 0, games.Score / games.Poss * 100, 0)
        games["DE"] = np.where(games.Opp_Poss > 0, games.OppScore / games.Opp_Poss * 100, 0)
        games["Margin"] = games.Score - games.OppScore
        games["FGPct"] = np.where(games.FGA > 0, games.FGM / games.FGA, 0)
        games["FG3Pct"] = np.where(games.FGA3 > 0, games.FGM3 / games.FGA3, 0)
        games["FTPct"] = np.where(games.FTA > 0, games.FTM / games.FTA, 0)
        games["is_women"] = is_w

        # Aggregate per team per season
        agg = games.groupby(["Season", "TeamID"]).agg(
            GP=("Win", "count"),
            Wins=("Win", "sum"),
            PPG=("Score", "mean"),
            PPGAllowed=("OppScore", "mean"),
            FGPct=("FGPct", "mean"),
            FG3Pct=("FG3Pct", "mean"),
            FTPct=("FTPct", "mean"),
            OR=("OR", "mean"),
            DR=("DR", "mean"),
            Ast=("Ast", "mean"),
            TO=("TO", "mean"),
            Stl=("Stl", "mean"),
            Blk=("Blk", "mean"),
            PF=("PF", "mean"),
            OE=("OE", "mean"),
            DE=("DE", "mean"),
            Margin=("Margin", "mean"),
        ).reset_index()
        agg["WinPct"] = agg.Wins / agg.GP
        agg["NetEM"] = agg.OE - agg.DE
        agg["is_women"] = is_w

        # SOS: average opponent win percentage
        opp_wp = games.groupby(["Season", "TeamID"])["OppID"].apply(list).reset_index()
        team_wp = agg.set_index(["Season", "TeamID"])["WinPct"]

        sos_rows = []
        for _, row in opp_wp.iterrows():
            opps = row["OppID"]
            opp_wps = [team_wp.get((row.Season, opp), 0.5) for opp in opps]
            sos_rows.append({"Season": row.Season, "TeamID": row.TeamID, "SOS": np.mean(opp_wps)})
        sos_df = pd.DataFrame(sos_rows)
        agg = agg.merge(sos_df, on=["Season", "TeamID"], how="left")
        agg["SOS"] = agg["SOS"].fillna(0.5)

        # GLM Team Quality via Ridge Regression on point margin
        for season in sorted(agg.Season.unique()):
            sg = games[games.Season == season]
            team_ids = np.sort(sg["TeamID"].unique())
            n_teams = len(team_ids)
            t2i = {t: i for i, t in enumerate(team_ids)}

            n_games = len(sg)
            rows_i, cols_i, vals_i = [], [], []
            y = sg["Margin"].values.astype(float)

            for j in range(n_games):
                oi = t2i.get(int(sg.iloc[j]["TeamID"]))
                di = t2i.get(int(sg.iloc[j]["OppID"]))
                if oi is None or di is None:
                    continue
                rows_i.append(j); cols_i.append(oi); vals_i.append(1.0)
                rows_i.append(j); cols_i.append(n_teams + di); vals_i.append(1.0)
                rows_i.append(j); cols_i.append(2 * n_teams); vals_i.append(float(sg.iloc[j]["Home"]))

            X_glm = sparse.csr_matrix((vals_i, (rows_i, cols_i)), shape=(n_games, 2 * n_teams + 1))
            ridge = Ridge(alpha=1.0, fit_intercept=True)
            ridge.fit(X_glm, y)
            off_coefs = {t: float(ridge.coef_[t2i[t]]) for t in team_ids}

            mask = agg.Season == season
            agg.loc[mask, "Quality"] = agg.loc[mask, "TeamID"].map(off_coefs)

        agg["Quality"] = agg["Quality"].fillna(0.0)
        all_profiles.append(agg)

    return pd.concat(all_profiles, ignore_index=True), games


def _build_elo_ratings(seasons):
    """Simple Elo from compact results (M+W)."""
    K = 32.0
    elo = {}  # (prefix, team_id) → rating

    all_elo = []  # (Season, TeamID, Elo, is_women)

    for prefix, is_w in [("M", 0), ("W", 1)]:
        compact = load_regular_season_compact(prefix)
        compact = compact[compact.Season.isin(seasons)].sort_values(["Season", "DayNum"])

        prev_season = None
        for _, r in compact.iterrows():
            season = r.Season
            if season != prev_season:
                # Season regression: move 30% toward 1500
                for key in list(elo.keys()):
                    if key[0] == prefix:
                        elo[key] = elo[key] * 0.7 + 1500 * 0.3
                prev_season = season

            w, l = (prefix, int(r.WTeamID)), (prefix, int(r.LTeamID))
            elo.setdefault(w, 1500.0)
            elo.setdefault(l, 1500.0)

            # Only update within temporal firewall
            if r.DayNum > REGULAR_SEASON_CUTOFF:
                continue

            expected_w = 1.0 / (1.0 + 10 ** ((elo[l] - elo[w]) / 400.0))
            elo[w] += K * (1.0 - expected_w)
            elo[l] += K * (0.0 - (1.0 - expected_w))

        # Snapshot end-of-regular-season Elo for each team active this season
        for s in sorted(seasons):
            teams_in_season = compact[compact.Season == s]["WTeamID"].unique().tolist() + \
                              compact[compact.Season == s]["LTeamID"].unique().tolist()
            for t in set(teams_in_season):
                all_elo.append({
                    "Season": s, "TeamID": t,
                    "Elo": elo.get((prefix, t), 1500.0),
                    "is_women": is_w,
                })

    return pd.DataFrame(all_elo)


def build_raddar_features(seasons):
    """Build the full Raddar feature set for all tournament matchups."""
    print("  Building box score profiles + GLM quality...")
    profiles, _ = _build_box_score_profiles(seasons)

    print("  Building Elo ratings...")
    elo_df = _build_elo_ratings(seasons)
    profiles = profiles.merge(elo_df, on=["Season", "TeamID", "is_women"], how="left")
    profiles["Elo"] = profiles["Elo"].fillna(1500.0)

    print("  Building seed lookup...")
    seeds = _build_seed_lookup(seasons)

    # Build tournament matchups
    all_matchups = []
    for prefix, is_w in [("M", 0), ("W", 1)]:
        tc = load_tourney_compact(prefix)
        tc = tc[tc.Season.isin(seasons)]
        for _, r in tc.iterrows():
            lo, hi = sorted([int(r.WTeamID), int(r.LTeamID)])
            result = 1 if int(r.WTeamID) == lo else 0
            pdiff = int(r.WScore) - int(r.LScore)
            point_diff = pdiff if int(r.WTeamID) == lo else -pdiff
            all_matchups.append({
                "Season": int(r.Season), "TeamA": lo, "TeamB": hi,
                "DayNum": int(r.DayNum),
                "GameID": make_game_id(int(r.Season), lo, hi),
                "Result": result, "PointDiff": point_diff, "is_women": is_w,
            })
    matchups = pd.DataFrame(all_matchups)

    # Merge profiles for Team A
    prof_cols = ["WinPct", "PPG", "PPGAllowed", "FGPct", "FG3Pct", "FTPct",
                 "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF",
                 "OE", "DE", "NetEM", "Margin", "SOS", "Quality", "Elo"]

    a_rename = {c: f"A_{c}" for c in prof_cols}
    matchups = matchups.merge(
        profiles[["Season", "TeamID"] + prof_cols].rename(columns={"TeamID": "TeamA", **a_rename}),
        on=["Season", "TeamA"], how="left"
    )
    b_rename = {c: f"B_{c}" for c in prof_cols}
    matchups = matchups.merge(
        profiles[["Season", "TeamID"] + prof_cols].rename(columns={"TeamID": "TeamB", **b_rename}),
        on=["Season", "TeamB"], how="left"
    )

    # Merge seeds
    matchups = matchups.merge(
        seeds[["Season", "TeamID", "SeedNum"]].rename(columns={"TeamID": "TeamA", "SeedNum": "A_Seed"}),
        on=["Season", "TeamA"], how="left"
    )
    matchups = matchups.merge(
        seeds[["Season", "TeamID", "SeedNum"]].rename(columns={"TeamID": "TeamB", "SeedNum": "B_Seed"}),
        on=["Season", "TeamB"], how="left"
    )

    # Compute deltas
    for col in prof_cols:
        matchups[f"Delta_{col}"] = matchups[f"A_{col}"] - matchups[f"B_{col}"]

    matchups["Delta_Seed"] = matchups["A_Seed"] - matchups["B_Seed"]

    return matchups


# ═══════════════════════════════════════════════════════════════════════
# TASK 2: LAPLACE FEATURES
# ═══════════════════════════════════════════════════════════════════════

def _compute_laplace_team_features(seasons):
    """Laplace smoothed away wins and L14 win ratio per team-season."""
    dfs = []
    for prefix in ["M", "W"]:
        compact = load_regular_season_compact(prefix)
        compact = compact[compact.Season.isin(seasons)]
        compact = compact[compact.DayNum <= REGULAR_SEASON_CUTOFF]

        # Away wins
        aw = compact[compact.WLoc == "A"].groupby(["Season", "WTeamID"]).size().reset_index(name="AW_W")
        al = compact[compact.WLoc == "H"].groupby(["Season", "LTeamID"]).size().reset_index(name="AW_L")
        away = pd.merge(aw.rename(columns={"WTeamID": "TeamID"}),
                        al.rename(columns={"LTeamID": "TeamID"}),
                        on=["Season", "TeamID"], how="outer").fillna(0)
        away["Laplace_AW"] = (away.AW_W + 1) / (away.AW_W + away.AW_L + 2)

        # L14 wins (DayNum >= 118)
        l14 = compact[compact.DayNum >= 118]
        l14w = l14.groupby(["Season", "WTeamID"]).size().reset_index(name="L14W")
        l14l = l14.groupby(["Season", "LTeamID"]).size().reset_index(name="L14L")
        l14s = pd.merge(l14w.rename(columns={"WTeamID": "TeamID"}),
                        l14l.rename(columns={"LTeamID": "TeamID"}),
                        on=["Season", "TeamID"], how="outer").fillna(0)
        l14s["Laplace_L14"] = (l14s.L14W + 1) / (l14s.L14W + l14s.L14L + 2)

        team_f = pd.merge(away[["Season", "TeamID", "Laplace_AW"]],
                          l14s[["Season", "TeamID", "Laplace_L14"]],
                          on=["Season", "TeamID"], how="outer").fillna(0.5)
        dfs.append(team_f)
    return pd.concat(dfs, ignore_index=True)


def _compute_prior_matchup(target_season):
    """Historical win rate between pairs, Laplace smoothed, strictly before target_season."""
    dfs = []
    for prefix in ["M", "W"]:
        compact = load_regular_season_compact(prefix)
        tourney = load_tourney_compact(prefix)
        all_g = pd.concat([compact, tourney], ignore_index=True)
        all_g = all_g[(all_g.Season < target_season) |
                      ((all_g.Season == target_season) & (all_g.DayNum <= REGULAR_SEASON_CUTOFF))]
        dfs.append(all_g)

    past = pd.concat(dfs, ignore_index=True)
    past["TeamA"] = np.minimum(past.WTeamID, past.LTeamID)
    past["TeamB"] = np.maximum(past.WTeamID, past.LTeamID)
    past["A_Won"] = (past.TeamA == past.WTeamID).astype(int)

    pair = past.groupby(["TeamA", "TeamB"])["A_Won"].agg(["sum", "count"]).reset_index()
    pair["Laplace_Prior_Matchup"] = (pair["sum"] + 1) / (pair["count"] + 2)
    return pair[["TeamA", "TeamB", "Laplace_Prior_Matchup"]]


# ═══════════════════════════════════════════════════════════════════════
# FEATURE LIST
# ═══════════════════════════════════════════════════════════════════════

RADDAR_FEATURES = [
    "Delta_Seed",
    "Delta_WinPct", "Delta_PPG", "Delta_PPGAllowed", "Delta_Margin",
    "Delta_FGPct", "Delta_FG3Pct", "Delta_FTPct",
    "Delta_OR", "Delta_DR", "Delta_Ast", "Delta_TO", "Delta_Stl", "Delta_Blk", "Delta_PF",
    "Delta_OE", "Delta_DE", "Delta_NetEM",
    "Delta_SOS", "Delta_Quality", "Delta_Elo",
    "A_Elo", "B_Elo", "A_Quality", "B_Quality",
    # Laplace
    "Delta_Laplace_AW", "Delta_Laplace_L14", "Laplace_Prior_Matchup",
]


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# SYMMETRIC DOUBLING
# ═══════════════════════════════════════════════════════════════════════

def _apply_symmetric_doubling(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate rows and swap A/B features and target."""
    df2 = df.copy()
    
    # Swap A and B features
    a_cols = [c for c in df.columns if c.startswith("A_")]
    b_cols = [c for c in df.columns if c.startswith("B_")]
    mapping = {a: b.replace("A_", "B_") for a, b in zip(a_cols, b_cols)}
    mapping.update({b: a.replace("B_", "A_") for a, b in zip(a_cols, b_cols)})
    
    # Delta features also need to be negated
    delta_cols = [c for c in df.columns if c.startswith("Delta_")]
    
    # Apply swap
    new_vals = {}
    for a, b in zip(a_cols, b_cols):
        new_vals[a] = df2[b]
        new_vals[b] = df2[a]
    for d in delta_cols:
        new_vals[d] = -df2[d]
        
    df2.update(pd.DataFrame(new_vals))
    
    # Swap Team IDs
    df2["TeamA"], df2["TeamB"] = df["TeamB"], df["TeamA"]
    
    # Flip Target
    df2["Result"] = 1 - df["Result"]
    if "PointDiff" in df2.columns:
        df2["PointDiff"] = -df["PointDiff"]
        
    return pd.concat([df, df2], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════
# ENSEMBLE API
# ═══════════════════════════════════════════════════════════════════════

def get_model_e_oof(seasons=None):
    """
    Build features and return OOF predictions for Model E.
    Used by base_models.py.
    """
    if seasons is None:
        seasons = list(range(FIRST_DETAILED_SEASON, LAST_HISTORICAL_SEASON + 1))
        
    matchups = build_raddar_features(seasons)
    laplace_team = _compute_laplace_team_features(seasons)
    
    matchups = matchups.merge(
        laplace_team.rename(columns={"TeamID": "TeamA", "Laplace_AW": "A_Laplace_AW", "Laplace_L14": "A_Laplace_L14"}),
        on=["Season", "TeamA"], how="left"
    )
    matchups = matchups.merge(
        laplace_team.rename(columns={"TeamID": "TeamB", "Laplace_AW": "B_Laplace_AW", "Laplace_L14": "B_Laplace_L14"}),
        on=["Season", "TeamB"], how="left"
    )
    matchups["A_Laplace_AW"] = matchups["A_Laplace_AW"].fillna(0.5)
    matchups["A_Laplace_L14"] = matchups["A_Laplace_L14"].fillna(0.5)
    matchups["B_Laplace_AW"] = matchups["B_Laplace_AW"].fillna(0.5)
    matchups["B_Laplace_L14"] = matchups["B_Laplace_L14"].fillna(0.5)
    matchups["Delta_Laplace_AW"] = matchups["A_Laplace_AW"] - matchups["B_Laplace_AW"]
    matchups["Delta_Laplace_L14"] = matchups["A_Laplace_L14"] - matchups["B_Laplace_L14"]
    matchups["Laplace_Prior_Matchup"] = 0.5
    
    matchups = matchups.dropna(subset=RADDAR_FEATURES).reset_index(drop=True)
    
    # Apply Symmetric Doubling to training data
    matchups = _apply_symmetric_doubling(matchups)
    
    # Expanding Window CV (similar to logic in Task 3 but standardized)
    oof_rows = []
    
    print("\n  Expanding Window CV for Model E (Raddar-Cauchy)...")
    
    # Reuse LOSO or Expanding logic? 
    # The original author used LOSO, but for our pipeline we use Expanding to avoid leakage.
    # Let's use Expanding.
    
    available_seasons = sorted(matchups.Season.unique())
    for i, test_season in enumerate(available_seasons):
        past = [s for s in available_seasons if s < test_season]
        if len(past) < 3:
            continue
            
        tr = matchups[matchups.Season.isin(past)].copy()
        va = matchups[matchups.Season == test_season].copy()
        
        # Prior matchup
        prior = _compute_prior_matchup(test_season)
        tr = tr.merge(prior, on=["TeamA", "TeamB"], how="left", suffixes=("", "_new"))
        if "Laplace_Prior_Matchup_new" in tr.columns:
            tr["Laplace_Prior_Matchup"] = tr["Laplace_Prior_Matchup_new"].fillna(tr["Laplace_Prior_Matchup"])
            tr = tr.drop(columns=["Laplace_Prior_Matchup_new"])
        tr["Laplace_Prior_Matchup"] = tr["Laplace_Prior_Matchup"].fillna(0.5)

        va = va.merge(prior, on=["TeamA", "TeamB"], how="left", suffixes=("", "_new"))
        if "Laplace_Prior_Matchup_new" in va.columns:
            va["Laplace_Prior_Matchup"] = va["Laplace_Prior_Matchup_new"].fillna(va["Laplace_Prior_Matchup"])
            va = va.drop(columns=["Laplace_Prior_Matchup_new"])
        va["Laplace_Prior_Matchup"] = va["Laplace_Prior_Matchup"].fillna(0.5)

        # Stage 1: XGBoost Cauchy (HPO Optimized)
        model = xgb_lib.XGBRegressor(
            n_estimators=40, max_depth=2, learning_rate=0.05,
            reg_alpha=0.7433685798354077, reg_lambda=5.043833569209126,
            subsample=0.8745163099522736,
            objective=cauchy_objective, random_state=42, n_jobs=-1,
        )
        model.fit(tr[RADDAR_FEATURES].values, tr["PointDiff"].values)
        
        # Stage 2: TF-IDF Stacking (HPO Optimized)
        tr_leaves = model.apply(tr[RADDAR_FEATURES].values)
        va_leaves = model.apply(va[RADDAR_FEATURES].values)
        
        tr_margin = model.predict(tr[RADDAR_FEATURES].values)
        va_margin = model.predict(va[RADDAR_FEATURES].values)
        tr_prob = 1.0 / (1.0 + np.exp(-tr_margin * SIGMOID_K))
        va_prob = 1.0 / (1.0 + np.exp(-va_margin * SIGMOID_K))
        
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        X_tr_leaves = encoder.fit_transform(tr_leaves)
        X_va_leaves = encoder.transform(va_leaves)
        
        X_tr = hstack([csr_matrix(tr_prob.reshape(-1, 1)), X_tr_leaves])
        X_va = hstack([csr_matrix(va_prob.reshape(-1, 1)), X_va_leaves])
        
        meta = LogisticRegression(penalty="l1", solver="liblinear", C=0.05160999690371322, max_iter=1000, random_state=42)
        meta.fit(X_tr, tr["Result"].values)
        
        preds = meta.predict_proba(X_va)[:, 1]
        
        for gid, pred in zip(va["GameID"], preds):
            oof_rows.append({"Season": test_season, "GameID": gid, "ModelE_Pred": float(pred)})
            
    return pd.DataFrame(oof_rows)


# ═══════════════════════════════════════════════════════════════════════
# PRODUCTION WRAPPER
# ═══════════════════════════════════════════════════════════════════════

class RaddarCauchyModel:
    """Wrapper for the 2-stage Model E pipeline."""
    def __init__(self, xgb_model, encoder, meta_lr):
        self.xgb_model = xgb_model
        self.encoder = encoder
        self.meta_lr = meta_lr

    def predict_proba(self, X):
        """Standard sklearn-like interface."""
        # Stage 1
        margin = self.xgb_model.predict(X)
        prob = 1.0 / (1.0 + np.exp(-margin * SIGMOID_K))
        leaves = self.xgb_model.apply(X)
        
        # Stage 2
        sparse_leaves = self.encoder.transform(leaves)
        X_meta = hstack([csr_matrix(prob.reshape(-1, 1)), sparse_leaves])
        
        final_probs = self.meta_lr.predict_proba(X_meta)
        return final_probs


def train_final_model_e():
    """Trains the final Model E pipeline on ALL historical data."""
    print("\n  Training Final Model E (Raddar-Cauchy)...")
    all_seasons = list(range(FIRST_DETAILED_SEASON, LAST_HISTORICAL_SEASON + 1))
    matchups = build_raddar_features(all_seasons)
    laplace_team = _compute_laplace_team_features(all_seasons)
    
    matchups = matchups.merge(
        laplace_team.rename(columns={"TeamID": "TeamA", "Laplace_AW": "A_Laplace_AW", "Laplace_L14": "A_Laplace_L14"}),
        on=["Season", "TeamA"], how="left"
    )
    matchups = matchups.merge(
        laplace_team.rename(columns={"TeamID": "TeamB", "Laplace_AW": "B_Laplace_AW", "Laplace_L14": "B_Laplace_L14"}),
        on=["Season", "TeamB"], how="left"
    )
    matchups["A_Laplace_AW"] = matchups["A_Laplace_AW"].fillna(0.5); matchups["A_Laplace_L14"] = matchups["A_Laplace_L14"].fillna(0.5)
    matchups["B_Laplace_AW"] = matchups["B_Laplace_AW"].fillna(0.5); matchups["B_Laplace_L14"] = matchups["B_Laplace_L14"].fillna(0.5)
    matchups["Delta_Laplace_AW"] = matchups["A_Laplace_AW"] - matchups["B_Laplace_AW"]
    matchups["Delta_Laplace_L14"] = matchups["A_Laplace_L14"] - matchups["B_Laplace_L14"]
    
    # Use full historical prior matchup
    prior = _compute_prior_matchup(LAST_HISTORICAL_SEASON + 1)
    matchups = matchups.merge(prior, on=["TeamA", "TeamB"], how="left")
    matchups["Laplace_Prior_Matchup"] = matchups["Laplace_Prior_Matchup"].fillna(0.5)
    matchups = matchups.dropna(subset=RADDAR_FEATURES).reset_index(drop=True)
    
    # Symmetric Doubling
    matchups = _apply_symmetric_doubling(matchups)
    
    # Stage 1
    xgb_final = xgb_lib.XGBRegressor(
        n_estimators=40, max_depth=4, learning_rate=0.05,
        objective=cauchy_objective, random_state=42, n_jobs=-1,
    )
    xgb_final.fit(matchups[RADDAR_FEATURES].values, matchups["PointDiff"].values)
    
    # Stage 2
    leaves = xgb_final.apply(matchups[RADDAR_FEATURES].values)
    margin = xgb_final.predict(matchups[RADDAR_FEATURES].values)
    prob = 1.0 / (1.0 + np.exp(-margin * SIGMOID_K))
    
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_leaves = encoder.fit_transform(leaves)
    X_meta = hstack([csr_matrix(prob.reshape(-1, 1)), X_leaves])
    
    meta_lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=1000, random_state=42)
    meta_lr.fit(X_meta, matchups["Result"].values)
    
    return RaddarCauchyModel(xgb_final, encoder, meta_lr)


if __name__ == "__main__":
    # Regression test
    oof = get_model_e_oof()
    print(f"Model E OOF complete. Shape: {oof.shape}")
    print(oof.head())
