"""
Submission Generator — Produce Kaggle submission CSVs for Stage 1 and Stage 2.

For each matchup in the sample submission:
  1. Look up features from tabular, graph, and Elo pipelines.
  2. Run the fitted base models to get predictions.
  3. Feed predictions + Seed_Diff + is_women into the XGBoost meta-learner.
  4. Clip and output.

XGBoost meta-learner handles NaN natively — no more fallback logic needed.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.config import (
    RAW_DIR, PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR,
    PREDICTION_SEASON, CLIP_LOW, CLIP_HIGH,
)
from src.data_loader import parse_submission_ids
from src.tabular.feature_engineering import (
    build_season_profiles, TABULAR_FEATURE_NAMES, _PROFILE_COLS,
)
from src.tabular.seeds import build_seed_lookup
from src.graph.feature_engineering import (
    build_graph_profiles, GRAPH_FEATURE_NAMES, _GRAPH_COLS,
)

from src.elo.feature_engineering import (
    build_elo_profiles, ELO_FEATURE_NAMES, _ELO_COLS,
)


def _build_delta_features(matchups: pd.DataFrame,
                          profiles: pd.DataFrame,
                          cols: list[str],
                          id_col: str = "TeamID") -> pd.DataFrame:
    """Generic helper to merge team profiles and compute A − B deltas."""
    a_rename = {c: f"A_{c}" for c in cols}
    b_rename = {c: f"B_{c}" for c in cols}

    matchups = matchups.merge(
        profiles[["Season", id_col] + cols].rename(
            columns={id_col: "TeamA", **a_rename}
        ), on=["Season", "TeamA"], how="left",
    )
    matchups = matchups.merge(
        profiles[["Season", id_col] + cols].rename(
            columns={id_col: "TeamB", **b_rename}
        ), on=["Season", "TeamB"], how="left",
    )
    for col in cols:
        matchups[f"Delta_{col}"] = matchups[f"A_{col}"] - matchups[f"B_{col}"]
    return matchups


def generate_submission(stage: int = 2) -> pd.DataFrame:
    """
    Generate a complete Kaggle submission CSV.
    """
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"▶ Generating Stage {stage} submission …")

    # Parse target matchups
    matchups = parse_submission_ids(stage)
    seasons = sorted(matchups.Season.unique())
    print(f"  {len(matchups):,} matchups across seasons {seasons}")

    # Determine which matchups are men's vs women's
    matchups["is_women"] = (matchups.TeamA >= 3000).astype(int)

    # ── Build features for prediction seasons ─────────────────────────
    print("  Building tabular profiles …")
    tab_profiles = build_season_profiles(seasons)
    seeds = build_seed_lookup()

    print("  Building graph profiles …")
    graph_profiles = build_graph_profiles(seasons)



    print("  Building Elo profiles …")
    elo_profiles = build_elo_profiles(seasons)

    # ── Merge features ────────────────────────────────────────────────
    # Tabular
    matchups = _build_delta_features(matchups, tab_profiles, _PROFILE_COLS)

    # Seeds
    matchups = matchups.merge(
        seeds[["Season", "TeamID", "SeedNum"]].rename(
            columns={"TeamID": "TeamA", "SeedNum": "A_Seed"}
        ), on=["Season", "TeamA"], how="left",
    )
    matchups = matchups.merge(
        seeds[["Season", "TeamID", "SeedNum"]].rename(
            columns={"TeamID": "TeamB", "SeedNum": "B_Seed"}
        ), on=["Season", "TeamB"], how="left",
    )
    matchups["Delta_Seed"] = matchups["A_Seed"] - matchups["B_Seed"]
    matchups["Seed_Diff"] = matchups["Delta_Seed"]  # alias for meta-learner

    # Graph
    matchups = _build_delta_features(matchups, graph_profiles, _GRAPH_COLS)



    # Elo
    matchups = _build_delta_features(matchups, elo_profiles, _ELO_COLS)

    # ── Load trained models ───────────────────────────────────────────
    print("  Loading models …")
    model_a = joblib.load(MODELS_DIR / "ModelA.pkl")
    model_b = joblib.load(MODELS_DIR / "ModelB.pkl")
    model_d = joblib.load(MODELS_DIR / "ModelD.pkl")
    meta    = joblib.load(MODELS_DIR / "meta_learner.pkl")

    # ── Generate base predictions ─────────────────────────────────────
    print("  Running base models …")

    # Model A: tabular features
    tab_mask = matchups[TABULAR_FEATURE_NAMES].notna().all(axis=1)
    matchups["ModelA_Pred"] = np.nan
    if tab_mask.sum() > 0:
        matchups.loc[tab_mask, "ModelA_Pred"] = model_a.predict_proba(
            matchups.loc[tab_mask, TABULAR_FEATURE_NAMES].values
        )[:, 1]

    # Model B: graph features
    graph_mask = matchups[GRAPH_FEATURE_NAMES].notna().all(axis=1)
    matchups["ModelB_Pred"] = np.nan
    if graph_mask.sum() > 0:
        matchups.loc[graph_mask, "ModelB_Pred"] = model_b.predict_proba(
            matchups.loc[graph_mask, GRAPH_FEATURE_NAMES].values
        )[:, 1]



    # Model D: Elo features
    elo_mask = matchups[ELO_FEATURE_NAMES].notna().all(axis=1)
    matchups["ModelD_Pred"] = np.nan
    if elo_mask.sum() > 0:
        matchups.loc[elo_mask, "ModelD_Pred"] = model_d.predict_proba(
            matchups.loc[elo_mask, ELO_FEATURE_NAMES].values
        )[:, 1]

    # ── Ensemble prediction via XGBoost meta-learner ──────────────────
    print("  Running XGBoost meta-learner …")

    from src.ensemble.meta_learner import META_FEATURES
    X_meta = matchups[META_FEATURES].values
    matchups["Pred"] = meta.predict_proba(X_meta)[:, 1]
    matchups["Pred"] = matchups["Pred"].clip(CLIP_LOW, CLIP_HIGH)

    # ── Write submission ──────────────────────────────────────────────
    submission = matchups[["GameID", "Pred"]].rename(columns={"GameID": "ID"})
    out_path = SUBMISSIONS_DIR / f"submission_stage{stage}.csv"
    submission.to_csv(out_path, index=False)

    print(f"\n  ✓ Submission saved → {out_path}")
    print(f"    Rows: {len(submission):,}")
    print(f"    Pred range: [{submission.Pred.min():.4f}, {submission.Pred.max():.4f}]")
    print(f"    Pred mean:  {submission.Pred.mean():.4f}")
    print(f"    ModelA: {tab_mask.sum():,} | ModelB: {graph_mask.sum():,} | "
          f"ModelD: {elo_mask.sum():,}")

    return submission


# ── Quick Run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_submission(stage=2)
