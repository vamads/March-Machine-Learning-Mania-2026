"""
Final Stage 2 Submission Script — 2026 Tournament.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm

from src.config import (
    PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR,
    PREDICTION_SEASON, CLIP_LOW, CLIP_HIGH
)
from src.data_loader import parse_submission_ids
from src.tabular.feature_engineering import build_season_profiles, _merge_profiles_to_matchups, TABULAR_FEATURE_NAMES
from src.tabular.seeds import build_seed_lookup
from src.elo.feature_engineering import build_elo_profiles
from src.graph.feature_engineering import build_graph_profiles, _merge_graph_to_matchups, GRAPH_FEATURE_NAMES
from src.ensemble.meta_learner import predict_ensemble, merge_hca_sensitivity

def run_submission():
    print("\n" + "="*60)
    print("  🏆 MARCH MADNESS 2026 — FINAL STAGE 2 SUBMISSION")
    print("="*60)

    # 1. Matchup Grid
    print("\n[1/6] Loading 2026 Matchup Grid …")
    sub_df = parse_submission_ids(stage=2)
    sub_df["is_women"] = (sub_df["TeamA"] >= 3000).astype(int)
    print(f"      Rows to predict: {len(sub_df):,}")

    # 2. Loading 2026 Profiles
    print("\n[2/6] Building/Loading 2026 Team Profiles …")
    tab_profiles = build_season_profiles(seasons=[2026])
    elo_profiles = build_elo_profiles(seasons=[2026])
    graph_profiles = build_graph_profiles(seasons=[2026])
    hca_profiles = pd.read_parquet(PROCESSED_DIR / "hca_sensitivity.parquet")
    injury_profiles = pd.read_parquet(PROCESSED_DIR / "injury_risk.parquet")

    # 3. Merging & Imputation
    print("\n[3/6] Merging Features & Imputing Seeds (Seed 20.0 fallback) …")
    seed_lookup = build_seed_lookup()
    
    # Tabular + Seeds
    sub_df = _merge_profiles_to_matchups(sub_df, tab_profiles, seed_lookup)
    sub_df["A_Seed"] = sub_df["A_Seed"].fillna(20.0)
    sub_df["B_Seed"] = sub_df["B_Seed"].fillna(20.0)
    sub_df["Delta_Seed"] = sub_df["A_Seed"] - sub_df["B_Seed"]
    
    # Elo
    # Merge Team A Elo
    sub_df = sub_df.merge(elo_profiles[["TeamID", "Elo_Final"]].rename(columns={"TeamID": "TeamA", "Elo_Final": "A_Elo"}), on="TeamA", how="left")
    sub_df = sub_df.merge(elo_profiles[["TeamID", "Elo_Final"]].rename(columns={"TeamID": "TeamB", "Elo_Final": "B_Elo"}), on="TeamB", how="left")
    sub_df["A_Elo"] = sub_df["A_Elo"].fillna(1500.0)
    sub_df["B_Elo"] = sub_df["B_Elo"].fillna(1500.0)
    sub_df["Delta_Elo_Final"] = sub_df["A_Elo"] - sub_df["B_Elo"]
    
    # Graph
    sub_df = _merge_graph_to_matchups(sub_df, graph_profiles)

    # 4. Base Model Inference
    print("\n[4/6] Running Base Model Inference (A, B, D) …")
    
    # Model A
    model_a = joblib.load(MODELS_DIR / "ModelA.pkl")
    X_a = sub_df[TABULAR_FEATURE_NAMES].fillna(0).values
    sub_df["ModelA_Pred"] = model_a.predict_proba(X_a)[:, 1]

    # Model B
    model_b = joblib.load(MODELS_DIR / "ModelB.pkl")
    X_b = sub_df[GRAPH_FEATURE_NAMES].fillna(0).values
    # Handle if X_b has fewer columns (GNN missing in 2026?)
    sub_df["ModelB_Pred"] = model_b.predict_proba(X_b)[:, 1]

    # Model D (Elo Sigmoid) - our Model D is usually an XGB but can be Sigmoid
    model_d = joblib.load(MODELS_DIR / "ModelD.pkl")
    X_d = sub_df[["Delta_Elo_Final"]].fillna(0).values
    sub_df["ModelD_Pred"] = model_d.predict_proba(X_d)[:, 1]
    
    # Model E (Fallback 0.5 for non-bracket teams)
    sub_df["ModelE_Pred"] = 0.5

    # 5. Ensemble + Platt + Hedge
    print("\n[5/6] Final Stacking + Platt + Injury Hedge …")
    payload = joblib.load(MODELS_DIR / "meta_learner.pkl")
    
    # HCA Sensitivity
    sub_df = merge_hca_sensitivity(sub_df, hca_profiles)
    
    # Injury Risk
    sub_df = sub_df.merge(injury_profiles.rename(columns={"TeamID": "TeamA", "Severity": "RiskA"}), on="TeamA", how="left")
    sub_df = sub_df.merge(injury_profiles.rename(columns={"TeamID": "TeamB", "Severity": "RiskB"}), on="TeamB", how="left")
    sub_df["RiskA"] = sub_df["RiskA"].fillna(0.0)
    sub_df["RiskB"] = sub_df["RiskB"].fillna(0.0)
    sub_df["Max_Risk"] = sub_df[["RiskA", "RiskB"]].max(axis=1)

    preds = predict_ensemble(
        payload,
        sub_df["ModelA_Pred"].values,
        sub_df["ModelB_Pred"].values,
        sub_df["ModelD_Pred"].values,
        sub_df["ModelE_Pred"].values,
        sub_df["Delta_Seed"].values,
        sub_df["is_women"].values,
        sub_df["Delta_HCA_Sensitivity"].values,
        injury_hedge=sub_df["Max_Risk"].values
    )

    # 6. Save
    print("\n[6/6] Saving Submission File …")
    submission = pd.DataFrame({
        "ID": sub_df["GameID"],
        "Pred": preds
    })
    
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = SUBMISSIONS_DIR / "submission_stage2.csv"
    submission.to_csv(out_file, index=False)
    
    print(f"\n✅ SUCCESS! saved → {out_file}")
    print(f"   Mean Pred: {submission.Pred.mean():.4f}")
    print(f"   Max Risk:  {sub_df.Max_Risk.max():.1%}")

if __name__ == "__main__":
    run_submission()
