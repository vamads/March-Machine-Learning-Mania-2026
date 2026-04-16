"""
Meta-Learner (Level 1) — Stacking Classifier with Interaction Features.

Takes OOF predictions from Models A, B, D plus Seed_Diff, is_women,
and engineered interaction features, then learns how to optimally combine them.

Interaction features capture ensemble consensus/disagreement:
  - Consensus_Spread: max(preds) - min(preds)  → ensemble uncertainty
  - Model_Variance: std(preds)                  → paradigm divergence
  - Algo_Conflict: |ModelA - ModelD|            → tabular vs Elo disagreement
  - Agreement_Mult: ModelA * ModelD             → confidence amplifier
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss, brier_score_loss
import joblib

from src.config import (
    PROCESSED_DIR, MODELS_DIR,
    CLIP_LOW, CLIP_HIGH,
    PREDICTION_SEASON,
)


# ═══════════════════════════════════════════════════════════════════════════
# META FEATURES
# ═══════════════════════════════════════════════════════════════════════════

# Base model predictions + raw features
_BASE_META_FEATURES = [
    "ModelA_Pred", "ModelB_Pred", "ModelD_Pred", "ModelE_Pred",
    "Seed_Diff", "is_women", "Delta_HCA_Sensitivity",
]

# Engineered interaction features
_INTERACTION_FEATURES = [
    "Consensus_Spread",   # max(preds) - min(preds)
    "Model_Variance",     # std(preds)
    "Algo_Conflict",      # |ModelA - ModelD|
    "Agreement_Mult",     # ModelA * ModelD
]

META_FEATURES = _BASE_META_FEATURES + _INTERACTION_FEATURES


def add_meta_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered interaction features from base model predictions.
    Handles NaN gracefully (fillna with 0.5 for preds before computing).
    
    Model E (Raddar-Cauchy) is clipped to [0.05, 0.95] specifically for
    interaction calculations to prevent it from dominating the ensemble spread.
    """
    df = df.copy()

    # Stack predictions for vectorized operations
    pred_cols = ["ModelA_Pred", "ModelB_Pred", "ModelD_Pred", "ModelE_Pred"]
    preds = df[pred_cols].copy()

    # Bounding Model E for interaction stability
    if "ModelE_Pred" in preds.columns:
        preds["ModelE_Pred"] = preds["ModelE_Pred"].clip(0.05, 0.95)

    # Fill NaN with 0.5 (neutral) for interaction computations
    preds_filled = preds.fillna(0.5)

    # Consensus Spread: max - min across all models
    df["Consensus_Spread"] = preds_filled.max(axis=1) - preds_filled.min(axis=1)

    # Model Variance: std across all models
    df["Model_Variance"] = preds_filled.std(axis=1)

    # Algorithmic Conflict: |Tabular - Elo| (short-term efficiency vs long-term trajectory)
    df["Algo_Conflict"] = (preds_filled["ModelA_Pred"] - preds_filled["ModelD_Pred"]).abs()

    # Agreement Multiplier: ModelA * ModelD (amplifies when both agree)
    df["Agreement_Mult"] = preds_filled["ModelA_Pred"] * preds_filled["ModelD_Pred"]

    return df


def merge_hca_sensitivity(df: pd.DataFrame, hca_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Merge HCA sensitivity metrics into the matchup dataframe and compute delta.
    Expects hca_profiles with columns: [Season, TeamID, HCA_Sensitivity].
    """
    df = df.copy()
    
    # Team A
    df = df.merge(
        hca_profiles[["Season", "TeamID", "HCA_Sensitivity"]].rename(
            columns={"TeamID": "TeamA", "HCA_Sensitivity": "A_HCA_Sens"}
        ),
        on=["Season", "TeamA"], how="left"
    )
    
    # Team B
    df = df.merge(
        hca_profiles[["Season", "TeamID", "HCA_Sensitivity"]].rename(
            columns={"TeamID": "TeamB", "HCA_Sensitivity": "B_HCA_Sens"}
        ),
        on=["Season", "TeamB"], how="left"
    )
    
    df["Delta_HCA_Sensitivity"] = (df["A_HCA_Sens"].fillna(0) - df["B_HCA_Sens"].fillna(0))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# META-LEARNER FACTORIES
# ═══════════════════════════════════════════════════════════════════════════

def _meta_xgb_factory():
    """XGBoost meta-learner (HPO Optimized for Brier Score)."""
    return xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.030953597876027257,
        colsample_bytree=0.9270114241560119,
        gamma=2.041098381876993,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
    )


def _meta_logreg_factory():
    """LogisticRegressionCV meta-learner — regularized linear stacker."""
    return LogisticRegressionCV(
        Cs=20,
        cv=5,
        penalty="l2",
        scoring="neg_brier_score",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )


def train_meta_learner(oof: pd.DataFrame, use_logreg: bool = False, dropout_rate: float = 0.20) -> object:
    """
    Train the Level 1 meta-learner on OOF base-model predictions.

    Uses ALL rows — XGBoost handles NaN natively, LogReg gets NaN-filled.
    Requires at least ModelA or ModelB to be present.
    If dropout_rate > 0 and using XGBoost, randomly mask base model predictions with np.nan during training.
    """
    # Add interaction features
    oof = add_meta_interactions(oof)

    # Only require at least one base model prediction
    has_any = oof[["ModelA_Pred", "ModelB_Pred"]].notna().any(axis=1)
    usable = oof[has_any].copy()
    print(f"  Meta-learner training on {len(usable):,} matchups "
          f"(dropped {len(oof) - len(usable):,} with no predictions)")

    X = usable[META_FEATURES].values
    y = usable["Result"].values

    if use_logreg:
        # Fill NaN for LogReg (can't handle missing values)
        X = np.nan_to_num(X, nan=0.5)
        meta = _meta_logreg_factory()
        meta_name = "LogisticRegressionCV"
        
        meta.fit(X, y)
    else:
        meta = _meta_xgb_factory()
        meta_name = "XGBoost"
        
        # Apply Input Dropout to Base Models (ModelA/B/D/E)
        X_train_corrupted = X.copy()
        if dropout_rate > 0.0:
            np.random.seed(42)
            num_base_models = 4 # Indices 0, 1, 2, 3
            mask = np.random.rand(X_train_corrupted.shape[0], num_base_models) < dropout_rate
            X_train_corrupted[:, :num_base_models][mask] = np.nan
            print(f"  [Dropout] Applied {dropout_rate*100:.0f}% dropout mask to base model inputs.")
            
        meta.fit(X_train_corrupted, y)

    # In-sample scores (for reference)
    preds_raw = meta.predict_proba(X)[:, 1]
    preds_clipped = np.clip(preds_raw, CLIP_LOW, CLIP_HIGH)
    
    # Final Platt Scaling (Calibration)
    calibrator = LogisticRegression(penalty="l2", C=1.0)
    calibrator.fit(preds_clipped.reshape(-1, 1), y)
    
    # Calibrated in-sample scores
    preds_cal = calibrator.predict_proba(preds_clipped.reshape(-1, 1))[:, 1]
    
    ll = log_loss(y, preds_cal)
    bs = brier_score_loss(y, preds_cal)
    print(f"  ✓ {meta_name} meta-learner in-sample (Calibrated) LL: {ll:.4f}, Brier: {bs:.4f}")

    if not use_logreg:
        # Feature importance for XGBoost
        importance = dict(zip(META_FEATURES, meta.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print(f"  Feature importance:")
        for feat, imp in sorted_imp:
            print(f"    {feat:20s} {imp:.3f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "meta_learner.pkl"
    # Bundle both models
    payload = {"meta": meta, "calibrator": calibrator, "features": META_FEATURES}
    joblib.dump(payload, path)
    print(f"  Saved → {path}")

    return payload


def predict_ensemble(payload: dict,
                     preds_a: np.ndarray,
                     preds_b: np.ndarray,
                     preds_d: np.ndarray,
                     preds_e: np.ndarray,
                     seed_diff: np.ndarray,
                     is_women: np.ndarray,
                     delta_hca: np.ndarray,
                     injury_hedge: np.ndarray = None) -> np.ndarray:
    """
    Generate final ensemble predictions with Platt calibration,
    Seed-Aware dynamic clipping, and optional Injury-Aware hedging.
    """
    meta = payload["meta"]
    calibrator = payload["calibrator"]
    
    df = pd.DataFrame({
        "ModelA_Pred": preds_a,
        "ModelB_Pred": preds_b,
        "ModelD_Pred": preds_d,
        "ModelE_Pred": preds_e,
        "Seed_Diff": seed_diff,
        "is_women": is_women,
        "Delta_HCA_Sensitivity": delta_hca
    })
    
    # Extract robust matrix in correct order
    X = add_meta_interactions(df)[META_FEATURES].values
    X = np.nan_to_num(X, nan=0.5)
    
    raw = meta.predict_proba(X)[:, 1]
    raw_clipped = np.clip(raw, CLIP_LOW, CLIP_HIGH)
    
    # ── Step 1: Apply Platt Calibration ──────────────────────────────────
    calibrated = calibrator.predict_proba(raw_clipped.reshape(-1, 1))[:, 1]
    
    # ── Step 2: Injury-Aware Hedging ──────────────────────────────────────
    # If injury_hedge is 0.2, pulls the prob 20% closer to the 0.5 anchor.
    if injury_hedge is not None:
        calibrated = 0.5 + (calibrated - 0.5) * (1.0 - injury_hedge)
    
    # ── Step 3: Seed-Aware Dynamic Clipping ──────────────────────────────
    seeds_abs = np.abs(seed_diff)
    low_clip = np.where(seeds_abs >= 10, 0.005, CLIP_LOW)
    high_clip = np.where(seeds_abs >= 10, 0.995, CLIP_HIGH)
    
    return np.clip(calibrated, low_clip, high_clip)


# ═══════════════════════════════════════════════════════════════════════════
# QUICK RUN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    oof = pd.read_parquet(PROCESSED_DIR / "oof_predictions.parquet")
    meta = train_meta_learner(oof)
