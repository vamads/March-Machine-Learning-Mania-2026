"""
Base Models — Train individual Level 0 models with Expanding Window CV.

Three base models:
  Model A (XGBoost):   Tabular delta features (Four Factors + Madness Metrics + Seed)
  Model B (LightGBM):  Graph delta features (PageRank, HITS, Dominance, WinFrac)
  Model D (LightGBM):  Elo delta features (Elo_Final + Elo_Recent)

Each model produces Out-Of-Fold (OOF) predictions via Expanding Window CV,
where for each test season S we train on all seasons < S (past only).
This prevents future-data leakage and matches production conditions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import joblib
from pathlib import Path

from src.config import (
    PROCESSED_DIR, MODELS_DIR,
    TABULAR_FEATURES_FILE, GRAPH_EMBEDDINGS_FILE,
    ELO_FEATURES_FILE,
    TRAIN_SEASONS, CLIP_LOW, CLIP_HIGH, MIN_TRAIN_SEASONS,
)
from src.tabular.feature_engineering import TABULAR_FEATURE_NAMES
from src.graph.feature_engineering import GRAPH_FEATURE_NAMES
from src.elo.feature_engineering import ELO_FEATURE_NAMES


# ═══════════════════════════════════════════════════════════════════════════
# EXPANDING WINDOW CV ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def expanding_window_cv(df: pd.DataFrame, feature_cols: list[str],
                        target_col: str, model_factory,
                        model_name: str,
                        min_train_seasons: int = MIN_TRAIN_SEASONS) -> pd.DataFrame:
    """
    Expanding Window Cross-Validation (no future data leakage).

    For each season S in the data:
      - Train on all seasons < S (past only)
      - Predict on season S
    Skips early seasons until min_train_seasons of data are available.

    Parameters
    ----------
    df : DataFrame with features, targets, and a Season column
    feature_cols : list of column names for X
    target_col : column name for y
    model_factory : callable() → model with .fit(X, y) and .predict_proba(X)
    model_name : str, used for logging
    min_train_seasons : int, minimum past seasons required before predicting

    Returns
    -------
    DataFrame with columns: Season, GameID, {model_name}_Pred
    """
    seasons = sorted(df["Season"].unique())
    oof_rows = []
    n_folds = 0

    print(f"  Expanding Window CV for {model_name} "
          f"({len(seasons)} seasons, min_train={min_train_seasons}) …")

    for i, test_season in enumerate(seasons):
        past_seasons = [s for s in seasons if s < test_season]
        if len(past_seasons) < min_train_seasons:
            continue

        train = df[df.Season.isin(past_seasons)]
        test  = df[df.Season == test_season]

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test  = test[feature_cols].values

        model = model_factory()
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        n_folds += 1

        for gid, pred in zip(test["GameID"], preds):
            oof_rows.append({
                "Season":       test_season,
                "GameID":       gid,
                f"{model_name}_Pred": float(pred),
            })

    oof = pd.DataFrame(oof_rows)

    # Compute overall expanding-window log loss
    merged = df[["GameID", target_col]].merge(oof, on="GameID")
    ll = log_loss(merged[target_col], merged[f"{model_name}_Pred"])
    first_season = min(r["Season"] for r in oof_rows) if oof_rows else "?"
    print(f"    ✓ {model_name} Expanding CV Log Loss: {ll:.4f} "
          f"({n_folds} folds, first OOF season: {first_season})")

    return oof


# ═══════════════════════════════════════════════════════════════════════════
# MODEL FACTORIES
# ═══════════════════════════════════════════════════════════════════════════

def _xgb_factory():
    """XGBoost binary classifier (Optuna-tuned baseline)."""
    return xgb.XGBClassifier(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.019,
        subsample=0.88,
        colsample_bytree=0.57,
        min_child_weight=10,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
    )


def _lgb_factory():
    """LightGBM binary classifier (Optuna-tuned baseline)."""
    return lgb.LGBMClassifier(
        n_estimators=400,
        max_depth=2,
        learning_rate=0.035,
        subsample=0.96,
        colsample_bytree=0.93,
        objective="binary",
        metric="binary_logloss",
        verbosity=-1,
        random_state=42,
    )


def _logreg_factory():
    """Logistic Regression with regularization."""
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def train_all_base_models() -> pd.DataFrame:
    """
    Train all four base models with Expanding Window CV and return combined
    OOF predictions aligned by GameID.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Model A: XGBoost on tabular features ──────────────────────────
    print("\n══ Model A: XGBoost (Tabular) ══")
    tab = pd.read_parquet(TABULAR_FEATURES_FILE)
    oof_a = expanding_window_cv(tab, TABULAR_FEATURE_NAMES, "Result",
                                _xgb_factory, "ModelA")

    # ── Model B: LightGBM on graph features ───────────────────────────
    print("\n══ Model B: LightGBM (Graph) ══")
    graph = pd.read_parquet(GRAPH_EMBEDDINGS_FILE)
    oof_b = expanding_window_cv(graph, GRAPH_FEATURE_NAMES, "Result",
                                _lgb_factory, "ModelB")

    # ── Model D: LightGBM on Elo features ───────────────────────────────
    print("\n══ Model D: LightGBM (Elo) ══")
    elo = pd.read_parquet(ELO_FEATURES_FILE)
    oof_d = expanding_window_cv(elo, ELO_FEATURE_NAMES, "Result",
                                _lgb_factory, "ModelD")

    # ── Model E: Raddar-Cauchy Leaf-Node Stacker ─────────────────────
    print("\n══ Model E: Raddar-Cauchy (Leaf Stack) ══")
    from src.ensemble.raddar_cauchy import get_model_e_oof
    oof_e = get_model_e_oof()

    # ── Combine OOF predictions ───────────────────────────────────────
    # Start from the full labels so we keep Result and is_women
    from src.data_loader import load_tourney_labels
    labels = load_tourney_labels()

    combined = labels.merge(oof_a, on=["Season", "GameID"], how="left")
    combined = combined.merge(oof_b, on=["Season", "GameID"], how="left")
    combined = combined.merge(oof_d, on=["Season", "GameID"], how="left")
    combined = combined.merge(oof_e, on=["Season", "GameID"], how="left")

    # Also merge Seed_Diff for the meta-learner
    tab_seed = pd.read_parquet(TABULAR_FEATURES_FILE)[["GameID", "Delta_Seed"]]
    combined = combined.merge(tab_seed, on="GameID", how="left")
    # Rename for clarity
    combined = combined.rename(columns={"Delta_Seed": "Seed_Diff"})

    print(f"\n  Combined OOF shape: {combined.shape}")
    print(f"  ModelA coverage: {combined.ModelA_Pred.notna().sum():,}")
    print(f"  ModelB coverage: {combined.ModelB_Pred.notna().sum():,}")
    print(f"  ModelD coverage: {combined.ModelD_Pred.notna().sum():,}")
    print(f"  ModelE coverage: {combined.ModelE_Pred.notna().sum():,}")
    print(f"  Seed_Diff coverage: {combined.Seed_Diff.notna().sum():,}")

    # Save combined OOF
    oof_path = PROCESSED_DIR / "oof_predictions.parquet"
    combined.to_parquet(oof_path, index=False)
    print(f"  Saved → {oof_path}")

    return combined


def train_final_models(save: bool = True) -> dict:
    """
    Train each base model on ALL historical data (for 2026 inference).
    Returns dict of fitted models.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}

    # Model A
    tab = pd.read_parquet(TABULAR_FEATURES_FILE)
    model_a = _xgb_factory()
    model_a.fit(tab[TABULAR_FEATURE_NAMES].values, tab["Result"].values)
    models["ModelA"] = (model_a, TABULAR_FEATURE_NAMES)

    # Model B
    graph = pd.read_parquet(GRAPH_EMBEDDINGS_FILE)
    model_b = _lgb_factory()
    model_b.fit(graph[GRAPH_FEATURE_NAMES].values, graph["Result"].values)
    models["ModelB"] = (model_b, GRAPH_FEATURE_NAMES)

    # Model D
    elo = pd.read_parquet(ELO_FEATURES_FILE)
    model_d = _lgb_factory()
    model_d.fit(elo[ELO_FEATURE_NAMES].values, elo["Result"].values)
    models["ModelD"] = (model_d, ELO_FEATURE_NAMES)

    # Model E
    from src.ensemble.raddar_cauchy import train_final_model_e, RADDAR_FEATURES
    model_e = train_final_model_e()
    models["ModelE"] = (model_e, RADDAR_FEATURES)

    if save:
        for name, (model, _) in models.items():
            path = MODELS_DIR / f"{name}.pkl"
            joblib.dump(model, path)
            print(f"  Saved {name} → {path}")

    return models


# ── Quick Run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_final_models()
