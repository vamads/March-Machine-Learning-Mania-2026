"""
Unified Early Fusion Model — Train a single LightGBM model on all concatenated features.

Instead of training isolated models for Tabular, Graph, and Elo domains, this approach
concatenates all features together. This allows tree-based models (like LightGBM) to
learn cross-domain interactions directly (e.g., Elo vs. Rebounding rate).
"""

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
import joblib
from pathlib import Path

from src.config import (
    PROCESSED_DIR, MODELS_DIR,
    TABULAR_FEATURES_FILE, GRAPH_EMBEDDINGS_FILE,
    ELO_FEATURES_FILE,
    MIN_TRAIN_SEASONS,
)
from src.tabular.feature_engineering import TABULAR_FEATURE_NAMES
from src.graph.feature_engineering import GRAPH_FEATURE_NAMES
from src.elo.feature_engineering import ELO_FEATURE_NAMES
from src.data_loader import load_tourney_labels


def _unified_lgb_factory():
    """Unified LightGBM binary classifier."""
    return lgb.LGBMClassifier(
        n_estimators=500,  # slightly more trees since we have more features
        max_depth=3,       # slightly deeper to capture cross-domain interactions
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary",
        metric="binary_logloss",
        verbosity=-1,
        random_state=42,
    )


def load_unified_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load Tabular, Graph, and Elo data and merge them on Season and GameID.
    Returns the merged DataFrame and the combined feature list.
    """
    tab = pd.read_parquet(TABULAR_FEATURES_FILE)
    graph = pd.read_parquet(GRAPH_EMBEDDINGS_FILE)
    elo = pd.read_parquet(ELO_FEATURES_FILE)

    labels = load_tourney_labels()

    # Base merge
    unified = labels.merge(tab, on=["Season", "GameID", "Result", "is_women"], how="inner")
    
    # Drop Result, is_women, TeamA, TeamB to avoid duplicates during merge
    graph_sub = graph.drop(columns=["Result", "is_women", "TeamA", "TeamB"], errors="ignore")
    unified = unified.merge(graph_sub, on=["Season", "GameID"], how="inner")

    elo_sub = elo.drop(columns=["Result", "is_women", "TeamA", "TeamB"], errors="ignore")
    unified = unified.merge(elo_sub, on=["Season", "GameID"], how="inner")

    unified_features = TABULAR_FEATURE_NAMES + GRAPH_FEATURE_NAMES + ELO_FEATURE_NAMES
    
    # Sanity check
    missing_features = [f for f in unified_features if f not in unified.columns]
    if missing_features:
        raise ValueError(f"Missing features in unified dataset: {missing_features}")

    return unified, unified_features


def expanding_window_cv_unified(df: pd.DataFrame, feature_cols: list[str],
                                target_col: str, model_factory,
                                model_name: str,
                                train_seasons: list[int] = None,
                                min_train_seasons: int = MIN_TRAIN_SEASONS) -> pd.DataFrame:
    """
    Expanding Window CV for early fusion model.
    """
    seasons = sorted(df["Season"].unique()) if train_seasons is None else train_seasons
    df_train = df[df.Season.isin(seasons)].copy() if train_seasons is not None else df
    oof_rows = []
    n_folds = 0

    print(f"  Expanding Window CV for {model_name} "
          f"({len(seasons)} seasons, min_train={min_train_seasons}) …")

    for test_season in seasons:
        past_seasons = [s for s in seasons if s < test_season]
        if len(past_seasons) < min_train_seasons:
            continue

        train = df_train[df_train.Season.isin(past_seasons)]
        test = df_train[df_train.Season == test_season]

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test = test[feature_cols].values

        model = model_factory()
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        n_folds += 1

        for gid, pred in zip(test["GameID"], preds):
            oof_rows.append({
                "Season": test_season,
                "GameID": gid,
                f"{model_name}_Pred": float(pred),
            })

    oof = pd.DataFrame(oof_rows)

    merged = df_train[["GameID", target_col]].merge(oof, on="GameID")
    ll = log_loss(merged[target_col], merged[f"{model_name}_Pred"])
    first_season = min(r["Season"] for r in oof_rows) if oof_rows else "?"
    print(f"    ✓ {model_name} Expanding CV Log Loss: {ll:.4f} "
          f"({n_folds} folds, first OOF season: {first_season})")

    return oof


def predict_holdout_unified(df: pd.DataFrame, feature_cols: list[str],
                            target_col: str, model_factory, model_name: str,
                            train_seasons: list[int],
                            holdout_seasons: list[int]) -> tuple[pd.DataFrame, object]:
    """
    Train on ALL train_seasons, predict on holdout_seasons.
    Returns (predictions_df, fitted_model).
    """
    train = df[df.Season.isin(train_seasons)]
    holdout = df[df.Season.isin(holdout_seasons)]

    if len(holdout) == 0:
        print(f"    ⚠ {model_name}: no holdout data found!")
        return pd.DataFrame(), None

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_holdout = holdout[feature_cols].values

    model = model_factory()
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_holdout)[:, 1]

    rows = []
    for gid, season, pred in zip(holdout["GameID"], holdout["Season"], preds):
        rows.append({
            "Season": season,
            "GameID": gid,
            f"{model_name}_Pred": float(pred),
            "Result": float(holdout.loc[holdout.GameID == gid, "Result"].iloc[0]),
            "is_women": int(holdout.loc[holdout.GameID == gid, "is_women"].iloc[0])
        })

    pred_df = pd.DataFrame(rows)
    return pred_df, model
