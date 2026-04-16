"""
True Temporal Holdout Evaluation — Train on 2003-2021, test on 2022-2025.

This simulates production conditions: the model has NEVER seen any tournament
outcomes from the holdout years, exactly like predicting 2026.

Compares results against the current pipeline's scores to quantify improvement.

Usage:
    conda activate madness
    python -m src.holdout_evaluate
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
import joblib
from pathlib import Path

from src.config import (
    PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR,
    TABULAR_FEATURES_FILE, GRAPH_EMBEDDINGS_FILE,
    ELO_FEATURES_FILE,
    HOLDOUT_SEASONS, TRAIN_ONLY_SEASONS,
    MIN_TRAIN_SEASONS,
)
from src.tabular.feature_engineering import TABULAR_FEATURE_NAMES
from src.graph.feature_engineering import GRAPH_FEATURE_NAMES
from src.elo.feature_engineering import ELO_FEATURE_NAMES
from src.ensemble.base_models import _xgb_factory, _lgb_factory
from src.ensemble.meta_learner import (
    META_FEATURES, CLIP_LOW, CLIP_HIGH,
    _meta_xgb_factory, _meta_logreg_factory, add_meta_interactions
)
from src.data_loader import load_tourney_labels


# ═══════════════════════════════════════════════════════════════════════════
# 1.  EXPANDING WINDOW CV ON TRAIN-ONLY SEASONS (2003-2021)
# ═══════════════════════════════════════════════════════════════════════════

def expanding_window_cv_restricted(df: pd.DataFrame, feature_cols: list[str],
                                   target_col: str, model_factory,
                                   model_name: str,
                                   train_seasons: list[int],
                                   min_train_seasons: int = MIN_TRAIN_SEASONS) -> pd.DataFrame:
    """
    Expanding Window CV restricted to train_seasons only.
    For each season S in train_seasons: train on all train_seasons < S, predict S.
    Skips until min_train_seasons of past data are available.
    """
    df_train = df[df.Season.isin(train_seasons)].copy()
    seasons = sorted(df_train.Season.unique())
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

    # Overall expanding-window log loss on train-only seasons
    merged = df_train[["GameID", target_col]].merge(oof, on="GameID")
    ll = log_loss(merged[target_col], merged[f"{model_name}_Pred"])
    first_season = min(r["Season"] for r in oof_rows) if oof_rows else "?"
    print(f"    ✓ {model_name} Expanding CV LL (train-only): {ll:.4f} "
          f"({n_folds} folds, first OOF: {first_season})")

    return oof


# ═══════════════════════════════════════════════════════════════════════════
# 2.  PREDICT HOLDOUT SEASONS
# ═══════════════════════════════════════════════════════════════════════════

def predict_holdout(df: pd.DataFrame, feature_cols: list[str],
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
        })

    pred_df = pd.DataFrame(rows)
    return pred_df, model


# ═══════════════════════════════════════════════════════════════════════════
# 3.  MAIN HOLDOUT EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def run_holdout_evaluation():
    """
    Full temporal holdout evaluation pipeline.

    Train on 2003-2021, test on 2022-2025.
    Uses 3 base models + XGBoost meta-learner.
    """
    print("=" * 60)
    print("  TRUE TEMPORAL HOLDOUT EVALUATION")
    print(f"  Train: {TRAIN_ONLY_SEASONS[0]}–{TRAIN_ONLY_SEASONS[-1]} "
          f"({len(TRAIN_ONLY_SEASONS)} seasons)")
    print(f"  Test:  {HOLDOUT_SEASONS}")
    print("=" * 60)

    # Load feature files
    print("\n── Loading feature data ──")
    tab = pd.read_parquet(TABULAR_FEATURES_FILE)
    graph = pd.read_parquet(GRAPH_EMBEDDINGS_FILE)
    elo = pd.read_parquet(ELO_FEATURES_FILE)
    labels = load_tourney_labels()

    print(f"  Tabular:  {len(tab):,} rows, seasons {tab.Season.min()}–{tab.Season.max()}")
    print(f"  Graph:    {len(graph):,} rows, seasons {graph.Season.min()}–{graph.Season.max()}")
    print(f"  Elo:      {len(elo):,} rows, seasons {elo.Season.min()}–{elo.Season.max()}")

    # ── Step 1: Expanding Window CV on train-only seasons ──
    print("\n── Step 1: Expanding Window CV on train-only seasons (2003-2021) ──")

    oof_a = expanding_window_cv_restricted(tab, TABULAR_FEATURE_NAMES, "Result",
                                _xgb_factory, "ModelA", TRAIN_ONLY_SEASONS)
    oof_b = expanding_window_cv_restricted(graph, GRAPH_FEATURE_NAMES, "Result",
                                _lgb_factory, "ModelB", TRAIN_ONLY_SEASONS)
    oof_d = expanding_window_cv_restricted(elo, ELO_FEATURE_NAMES, "Result",
                                _lgb_factory, "ModelD", TRAIN_ONLY_SEASONS)

    # Combine OOF predictions
    train_labels = labels[labels.Season.isin(TRAIN_ONLY_SEASONS)].copy()
    oof = train_labels.merge(oof_a, on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_b, on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_d, on=["Season", "GameID"], how="left")

    # Add Seed_Diff
    tab_seed = tab[["GameID", "Delta_Seed"]].copy()
    tab_seed = tab_seed.rename(columns={"Delta_Seed": "Seed_Diff"})
    oof = oof.merge(tab_seed, on="GameID", how="left")

    has_any = oof[["ModelA_Pred", "ModelB_Pred"]].notna().any(axis=1)
    usable = oof[has_any]
    print(f"\n  Combined OOF: {len(usable):,} usable rows "
          f"(dropped {len(oof) - len(usable):,})")

    # ── Step 2: Train meta-learner on train-only OOF ──
    print("\n── Step 2: Train XGBoost meta-learner (2003-2021 only) ──")

    X_meta_train = usable[META_FEATURES].values
    y_meta_train = usable["Result"].values

    meta = _meta_xgb_factory()
    
    # --- Meta-Learner Input Dropout ---
    dropout_rate = 0.0
    X_train_corrupted = X_meta_train.copy()
    if dropout_rate > 0.0:
        np.random.seed(42)
        num_base_models = 3 # Indices 0, 1, 2 for A, B, D
        mask = np.random.rand(X_train_corrupted.shape[0], num_base_models) < dropout_rate
        X_train_corrupted[:, :num_base_models][mask] = np.nan
        print(f"  [Dropout] Applied {dropout_rate*100:.0f}% dropout mask to base model inputs.")
        
    meta.fit(X_train_corrupted, y_meta_train)

    meta_preds_train = meta.predict_proba(X_meta_train)[:, 1]
    meta_preds_clipped = np.clip(meta_preds_train, CLIP_LOW, CLIP_HIGH)
    train_ll = log_loss(y_meta_train, meta_preds_clipped)
    train_bs = brier_score_loss(y_meta_train, meta_preds_clipped)
    print(f"  Meta-learner in-sample LL: {train_ll:.4f}, Brier: {train_bs:.4f}")

    # Fit Isotonic calibrator on OOF predictions
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(meta_preds_clipped, y_meta_train)
    print("  ✓ Isotonic calibrator fitted on OOF predictions")

    # ── Step 3: Predict holdout seasons ──
    print("\n── Step 3: Predict holdout seasons (2022-2025) ──")

    pred_a, _ = predict_holdout(tab, TABULAR_FEATURE_NAMES, "Result",
                                _xgb_factory, "ModelA",
                                TRAIN_ONLY_SEASONS, HOLDOUT_SEASONS)
    pred_b, _ = predict_holdout(graph, GRAPH_FEATURE_NAMES, "Result",
                                _lgb_factory, "ModelB",
                                TRAIN_ONLY_SEASONS, HOLDOUT_SEASONS)
    pred_d, _ = predict_holdout(elo, ELO_FEATURE_NAMES, "Result",
                                _lgb_factory, "ModelD",
                                TRAIN_ONLY_SEASONS, HOLDOUT_SEASONS)

    # Combine holdout predictions
    holdout_labels = labels[labels.Season.isin(HOLDOUT_SEASONS)].copy()
    holdout = holdout_labels.merge(pred_a, on=["Season", "GameID"], how="left")
    holdout = holdout.merge(pred_b, on=["Season", "GameID"], how="left")
    holdout = holdout.merge(pred_d, on=["Season", "GameID"], how="left")

    # Add Seed_Diff for holdout games
    holdout = holdout.merge(tab_seed, on="GameID", how="left")

    # Ensemble prediction via XGBoost meta-learner (handles NaN natively)
    X_holdout_meta = holdout[META_FEATURES].values
    raw_preds = meta.predict_proba(X_holdout_meta)[:, 1]

    # Apply Isotonic calibration, then clip
    holdout["Pred_Raw"] = raw_preds
    holdout["Pred"] = calibrator.predict(raw_preds)
    holdout["Pred"] = holdout["Pred"].clip(CLIP_LOW, CLIP_HIGH)

    print(f"\n  Holdout predictions: {len(holdout):,} games")
    for col in ["ModelA_Pred", "ModelB_Pred", "ModelD_Pred"]:
        n = holdout[col].notna().sum()
        print(f"  {col}: {n:,}/{len(holdout):,} coverage")

    # ── Step 4: Score ──
    print("\n" + "=" * 60)
    print("  HOLDOUT RESULTS (TRUE OUT-OF-SAMPLE)")
    print("=" * 60)

    overall_ll = log_loss(holdout.Result, holdout.Pred)
    overall_bs = brier_score_loss(holdout.Result, holdout.Pred)
    holdout["Correct"] = ((holdout.Pred > 0.5).astype(int) == holdout.Result).astype(int)
    overall_acc = holdout.Correct.mean()

    print(f"\n  Overall Log Loss:   {overall_ll:.4f}")
    print(f"  Overall Brier:      {overall_bs:.4f}")
    print(f"  Overall Accuracy:   {overall_acc:.1%} "
          f"({holdout.Correct.sum()}/{len(holdout)})")

    # Per season
    print(f"\n{'─'*55}")
    print(f"  {'Season':<8} {'Log Loss':>10} {'Brier':>10} {'Accuracy':>10} {'N Games':>8}")
    print(f"{'─'*55}")
    for s in sorted(holdout.Season.unique()):
        m = holdout[holdout.Season == s]
        s_ll = log_loss(m.Result, m.Pred)
        s_bs = brier_score_loss(m.Result, m.Pred)
        s_acc = m.Correct.mean()
        print(f"  {s:<8} {s_ll:>10.4f} {s_bs:>10.4f} {s_acc:>10.1%} {len(m):>8}")

    # By gender
    print(f"\n{'─'*55}")
    for label, is_w in [("Men's", 0), ("Women's", 1)]:
        m = holdout[holdout.is_women == is_w]
        if len(m) > 0:
            g_ll = log_loss(m.Result, m.Pred)
            g_bs = brier_score_loss(m.Result, m.Pred)
            g_acc = m.Correct.mean()
            print(f"  {label:<10} LL={g_ll:.4f}  Brier={g_bs:.4f}  "
                  f"Acc={g_acc:.1%}  ({len(m)} games)")

    # ── Step 5: Compare with previous baseline ──
    print("\n" + "=" * 60)
    print("  COMPARISON WITH PREVIOUS BASELINE")
    print("=" * 60)
    print(f"\n  Previous holdout (4-model, with Massey ordinals):")
    print(f"    LL=0.5189  Brier=0.1759  Acc=74.0%")
    print(f"\n  Current holdout (3-model, no ordinals):")
    print(f"    LL={overall_ll:.4f}  Brier={overall_bs:.4f}  Acc={overall_acc:.1%}")
    ll_delta = overall_ll - 0.5189
    bs_delta = overall_bs - 0.1759
    acc_delta = overall_acc - 0.740
    print(f"\n  Delta: LL={ll_delta:+.4f}  Brier={bs_delta:+.4f}  Acc={acc_delta:+.1%}")

    print(f"\n{'='*60}")
    print("  ✓ Holdout evaluation complete")
    print(f"{'='*60}\n")

    return holdout


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_holdout_evaluation()
