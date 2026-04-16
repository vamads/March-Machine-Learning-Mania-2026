"""
Feature Importance & Redundancy Analysis — Permutation Importance + SHAP.

Two analyses on the 2022–2025 holdout set:

  Part 1: Permutation Importance (Brier Score)
    - Shuffles each feature individually and measures Brier Score delta
    - Runs at two levels: (a) base model features, (b) meta-learner inputs
    - Positive delta = feature was helping; negative = feature was hurting

  Part 2: SHAP Values (Meta-Learner Interdependency)
    - SHAP summary/beeswarm of meta-learner inputs
    - SHAP dependence plot (ModelA vs ModelB interaction)
    - Correlation matrix of SHAP values to detect redundancy

Usage:
    conda activate madness
    python -m src.feature_analysis
"""

import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.config import (
    PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR,
    TABULAR_FEATURES_FILE, GRAPH_EMBEDDINGS_FILE, ORDINALS_FEATURES_FILE,
    ELO_FEATURES_FILE,
    HOLDOUT_SEASONS, TRAIN_ONLY_SEASONS,
    CLIP_LOW, CLIP_HIGH, MIN_TRAIN_SEASONS,
)
from src.tabular.feature_engineering import TABULAR_FEATURE_NAMES
from src.graph.feature_engineering import GRAPH_FEATURE_NAMES
from src.ordinals.feature_engineering import ORDINALS_FEATURE_NAMES
from src.elo.feature_engineering import ELO_FEATURE_NAMES
from src.ensemble.base_models import _xgb_factory, _lgb_factory, _logreg_factory
from src.ensemble.meta_learner import META_FEATURES, _meta_xgb_factory
from src.data_loader import load_tourney_labels


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _train_base_model(df, feature_cols, target_col, model_factory, train_seasons):
    """Train a base model on train_seasons, return fitted model."""
    train = df[df.Season.isin(train_seasons)]
    model = model_factory()
    model.fit(train[feature_cols].values, train[target_col].values)
    return model


def _predict_holdout_features(df, feature_cols, model, holdout_seasons):
    """Predict on holdout data, return (GameIDs, raw_predictions, X_holdout_df)."""
    holdout = df[df.Season.isin(holdout_seasons)].copy()
    if len(holdout) == 0:
        return np.array([]), np.array([]), pd.DataFrame()
    X = holdout[feature_cols].values
    preds = model.predict_proba(X)[:, 1]
    return holdout["GameID"].values, preds, holdout


def _build_holdout_pipeline():
    """
    Build the full holdout pipeline from scratch:
      - Train 4 base models on 2003–2021
      - Predict holdout 2022–2025
      - Train meta-learner on expanding-window OOF from train-only seasons
      - Return everything needed for permutation importance

    Returns: dict with all models, data, predictions, and baseline Brier.
    """
    print("═" * 60)
    print("  BUILDING HOLDOUT PIPELINE FOR ANALYSIS")
    print("═" * 60)

    # Load feature data
    print("\n── Loading feature data ──")
    tab = pd.read_parquet(TABULAR_FEATURES_FILE)
    graph = pd.read_parquet(GRAPH_EMBEDDINGS_FILE)
    ords = pd.read_parquet(ORDINALS_FEATURES_FILE)
    elo = pd.read_parquet(ELO_FEATURES_FILE)
    labels = load_tourney_labels()

    # ── Train base models on TRAIN_ONLY seasons ──
    print("\n── Training base models on 2003–2021 ──")
    model_a = _train_base_model(tab, TABULAR_FEATURE_NAMES, "Result",
                                 _xgb_factory, TRAIN_ONLY_SEASONS)
    model_b = _train_base_model(graph, GRAPH_FEATURE_NAMES, "Result",
                                 _lgb_factory, TRAIN_ONLY_SEASONS)
    model_c = _train_base_model(ords, ORDINALS_FEATURE_NAMES, "Result",
                                 _logreg_factory, TRAIN_ONLY_SEASONS)
    model_d = _train_base_model(elo, ELO_FEATURE_NAMES, "Result",
                                 _lgb_factory, TRAIN_ONLY_SEASONS)
    print("  ✓ All 4 base models trained")

    # ── Build meta-learner training data (expanding window OOF on train-only) ──
    print("\n── Building OOF for meta-learner training ──")
    from src.holdout_evaluate import expanding_window_cv_restricted

    oof_a = expanding_window_cv_restricted(tab, TABULAR_FEATURE_NAMES, "Result",
                                            _xgb_factory, "ModelA", TRAIN_ONLY_SEASONS)
    oof_b = expanding_window_cv_restricted(graph, GRAPH_FEATURE_NAMES, "Result",
                                            _lgb_factory, "ModelB", TRAIN_ONLY_SEASONS)
    oof_c = expanding_window_cv_restricted(ords, ORDINALS_FEATURE_NAMES, "Result",
                                            _logreg_factory, "ModelC", TRAIN_ONLY_SEASONS)
    oof_d = expanding_window_cv_restricted(elo, ELO_FEATURE_NAMES, "Result",
                                            _lgb_factory, "ModelD", TRAIN_ONLY_SEASONS)

    train_labels = labels[labels.Season.isin(TRAIN_ONLY_SEASONS)].copy()
    oof = train_labels.merge(oof_a, on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_b, on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_c, on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_d, on=["Season", "GameID"], how="left")

    tab_seed = tab[["GameID", "Delta_Seed"]].copy().rename(columns={"Delta_Seed": "Seed_Diff"})
    oof = oof.merge(tab_seed, on="GameID", how="left")

    has_any = oof[["ModelA_Pred", "ModelB_Pred"]].notna().any(axis=1)
    oof = oof[has_any]

    # ── Train meta-learner ──
    print("\n── Training XGBoost meta-learner ──")
    meta = _meta_xgb_factory()
    meta.fit(oof[META_FEATURES].values, oof["Result"].values)
    print("  ✓ Meta-learner trained")

    # ── Predict holdout with base models ──
    print("\n── Predicting holdout (2022–2025) ──")
    holdout_labels = labels[labels.Season.isin(HOLDOUT_SEASONS)].copy()

    gid_a, pred_a, hold_tab = _predict_holdout_features(
        tab, TABULAR_FEATURE_NAMES, model_a, HOLDOUT_SEASONS)
    gid_b, pred_b, hold_graph = _predict_holdout_features(
        graph, GRAPH_FEATURE_NAMES, model_b, HOLDOUT_SEASONS)
    gid_c, pred_c, hold_ords = _predict_holdout_features(
        ords, ORDINALS_FEATURE_NAMES, model_c, HOLDOUT_SEASONS)
    gid_d, pred_d, hold_elo = _predict_holdout_features(
        elo, ELO_FEATURE_NAMES, model_d, HOLDOUT_SEASONS)

    # Build holdout meta-features
    holdout = holdout_labels.copy()
    for gids, preds, name in [
        (gid_a, pred_a, "ModelA_Pred"),
        (gid_b, pred_b, "ModelB_Pred"),
        (gid_c, pred_c, "ModelC_Pred"),
        (gid_d, pred_d, "ModelD_Pred"),
    ]:
        pred_df = pd.DataFrame({"GameID": gids, name: preds})
        holdout = holdout.merge(pred_df, on="GameID", how="left")

    holdout = holdout.merge(tab_seed, on="GameID", how="left")

    # Meta-learner prediction
    X_meta = holdout[META_FEATURES].values
    holdout["Pred"] = meta.predict_proba(X_meta)[:, 1]
    holdout["Pred"] = holdout["Pred"].clip(CLIP_LOW, CLIP_HIGH)

    baseline_brier = brier_score_loss(holdout["Result"], holdout["Pred"])
    print(f"\n  Baseline Brier Score: {baseline_brier:.4f}")

    return {
        # Data
        "tab": tab, "graph": graph, "ords": ords, "elo": elo,
        "hold_tab": hold_tab, "hold_graph": hold_graph,
        "hold_ords": hold_ords, "hold_elo": hold_elo,
        "holdout": holdout, "labels": labels, "tab_seed": tab_seed,
        # Models
        "model_a": model_a, "model_b": model_b,
        "model_c": model_c, "model_d": model_d,
        "meta": meta,
        # Baseline
        "baseline_brier": baseline_brier,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: PERMUTATION IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def _permute_base_feature(pipeline, feature_name, model_key, hold_key,
                          feature_cols, n_repeats=5):
    """
    Shuffle one base-model feature, re-predict through the full pipeline,
    and return the average Brier delta.
    """
    holdout = pipeline["holdout"].copy()
    model = pipeline[model_key]
    hold_data = pipeline[hold_key].copy()
    meta = pipeline["meta"]
    rng = np.random.RandomState(42)

    deltas = []
    for _ in range(n_repeats):
        # Shuffle the feature
        shuffled = hold_data.copy()
        shuffled[feature_name] = rng.permutation(shuffled[feature_name].values)

        # Re-predict with base model
        new_preds = model.predict_proba(shuffled[feature_cols].values)[:, 1]
        pred_col = {
            "model_a": "ModelA_Pred", "model_b": "ModelB_Pred",
            "model_c": "ModelC_Pred", "model_d": "ModelD_Pred",
        }[model_key]

        # Update holdout with new base prediction
        new_pred_df = pd.DataFrame({
            "GameID": shuffled["GameID"].values, pred_col: new_preds
        })
        h = holdout.drop(columns=[pred_col]).merge(new_pred_df, on="GameID", how="left")

        # Re-run meta-learner
        X_meta = h[META_FEATURES].values
        h["Pred"] = meta.predict_proba(X_meta)[:, 1]
        h["Pred"] = h["Pred"].clip(CLIP_LOW, CLIP_HIGH)

        new_brier = brier_score_loss(h["Result"], h["Pred"])
        deltas.append(new_brier - pipeline["baseline_brier"])

    return np.mean(deltas), np.std(deltas)


def _permute_meta_feature(pipeline, feature_name, n_repeats=10):
    """
    Shuffle one meta-learner input, re-predict, return Brier delta.
    """
    holdout = pipeline["holdout"].copy()
    meta = pipeline["meta"]
    rng = np.random.RandomState(42)

    deltas = []
    for _ in range(n_repeats):
        h = holdout.copy()
        h[feature_name] = rng.permutation(h[feature_name].values)
        X_meta = h[META_FEATURES].values
        h["Pred"] = meta.predict_proba(X_meta)[:, 1]
        h["Pred"] = h["Pred"].clip(CLIP_LOW, CLIP_HIGH)
        new_brier = brier_score_loss(h["Result"], h["Pred"])
        deltas.append(new_brier - pipeline["baseline_brier"])

    return np.mean(deltas), np.std(deltas)


def run_permutation_importance(pipeline):
    """
    Run permutation importance at two levels:
      1. Individual features within each base model
      2. Meta-learner inputs (base model predictions + Seed_Diff + is_women)
    """
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Level 1: Base Model Features ─────────────────────────────────
    print("\n" + "═" * 60)
    print("  PART 1A: PERMUTATION IMPORTANCE — BASE MODEL FEATURES")
    print("═" * 60)

    base_configs = [
        ("Model A (XGBoost/Tabular)", "model_a", "hold_tab",
         TABULAR_FEATURE_NAMES),
        ("Model B (LightGBM/Graph)", "model_b", "hold_graph",
         GRAPH_FEATURE_NAMES),
        ("Model C (LogReg/Ordinals)", "model_c", "hold_ords",
         ORDINALS_FEATURE_NAMES),
        ("Model D (LightGBM/Elo)", "model_d", "hold_elo",
         ELO_FEATURE_NAMES),
    ]

    all_results = []

    for model_label, model_key, hold_key, feature_cols in base_configs:
        hold_data = pipeline[hold_key]
        if len(hold_data) == 0:
            print(f"\n  ⚠ Skipping {model_label} — no holdout data")
            continue

        print(f"\n  ── {model_label} ({len(feature_cols)} features) ──")
        for feat in tqdm(feature_cols, desc=f"  {model_label}"):
            mean_delta, std_delta = _permute_base_feature(
                pipeline, feat, model_key, hold_key, feature_cols, n_repeats=5
            )
            all_results.append({
                "Model": model_label,
                "Feature": feat,
                "Brier_Delta_Mean": mean_delta,
                "Brier_Delta_Std": std_delta,
            })

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("Brier_Delta_Mean", ascending=False)

    print(f"\n{'─' * 75}")
    print(f"  {'Feature':40s} {'Model':25s} {'Δ Brier':>10s}")
    print(f"{'─' * 75}")
    for _, row in results_df.iterrows():
        sign = "+" if row.Brier_Delta_Mean >= 0 else ""
        status = "✓ Helps" if row.Brier_Delta_Mean > 0.0001 else \
                 "✗ Hurts" if row.Brier_Delta_Mean < -0.0001 else \
                 "─ Neutral"
        print(f"  {row.Feature:40s} {row.Model:25s} "
              f"{sign}{row.Brier_Delta_Mean:.6f}  {status}")

    # Plot base model feature importance
    fig, ax = plt.subplots(figsize=(12, max(6, len(results_df) * 0.3)))
    colors = ['#4CAF50' if d > 0.0001 else '#F44336' if d < -0.0001 else '#9E9E9E'
              for d in results_df.Brier_Delta_Mean]
    ax.barh(range(len(results_df)), results_df.Brier_Delta_Mean, color=colors,
            xerr=results_df.Brier_Delta_Std, capsize=2, alpha=0.85)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels([f"{r.Feature}  ({r.Model.split('/')[0].split('(')[1]})"
                        for _, r in results_df.iterrows()], fontsize=7)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel("Δ Brier Score (positive = feature helps)", fontsize=11)
    ax.set_title("Permutation Importance — Base Model Features\n"
                 "(Holdout 2022–2025, shuffled → re-predicted through meta-learner)",
                 fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    path = SUBMISSIONS_DIR / "permutation_importance_base_models.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved → {path}")

    # ── Level 2: Meta-Learner Inputs ─────────────────────────────────
    print("\n" + "═" * 60)
    print("  PART 1B: PERMUTATION IMPORTANCE — META-LEARNER INPUTS")
    print("═" * 60)

    meta_results = []
    for feat in META_FEATURES:
        print(f"  Shuffling {feat} …")
        mean_delta, std_delta = _permute_meta_feature(pipeline, feat, n_repeats=10)
        meta_results.append({
            "Feature": feat,
            "Brier_Delta_Mean": mean_delta,
            "Brier_Delta_Std": std_delta,
        })

    meta_df = pd.DataFrame(meta_results)
    meta_df = meta_df.sort_values("Brier_Delta_Mean", ascending=False)

    print(f"\n{'─' * 55}")
    print(f"  {'Meta Feature':25s} {'Δ Brier':>12s} {'Std':>10s}")
    print(f"{'─' * 55}")
    for _, row in meta_df.iterrows():
        sign = "+" if row.Brier_Delta_Mean >= 0 else ""
        print(f"  {row.Feature:25s} {sign}{row.Brier_Delta_Mean:.6f}   "
              f"±{row.Brier_Delta_Std:.6f}")

    # Plot meta-learner importance
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#2196F3' if d > 0 else '#FF9800' for d in meta_df.Brier_Delta_Mean]
    bars = ax.barh(range(len(meta_df)), meta_df.Brier_Delta_Mean,
                   xerr=meta_df.Brier_Delta_Std, color=colors, capsize=4, alpha=0.85)
    ax.set_yticks(range(len(meta_df)))
    ax.set_yticklabels(meta_df.Feature, fontsize=11)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel("Δ Brier Score (positive = input helps ensemble)", fontsize=11)
    ax.set_title("Permutation Importance — Meta-Learner Inputs\n"
                 "(Holdout 2022–2025)", fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    path = SUBMISSIONS_DIR / "permutation_importance_meta.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved → {path}")

    return results_df, meta_df


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: SHAP VALUES
# ═══════════════════════════════════════════════════════════════════════════

def run_shap_analysis(pipeline):
    """
    SHAP analysis of the XGBoost meta-learner to understand:
      - Which base models contribute most to each prediction
      - Whether any base models are redundant (correlated SHAP impacts)
    """
    import shap

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  PART 2: SHAP VALUES — META-LEARNER INTERDEPENDENCY")
    print("═" * 60)

    holdout = pipeline["holdout"]
    meta = pipeline["meta"]

    X_holdout = holdout[META_FEATURES].copy()
    X_holdout.columns = META_FEATURES

    # Compute SHAP values using TreeExplainer (fast for XGBoost)
    print("\n  Computing SHAP values …")
    explainer = shap.TreeExplainer(meta)
    shap_values = explainer(X_holdout)

    # ── Summary / Beeswarm Plot ──────────────────────────────────────
    print("  Generating SHAP summary plot …")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.beeswarm(shap_values, show=False, max_display=len(META_FEATURES))
    plt.title("SHAP Summary — Meta-Learner Inputs\n"
              "(each dot = one holdout game)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = SUBMISSIONS_DIR / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")

    # ── Mean Absolute SHAP ───────────────────────────────────────────
    print("\n  Mean |SHAP| per meta-learner input:")
    shap_array = shap_values.values
    mean_abs = np.mean(np.abs(shap_array), axis=0)
    for feat, val in sorted(zip(META_FEATURES, mean_abs),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(val / max(mean_abs) * 30)
        print(f"    {feat:20s}  {val:.4f}  {bar}")

    # ── SHAP Bar Plot ────────────────────────────────────────────────
    print("  Generating SHAP bar plot …")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.bar(shap_values, show=False, max_display=len(META_FEATURES))
    plt.title("Mean |SHAP| — Feature Importance in Meta-Learner",
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = SUBMISSIONS_DIR / "shap_bar.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")

    # ── Dependence Plot: ModelA vs ModelB ─────────────────────────────
    print("  Generating SHAP dependence plot (ModelA vs ModelB) …")
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.scatter(shap_values[:, "ModelA_Pred"],
                       color=shap_values[:, "ModelB_Pred"],
                       show=False)
    plt.title("SHAP Dependence — ModelA colored by ModelB\n"
              "(similar patterns → redundancy)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = SUBMISSIONS_DIR / "shap_dependence.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")

    # ── SHAP Correlation Matrix (Redundancy Test) ────────────────────
    print("\n  Computing SHAP value correlation matrix …")
    shap_df = pd.DataFrame(shap_array, columns=META_FEATURES)
    corr = shap_df.corr()

    print(f"\n  SHAP Correlation Matrix:")
    print(f"  {'':20s}", end="")
    for f in META_FEATURES:
        print(f"  {f[:8]:>8s}", end="")
    print()
    for i, fi in enumerate(META_FEATURES):
        print(f"  {fi:20s}", end="")
        for j, fj in enumerate(META_FEATURES):
            val = corr.iloc[i, j]
            print(f"  {val:>8.3f}", end="")
        print()

    # Highlight high correlations (redundancy)
    print(f"\n  Redundancy flags (|r| > 0.7 between base models):")
    model_features = ["ModelA_Pred", "ModelB_Pred", "ModelC_Pred", "ModelD_Pred"]
    found_redundancy = False
    for i, fi in enumerate(model_features):
        for j, fj in enumerate(model_features):
            if j <= i:
                continue
            r = corr.loc[fi, fj]
            if abs(r) > 0.7:
                print(f"    ⚠ {fi} ↔ {fj}: r = {r:.3f} (HIGH REDUNDANCY)")
                found_redundancy = True
            elif abs(r) > 0.4:
                print(f"    ~ {fi} ↔ {fj}: r = {r:.3f} (moderate overlap)")
                found_redundancy = True
    if not found_redundancy:
        print(f"    ✓ No high redundancy detected between base models")

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(META_FEATURES)))
    ax.set_yticks(range(len(META_FEATURES)))
    ax.set_xticklabels(META_FEATURES, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(META_FEATURES, fontsize=10)
    for i in range(len(META_FEATURES)):
        for j in range(len(META_FEATURES)):
            val = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=9, color=color)
    plt.colorbar(im, label='Pearson r')
    ax.set_title("SHAP Value Correlation — Meta-Learner Inputs\n"
                 "(high |r| between base models = redundancy)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = SUBMISSIONS_DIR / "shap_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved → {path}")

    return shap_values, corr


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Build the holdout pipeline
    pipeline = _build_holdout_pipeline()

    # Part 1: Permutation Importance
    base_imp, meta_imp = run_permutation_importance(pipeline)

    # Part 2: SHAP Values
    shap_vals, shap_corr = run_shap_analysis(pipeline)

    print("\n" + "═" * 60)
    print("  ✓ FEATURE ANALYSIS COMPLETE")
    print("═" * 60)
    print(f"\n  Plots saved to: {SUBMISSIONS_DIR}/")
    print("    • permutation_importance_base_models.png")
    print("    • permutation_importance_meta.png")
    print("    • shap_summary.png")
    print("    • shap_bar.png")
    print("    • shap_dependence.png")
    print("    • shap_correlation.png")
