"""
Nested Cross-Validation for Hyperparameter Tuning.

Simulates strict temporal evaluation by isolating the Optuna tuning process.
For each holdout year (e.g. 2022), it limits training data strictly to past years
(e.g. 2003-2021). It launches an independent Optuna study on this constrained horizon
to find hyperparameters, trains on the horizon, and predicts the single holdout year.

Optimized for:
- Pre-computing Model E OOF (50x speed increase)
- Resilience (Resume capability)
- Transparency (Timestamped CSV logs)
"""

import argparse
import pandas as pd
import numpy as np
import xgboost as xgb_lib
import lightgbm as lgb_lib
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import optuna
import json
import csv
from datetime import datetime
from pathlib import Path

from src.config import (
    MODELS_DIR, TABULAR_FEATURES_FILE, GRAPH_EMBEDDINGS_FILE, ELO_FEATURES_FILE,
    HOLDOUT_SEASONS, TRAIN_SEASONS, MIN_TRAIN_SEASONS, CLIP_LOW, CLIP_HIGH,
    ELO_INIT, FIRST_COMPACT_SEASON_M, FIRST_COMPACT_SEASON_W
)
from src.tabular.feature_engineering import TABULAR_FEATURE_NAMES
from src.graph.feature_engineering import GRAPH_FEATURE_NAMES
from src.elo.feature_engineering import (
    ELO_FEATURE_NAMES, compute_elo_ratings, fit_movda_params, _merge_elo_to_matchups
)
from src.ensemble.meta_learner import META_FEATURES, add_meta_interactions
from src.data_loader import load_regular_season_compact, load_tourney_labels
from src.ensemble.tune import _expanding_cv, _rebuild_elo_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

def _nested_objective(trial, available_seasons, tab, graph, compact_m, compact_w,
                      movda_m, movda_w, labels, tab_seed, oof_e):
    """Inner Optuna objective using pre-computed Model E OOF."""
    xgb_depth = trial.suggest_int("xgb_depth", 2, 6)
    xgb_lr = trial.suggest_float("xgb_lr", 0.01, 0.2, log=True)
    xgb_n = trial.suggest_int("xgb_n", 50, 250, step=50)

    def xgb_factory():
        return xgb_lib.XGBClassifier(
            n_estimators=xgb_n, max_depth=xgb_depth, learning_rate=xgb_lr,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, verbosity=0, random_state=42
        )

    lgb_depth = trial.suggest_int("lgb_depth", 2, 6)
    lgb_lr = trial.suggest_float("lgb_lr", 0.01, 0.2, log=True)
    lgb_n = trial.suggest_int("lgb_n", 50, 250, step=50)

    def lgb_factory():
        return lgb_lib.LGBMClassifier(
            n_estimators=lgb_n, max_depth=lgb_depth, learning_rate=lgb_lr,
            objective="binary", metric="binary_logloss", verbosity=-1, random_state=42
        )

    elo_k = trial.suggest_float("elo_k", 20.0, 40.0)
    elo_k_decay = trial.suggest_float("elo_k_decay", 0.2, 0.8)
    elo_regress = trial.suggest_float("elo_regress", 0.4, 0.8)
    elo_lambda = trial.suggest_float("elo_lambda", 0.5, 1.5)

    elo = _rebuild_elo_features(
        compact_m, compact_w, movda_m, movda_w, labels,
        elo_k, elo_k_decay, elo_regress, elo_lambda,
        seasons=available_seasons
    )

    lgb_elo_depth = trial.suggest_int("lgb_elo_depth", 1, 3)
    lgb_elo_n = trial.suggest_int("lgb_elo_n", 50, 200, step=50)

    def lgb_elo_factory():
        return lgb_lib.LGBMClassifier(
            n_estimators=lgb_elo_n, max_depth=lgb_elo_depth, learning_rate=lgb_lr,
            objective="binary", metric="binary_logloss", verbosity=-1, random_state=42
        )

    meta_depth = trial.suggest_int("meta_depth", 1, 3)
    meta_lr = trial.suggest_float("meta_lr", 0.01, 0.2, log=True)
    meta_n = trial.suggest_int("meta_n", 50, 200, step=50)

    def meta_factory():
        return xgb_lib.XGBClassifier(
            n_estimators=meta_n, max_depth=meta_depth, learning_rate=meta_lr,
            objective="binary:logistic", eval_metric="logloss",
            use_label_encoder=False, verbosity=0, random_state=42
        )

    # Base models OOF (fast)
    oof_a = _expanding_cv(tab, TABULAR_FEATURE_NAMES, "Result", xgb_factory, available_seasons)
    oof_b = _expanding_cv(graph, GRAPH_FEATURE_NAMES, "Result", lgb_factory, available_seasons)
    oof_d = _expanding_cv(elo, ELO_FEATURE_NAMES, "Result", lgb_elo_factory, available_seasons)
    
    oof_labels = labels[labels.Season.isin(available_seasons)].copy()
    oof = oof_labels[["Season", "GameID", "Result", "is_women"]].copy()
    oof = oof.merge(oof_a.rename(columns={"pred": "ModelA_Pred"}), on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_b.rename(columns={"pred": "ModelB_Pred"}), on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_d.rename(columns={"pred": "ModelD_Pred"}), on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_e, on=["Season", "GameID"], how="left")
    oof = oof.merge(tab_seed, on="GameID", how="left")
    
    oof = oof.dropna(subset=["ModelA_Pred", "ModelB_Pred", "ModelD_Pred", "ModelE_Pred"])
    if len(oof) < 100: return 1.0

    oof = add_meta_interactions(oof)
    meta_oof = _expanding_cv(oof, META_FEATURES, "Result", meta_factory, available_seasons, min_train=3)
    if len(meta_oof) < 100: return 1.0

    meta_oof = meta_oof.merge(oof[["GameID", "Result"]], on="GameID")
    raw_preds = meta_oof["pred"].values
    y_true = meta_oof["Result"].values
    return log_loss(y_true, raw_preds)

def run_nested_cv(n_trials=50, fresh=False):
    print("=" * 60)
    print("  NESTED CROSS-VALIDATION HYPERPARAMETER TUNING")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = MODELS_DIR / "nested_cv_trials.csv"
    if fresh and log_path.exists():
        log_path.unlink()
        print(f"  [!] Deleted existing logs for fresh start.")

    tab = pd.read_parquet(TABULAR_FEATURES_FILE)
    graph = pd.read_parquet(GRAPH_EMBEDDINGS_FILE)
    labels = load_tourney_labels()
    tab_seed = tab[["GameID", "Delta_Seed"]].copy().rename(columns={"Delta_Seed": "Seed_Diff"})

    compact_m = load_regular_season_compact("M")
    compact_w = load_regular_season_compact("W")
    movda_m = fit_movda_params(compact_m)
    movda_w = fit_movda_params(compact_w)

    final_predictions_all = []

    for target_season in sorted(HOLDOUT_SEASONS):
        available_seasons = sorted([s for s in TRAIN_SEASONS if s < target_season])
        if len(available_seasons) < MIN_TRAIN_SEASONS: continue

        # Smart Resume: Find best params from existing CSV if available
        best_p = None
        if log_path.exists():
            existing = pd.read_csv(log_path)
            if "target_season" in existing.columns:
                season_trials = existing[existing.target_season == target_season]
                if len(season_trials) >= n_trials:
                    print(f"\n  [✓] Resuming Season {target_season}: Using existing best trials.")
                    best_row = season_trials.sort_values("log_loss").iloc[0]
                    # Map row back to params (exclude non-param columns)
                    exclude = ["timestamp", "target_season", "trial", "log_loss"]
                    best_p = {k: v for k, v in best_row.to_dict().items() if k not in exclude}

        if best_p is None:
            print(f"\nEvaluating Holdout Season: {target_season}")
            from src.ensemble.raddar_cauchy import get_model_e_oof
            print(f"  Pre-calculating Model E OOF...")
            oof_e_horizon = get_model_e_oof(seasons=available_seasons)

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
            for i in range(n_trials):
                trial = study.ask()
                try:
                    value = _nested_objective(trial, available_seasons, tab, graph, compact_m, compact_w,
                                              movda_m, movda_w, labels, tab_seed, oof_e_horizon)
                    study.tell(trial, value)
                    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "target_season": target_season, "trial": i + 1, "log_loss": value}
                    row.update(trial.params)
                    with open(log_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if f.tell() == 0: writer.writeheader()
                        writer.writerow(row)
                    if (i+1) % 10 == 0: print(f"    Trial {i+1:2d}/{n_trials} done. Best LL: {study.best_value:.4f}")
                except Exception as e:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
            best_p = study.best_params

        # Final Evaluation for target_season (Fast)
        print(f"  Finalizing {target_season} Predictions...")
        def _f_xgb(): return xgb_lib.XGBClassifier(n_estimators=int(best_p["xgb_n"]), max_depth=int(best_p["xgb_depth"]), 
                                                  learning_rate=best_p["xgb_lr"], objective="binary:logistic", random_state=42)
        def _f_lgb(): return lgb_lib.LGBMClassifier(n_estimators=int(best_p["lgb_n"]), max_depth=int(best_p["lgb_depth"]), 
                                                  learning_rate=best_p["lgb_lr"], objective="binary", random_state=42)
        def _f_elo(): return lgb_lib.LGBMClassifier(n_estimators=int(best_p["lgb_elo_n"]), max_depth=int(best_p["lgb_elo_depth"]), 
                                                  learning_rate=best_p["xgb_lr"], objective="binary", random_state=42)
        def _f_meta(): return xgb_lib.XGBClassifier(n_estimators=int(best_p["meta_n"]), max_depth=int(best_p["meta_depth"]), 
                                                   learning_rate=best_p["meta_lr"], objective="binary:logistic", random_state=42)

        elo_train = _rebuild_elo_features(compact_m, compact_w, movda_m, movda_w, labels,
                                         best_p["elo_k"], best_p["elo_k_decay"], best_p["elo_regress"], best_p["elo_lambda"],
                                         seasons=available_seasons)
        elo_holdout = _rebuild_elo_features(compact_m, compact_w, movda_m, movda_w, labels,
                                           best_p["elo_k"], best_p["elo_k_decay"], best_p["elo_regress"], best_p["elo_lambda"],
                                           seasons=[target_season])
        
        m_a, m_b, m_d = _f_xgb(), _f_lgb(), _f_elo()
        m_a.fit(tab[tab.Season.isin(available_seasons)][TABULAR_FEATURE_NAMES].values, tab[tab.Season.isin(available_seasons)]["Result"].values)
        m_b.fit(graph[graph.Season.isin(available_seasons)][GRAPH_FEATURE_NAMES].values, graph[graph.Season.isin(available_seasons)]["Result"].values)
        m_d.fit(elo_train[ELO_FEATURE_NAMES].values, elo_train["Result"].values)

        oof_a = _expanding_cv(tab, TABULAR_FEATURE_NAMES, "Result", _f_xgb, available_seasons)
        oof_b = _expanding_cv(graph, GRAPH_FEATURE_NAMES, "Result", _f_lgb, available_seasons)
        oof_d = _expanding_cv(elo_train, ELO_FEATURE_NAMES, "Result", _f_elo, available_seasons)
        from src.ensemble.raddar_cauchy import get_model_e_oof
        oof_e = get_model_e_oof(seasons=available_seasons)
        
        train_oof = labels[labels.Season.isin(available_seasons)][["Season", "GameID", "Result", "is_women"]].copy()
        train_oof = train_oof.merge(oof_a.rename(columns={"pred": "ModelA_Pred"}), on=["Season", "GameID"]).merge(oof_b.rename(columns={"pred": "ModelB_Pred"}), on=["Season", "GameID"]).merge(oof_d.rename(columns={"pred": "ModelD_Pred"}), on=["Season", "GameID"]).merge(oof_e, on=["Season", "GameID"]).merge(tab_seed, on="GameID")
        train_oof = add_meta_interactions(train_oof)
        
        meta = _f_meta()
        meta.fit(train_oof[META_FEATURES].values, train_oof["Result"].values)

        holdout_df = labels[labels.Season == target_season][["Season", "GameID", "Result", "is_women"]].copy()
        holdout_df["ModelA_Pred"] = m_a.predict_proba(tab[tab.Season == target_season][TABULAR_FEATURE_NAMES].values)[:, 1]
        holdout_df["ModelB_Pred"] = m_b.predict_proba(graph[graph.Season == target_season][GRAPH_FEATURE_NAMES].values)[:, 1]
        holdout_df["ModelD_Pred"] = m_d.predict_proba(elo_holdout[ELO_FEATURE_NAMES].values)[:, 1]
        holdout_df = holdout_df.merge(get_model_e_oof(seasons=available_seasons + [target_season])[lambda df: df.Season == target_season], on=["Season", "GameID"])
        holdout_df = holdout_df.merge(tab_seed, on="GameID")
        holdout_df = add_meta_interactions(holdout_df)
        
        X_holdout = holdout_df[META_FEATURES].values
        # Final_Pred is now the raw, stable XGBoost output
        raw_final = meta.predict_proba(X_holdout)[:, 1]
        
        # ── Seed-Aware Dynamic Clipping ──────────────────────────────────
        seeds_abs = np.abs(holdout_df["Seed_Diff"].values)
        low_clip = np.where(seeds_abs >= 10, 0.005, CLIP_LOW)
        high_clip = np.where(seeds_abs >= 10, 0.995, CLIP_HIGH)
        
        holdout_df["Final_Pred"] = np.clip(raw_final, low_clip, high_clip)
        final_predictions_all.append(holdout_df)

    if not final_predictions_all: return
    overall = pd.concat(final_predictions_all, ignore_index=True)
    results = {"overall_log_loss": float(log_loss(overall.Result, overall.Final_Pred)),
               "overall_brier": float(brier_score_loss(overall.Result, overall.Final_Pred)),
               "overall_accuracy": float(((overall.Final_Pred > 0.5).astype(int) == overall.Result).mean()),
               "seasonal": []}
    for s in sorted(overall.Season.unique()):
        m = (overall.Season == s)
        results["seasonal"].append({"season": int(s), "log_loss": float(log_loss(overall[m].Result, overall[m].Final_Pred)),
                                   "brier": float(brier_score_loss(overall[m].Result, overall[m].Final_Pred)),
                                   "accuracy": float(((overall[m].Final_Pred > 0.5).astype(int) == overall[m].Result).mean())})
    with open("ensemble_results_4model.json", "w") as f: json.dump(results, f, indent=4)
    print(f"\n  Overall Brier: {results['overall_brier']:.4f}. Saved to ensemble_results_4model.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    run_nested_cv(args.n_trials, args.fresh)
