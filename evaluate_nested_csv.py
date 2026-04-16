import pandas as pd
import numpy as np
import xgboost as xgb_lib
import lightgbm as lgb_lib
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import json

from src.config import (
    MODELS_DIR, TABULAR_FEATURES_FILE, GRAPH_EMBEDDINGS_FILE,
    HOLDOUT_SEASONS, TRAIN_SEASONS, CLIP_LOW, CLIP_HIGH
)
from src.tabular.feature_engineering import TABULAR_FEATURE_NAMES
from src.graph.feature_engineering import GRAPH_FEATURE_NAMES
from src.elo.feature_engineering import ELO_FEATURE_NAMES
from src.ensemble.meta_learner import META_FEATURES, add_meta_interactions
from src.data_loader import load_regular_season_compact, load_tourney_labels
from src.ensemble.tune import _expanding_cv, _rebuild_elo_features
from src.elo.feature_engineering import fit_movda_params

print("Loading data...")
tab = pd.read_parquet(TABULAR_FEATURES_FILE)
graph = pd.read_parquet(GRAPH_EMBEDDINGS_FILE)
labels = load_tourney_labels()
tab_seed = tab[["GameID", "Delta_Seed"]].copy().rename(columns={"Delta_Seed": "Seed_Diff"})

compact_m = load_regular_season_compact("M")
compact_w = load_regular_season_compact("W")
movda_m = fit_movda_params(compact_m)
movda_w = fit_movda_params(compact_w)

# Load CSV
df = pd.read_csv(MODELS_DIR / "nested_cv_trials.csv")

final_predictions_all = []

for target_season in sorted(HOLDOUT_SEASONS):
    print(f"\nEvaluating target season: {target_season}")
    subset = df[df.target_season == target_season]
    best_row = subset.loc[subset.log_loss.idxmin()]
    best_p = best_row.to_dict()
    print(f"  Best Past LL: {best_p['log_loss']:.4f}")
    
    available_seasons = sorted([s for s in TRAIN_SEASONS if s < target_season])
    
    def _f_xgb(): return xgb_lib.XGBClassifier(
        n_estimators=int(best_p["xgb_n"]), max_depth=int(best_p["xgb_depth"]), learning_rate=best_p["xgb_lr"],
        objective="binary:logistic", use_label_encoder=False, verbosity=0, random_state=42
    )
    def _f_lgb(): return lgb_lib.LGBMClassifier(
        n_estimators=int(best_p["lgb_n"]), max_depth=int(best_p["lgb_depth"]), learning_rate=best_p["lgb_lr"],
        objective="binary", verbosity=-1, random_state=42
    )
    def _f_elo(): return lgb_lib.LGBMClassifier(
        n_estimators=int(best_p["lgb_elo_n"]), max_depth=int(best_p["lgb_elo_depth"]), learning_rate=best_p["lgb_lr"],
        objective="binary", verbosity=-1, random_state=42
    )
    def _f_meta(): return xgb_lib.XGBClassifier(
        n_estimators=int(best_p["meta_n"]), max_depth=int(best_p["meta_depth"]), learning_rate=best_p["meta_lr"],
        objective="binary:logistic", use_label_encoder=False, verbosity=0, random_state=42
    )

    elo_train = _rebuild_elo_features(
        compact_m, compact_w, movda_m, movda_w, labels,
        best_p["elo_k"], best_p["elo_k_decay"], best_p["elo_regress"], best_p["elo_lambda"],
        seasons=available_seasons
    )
    elo_holdout = _rebuild_elo_features(
        compact_m, compact_w, movda_m, movda_w, labels,
        best_p["elo_k"], best_p["elo_k_decay"], best_p["elo_regress"], best_p["elo_lambda"],
        seasons=[target_season]
    )

    train_tab = tab[tab.Season.isin(available_seasons)]
    train_grf = graph[graph.Season.isin(available_seasons)]
    
    m_xgb, m_lgb, m_elo = _f_xgb(), _f_lgb(), _f_elo()
    m_xgb.fit(train_tab[TABULAR_FEATURE_NAMES].values, train_tab["Result"].values)
    m_lgb.fit(train_grf[GRAPH_FEATURE_NAMES].values, train_grf["Result"].values)
    m_elo.fit(elo_train[ELO_FEATURE_NAMES].values, elo_train["Result"].values)
    
    oof_a = _expanding_cv(tab, TABULAR_FEATURE_NAMES, "Result", _f_xgb, available_seasons)
    oof_b = _expanding_cv(graph, GRAPH_FEATURE_NAMES, "Result", _f_lgb, available_seasons)
    oof_d = _expanding_cv(elo_train, ELO_FEATURE_NAMES, "Result", _f_elo, available_seasons)
    
    oof = labels[labels.Season.isin(available_seasons)][["Season", "GameID", "Result", "is_women"]].copy()
    oof = oof.merge(oof_a.rename(columns={"pred": "ModelA_Pred"}), on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_b.rename(columns={"pred": "ModelB_Pred"}), on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_d.rename(columns={"pred": "ModelD_Pred"}), on=["Season", "GameID"], how="left")
    oof = oof.merge(tab_seed, on="GameID", how="left")
    oof = oof.dropna(subset=["ModelA_Pred", "ModelB_Pred", "ModelD_Pred"])
    
    oof = add_meta_interactions(oof)
    meta = _f_meta()
    meta.fit(oof[META_FEATURES].values, oof["Result"].values)
    
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.clip(meta.predict_proba(oof[META_FEATURES].values)[:, 1], CLIP_LOW, CLIP_HIGH), oof["Result"].values)
    
    holdout_tab = tab[tab.Season == target_season]
    holdout_grf = graph[graph.Season == target_season]
    
    h_df = labels[labels.Season == target_season][["Season", "GameID", "Result", "is_women"]].copy()
    h_df["ModelA_Pred"] = m_xgb.predict_proba(holdout_tab[TABULAR_FEATURE_NAMES].values)[:, 1]
    h_df["ModelB_Pred"] = m_lgb.predict_proba(holdout_grf[GRAPH_FEATURE_NAMES].values)[:, 1]
    h_df["ModelD_Pred"] = m_elo.predict_proba(elo_holdout[ELO_FEATURE_NAMES].values)[:, 1]
    h_df = h_df.merge(tab_seed, on="GameID", how="left")
    
    h_df = add_meta_interactions(h_df)
    raw_meta = meta.predict_proba(h_df[META_FEATURES].values)[:, 1]
    h_df["Final_Pred"] = np.clip(calibrator.predict(raw_meta), CLIP_LOW, CLIP_HIGH)
    final_predictions_all.append(h_df)

overall = pd.concat(final_predictions_all, ignore_index=True)
overall_ll = log_loss(overall.Result, overall.Final_Pred)
overall_bs = brier_score_loss(overall.Result, overall.Final_Pred)
acc = ((overall.Final_Pred > 0.5).astype(int) == overall.Result).mean()

print("\n" + "=" * 60)
print("  NESTED CV RESULTS (UNBIASED HOLDOUT)")
print("=" * 60)
print(f"  Overall Log Loss:   {overall_ll:.4f}")
print(f"  Overall Brier:      {overall_bs:.4f}")
print(f"  Overall Accuracy:   {acc:.1%}")
