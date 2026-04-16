"""
4-Model Holdout Evaluation — Robust Stacking with Interaction Features.

Usage:
    conda run -n madness python -m src.holdout_evaluate_4model
"""

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss

from src.config import (
    TABULAR_FEATURES_FILE, GRAPH_EMBEDDINGS_FILE, ELO_FEATURES_FILE,
    PROCESSED_DIR, HOLDOUT_SEASONS, TRAIN_ONLY_SEASONS,
    CLIP_LOW, CLIP_HIGH, FIRST_DETAILED_SEASON, LAST_HISTORICAL_SEASON,
)
from src.tabular.feature_engineering import TABULAR_FEATURE_NAMES
from src.graph.feature_engineering import GRAPH_FEATURE_NAMES
from src.elo.feature_engineering import ELO_FEATURE_NAMES
from src.data_loader import load_tourney_labels
from src.ensemble.base_models import _xgb_factory, _lgb_factory
from src.ensemble.meta_learner import (
    META_FEATURES, add_meta_interactions, merge_hca_sensitivity, 
    train_meta_learner, predict_ensemble
)
from src.ensemble.raddar_cauchy import build_raddar_features, _compute_laplace_team_features, RADDAR_FEATURES, SIGMOID_K
def report_calibration(y_true, y_prob, n_bins=10):
    """
    Compute reliability diagram (binning) and Expected Calibration Error (ECE).
    """
    from sklearn.calibration import calibration_curve
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    # Calculate ECE manually
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)
    
    ece = 0.0
    print("\n  Reliability Diagram (Calibration Check):")
    print(f"    {'Bin Range':15s} {'Pred Prob':10s} {'Actual Rate':12s} {'Samples':8s} {'Error':8s}")
    print("    " + "-" * 58)
    
    for i in range(len(prob_true)):
        mask = binids == i
        if mask.sum() == 0: continue
        
        bin_obs = prob_true[i]
        bin_pred = prob_pred[i]
        bin_size = mask.sum()
        weight = bin_size / len(y_true)
        error = bin_pred - bin_obs
        ece += weight * np.abs(error)
        
        lower = bins[i]
        upper = bins[i+1]
        print(f"    {lower:4.2f}-{upper:4.2f}    {bin_pred:10.3f}    {bin_obs:12.3f}    {bin_size:8d}    {error:+7.3f}")
        
    print("    " + "-" * 58)
    print(f"    Expected Calibration Error (ECE): {ece:.4f}")
    
    if ece < 0.02:
        print("    [STATUS] Excellent calibration!")
    elif ece < 0.05:
        print("    [STATUS] Good calibration.")
    else:
        print("    [WARNING] Model may be miscalibrated.")


def _expanding_cv(df, feature_cols, target_col, model_factory, model_name, train_seasons):
    from src.config import MIN_TRAIN_SEASONS
    df_train = df[df.Season.isin(train_seasons)].copy()
    seasons = sorted(df_train.Season.unique())
    oof_rows = []

    for test_season in seasons:
        past = [s for s in seasons if s < test_season]
        if len(past) < MIN_TRAIN_SEASONS:
            continue
        train = df_train[df_train.Season.isin(past)]
        test = df_train[df_train.Season == test_season]

        model = model_factory()
        model.fit(train[feature_cols].values, train[target_col].values)
        preds = model.predict_proba(test[feature_cols].values)[:, 1]

        for gid, season, pred in zip(test["GameID"], test["Season"], preds):
            oof_rows.append({"Season": season, "GameID": gid, f"{model_name}_Pred": float(pred)})

    return pd.DataFrame(oof_rows)


def _predict_holdout(df, feature_cols, target_col, model_factory, model_name,
                     train_seasons, holdout_seasons):
    train = df[df.Season.isin(train_seasons)]
    holdout = df[df.Season.isin(holdout_seasons)]
    if len(holdout) == 0:
        return pd.DataFrame()

    model = model_factory()
    model.fit(train[feature_cols].values, train[target_col].values)
    preds = model.predict_proba(holdout[feature_cols].values)[:, 1]

    rows = []
    for gid, season, pred in zip(holdout["GameID"], holdout["Season"], preds):
        rows.append({"Season": season, "GameID": gid, f"{model_name}_Pred": float(pred)})
    return pd.DataFrame(rows)


def _make_cauchy_xgb():
    import xgboost as xgb_lib
    from src.ensemble.raddar_cauchy import cauchy_objective
    return xgb_lib.XGBRegressor(
        n_estimators=40, max_depth=4, learning_rate=0.05,
        objective=cauchy_objective, random_state=42, n_jobs=-1,
    )


def _raddar_cauchy_oof(matchups, train_seasons):
    from src.ensemble.raddar_cauchy import _compute_prior_matchup, RADDAR_FEATURES, SIGMOID_K
    from sklearn.preprocessing import OneHotEncoder
    from scipy.sparse import hstack, csr_matrix
    from sklearn.linear_model import LogisticRegression
    
    df_train = matchups[matchups.Season.isin(train_seasons)].copy()
    oof_leaves, oof_sigmoid, oof_y, oof_gids, oof_seasons = [], [], [], [], []

    for val_season in train_seasons:
        past = [s for s in train_seasons if s < val_season]
        if len(past) < 3:
            continue

        prior = _compute_prior_matchup(val_season)
        tr = df_train[df_train.Season.isin(past)].copy()
        va = df_train[df_train.Season == val_season].copy()

        # Merge prior matchup
        for d in [tr, va]:
            d2 = d.merge(prior, on=["TeamA", "TeamB"], how="left", suffixes=("", "_new"))
            if "Laplace_Prior_Matchup_new" in d2.columns:
                d2["Laplace_Prior_Matchup"] = d2["Laplace_Prior_Matchup_new"].fillna(d2["Laplace_Prior_Matchup"])
                d2 = d2.drop(columns=["Laplace_Prior_Matchup_new"])
            d2["Laplace_Prior_Matchup"] = d2["Laplace_Prior_Matchup"].fillna(0.5)
            if d is tr: tr = d2
            else: va = d2

        if len(tr) == 0 or len(va) == 0: continue

        model = _make_cauchy_xgb()
        model.fit(tr[RADDAR_FEATURES].values, tr["PointDiff"].values)

        margin = model.predict(va[RADDAR_FEATURES].values)
        prob = 1.0 / (1.0 + np.exp(-margin * SIGMOID_K))
        leaves = model.apply(va[RADDAR_FEATURES].values)

        oof_leaves.append(leaves)
        oof_sigmoid.extend(prob.tolist())
        oof_y.extend(va["Result"].values.tolist())
        oof_gids.extend(va["GameID"].values.tolist())
        oof_seasons.extend(va["Season"].values.tolist())

    oof_leaves_arr = np.vstack(oof_leaves)
    oof_sigmoid_arr = np.array(oof_sigmoid).reshape(-1, 1)
    oof_y_arr = np.array(oof_y)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_leaves = encoder.fit_transform(oof_leaves_arr)
    X_train = hstack([csr_matrix(oof_sigmoid_arr), X_leaves])

    meta_e = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, max_iter=1000, random_state=42)
    meta_e.fit(X_train, oof_y_arr)
    oof_preds = meta_e.predict_proba(X_train)[:, 1]

    return pd.DataFrame({"Season": oof_seasons, "GameID": oof_gids, "ModelE_Pred": oof_preds}), encoder, meta_e


def _raddar_cauchy_holdout(matchups, train_seasons, holdout_seasons, encoder, meta_e):
    from src.ensemble.raddar_cauchy import _compute_prior_matchup, RADDAR_FEATURES, SIGMOID_K
    from scipy.sparse import hstack, csr_matrix
    
    df_train = matchups[matchups.Season.isin(train_seasons)].copy()
    holdout = matchups[matchups.Season.isin(holdout_seasons)].copy()

    prior = _compute_prior_matchup(max(holdout_seasons) + 1)
    for d in [df_train, holdout]:
        d2 = d.merge(prior, on=["TeamA", "TeamB"], how="left", suffixes=("", "_new"))
        if "Laplace_Prior_Matchup_new" in d2.columns:
            d2["Laplace_Prior_Matchup"] = d2["Laplace_Prior_Matchup_new"].fillna(d2["Laplace_Prior_Matchup"])
            d2 = d2.drop(columns=["Laplace_Prior_Matchup_new"])
        d2["Laplace_Prior_Matchup"] = d2["Laplace_Prior_Matchup"].fillna(0.5)
        if d is df_train: df_train = d2
        else: holdout = d2

    model = _make_cauchy_xgb()
    model.fit(df_train[RADDAR_FEATURES].values, df_train["PointDiff"].values)

    h_margin = model.predict(holdout[RADDAR_FEATURES].values)
    h_prob = 1.0 / (1.0 + np.exp(-h_margin * SIGMOID_K))
    h_sparse = encoder.transform(model.apply(holdout[RADDAR_FEATURES].values))
    X_holdout = hstack([csr_matrix(h_prob.reshape(-1, 1)), h_sparse])

    return pd.DataFrame({"Season": holdout["Season"].values, "GameID": holdout["GameID"].values, "ModelE_Pred": meta_e.predict_proba(X_holdout)[:, 1]})

import warnings
warnings.filterwarnings("ignore")

def run():
    print("=" * 60)
    print("  4-MODEL HOLDOUT EVALUATION (Platt Calibrated + HCA)")
    print(f"  Train: {TRAIN_ONLY_SEASONS[0]}–{TRAIN_ONLY_SEASONS[-1]}")
    print(f"  Test:  {HOLDOUT_SEASONS}")
    print("=" * 60)

    # ── Load data ──
    tab = pd.read_parquet(TABULAR_FEATURES_FILE)
    graph = pd.read_parquet(GRAPH_EMBEDDINGS_FILE)
    elo = pd.read_parquet(ELO_FEATURES_FILE)
    labels = load_tourney_labels()
    hca_profiles = pd.read_parquet(PROCESSED_DIR / "hca_sensitivity.parquet")

    # ── Model E Features ──
    all_seasons = list(range(FIRST_DETAILED_SEASON, LAST_HISTORICAL_SEASON + 1))
    raddar_matchups = build_raddar_features(all_seasons)
    laplace_team = _compute_laplace_team_features(all_seasons)
    raddar_matchups = raddar_matchups.merge(
        laplace_team.rename(columns={"TeamID": "TeamA", "Laplace_AW": "A_Laplace_AW", "Laplace_L14": "A_Laplace_L14"}),
        on=["Season", "TeamA"], how="left"
    )
    raddar_matchups = raddar_matchups.merge(
        laplace_team.rename(columns={"TeamID": "TeamB", "Laplace_AW": "B_Laplace_AW", "Laplace_L14": "B_Laplace_L14"}),
        on=["Season", "TeamB"], how="left"
    )
    for c in ["A_Laplace_AW", "A_Laplace_L14", "B_Laplace_AW", "B_Laplace_L14"]:
        raddar_matchups[c] = raddar_matchups[c].fillna(0.5)
    raddar_matchups["Delta_Laplace_AW"] = raddar_matchups["A_Laplace_AW"] - raddar_matchups["B_Laplace_AW"]
    raddar_matchups["Delta_Laplace_L14"] = raddar_matchups["A_Laplace_L14"] - raddar_matchups["B_Laplace_L14"]
    raddar_matchups["Laplace_Prior_Matchup"] = 0.5
    raddar_matchups = raddar_matchups.dropna(subset=RADDAR_FEATURES).reset_index(drop=True)

    # ── Step 1 & 2: OOF ──
    oof_a = _expanding_cv(tab, TABULAR_FEATURE_NAMES, "Result", _xgb_factory, "ModelA", TRAIN_ONLY_SEASONS)
    oof_b = _expanding_cv(graph, GRAPH_FEATURE_NAMES, "Result", _lgb_factory, "ModelB", TRAIN_ONLY_SEASONS)
    oof_d = _expanding_cv(elo, ELO_FEATURE_NAMES, "Result", _lgb_factory, "ModelD", TRAIN_ONLY_SEASONS)
    oof_e, encoder_e, meta_e_model = _raddar_cauchy_oof(raddar_matchups, TRAIN_ONLY_SEASONS)

    # ── Combined OOF ──
    train_labels = labels[labels.Season.isin(TRAIN_ONLY_SEASONS)].copy()
    oof = train_labels.merge(oof_a, on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_b, on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_d, on=["Season", "GameID"], how="left")
    oof = oof.merge(oof_e, on=["Season", "GameID"], how="left")
    tab_seed = tab[["GameID", "Delta_Seed"]].rename(columns={"Delta_Seed": "Seed_Diff"})
    oof = oof.merge(tab_seed, on="GameID", how="left")
    oof = merge_hca_sensitivity(oof, hca_profiles)
    oof = add_meta_interactions(oof)
    
    usable = oof[oof[["ModelA_Pred", "ModelB_Pred"]].notna().any(axis=1)].copy()
    usable["ModelE_Pred"] = usable["ModelE_Pred"].fillna(0.5)

    # ── Step 3: Train ──
    print("\n── Step 3: Train Unified Meta-Learner + Platt Calibrator ──")
    meta_payload = train_meta_learner(usable, dropout_rate=0.20)
    
    # ── Step 4: Predict Holdout ──
    pred_a = _predict_holdout(tab, TABULAR_FEATURE_NAMES, "Result", _xgb_factory, "ModelA", TRAIN_ONLY_SEASONS, HOLDOUT_SEASONS)
    pred_b = _predict_holdout(graph, GRAPH_FEATURE_NAMES, "Result", _lgb_factory, "ModelB", TRAIN_ONLY_SEASONS, HOLDOUT_SEASONS)
    pred_d = _predict_holdout(elo, ELO_FEATURE_NAMES, "Result", _lgb_factory, "ModelD", TRAIN_ONLY_SEASONS, HOLDOUT_SEASONS)
    pred_e = _raddar_cauchy_holdout(raddar_matchups, TRAIN_ONLY_SEASONS, HOLDOUT_SEASONS, encoder_e, meta_e_model)

    holdout = labels[labels.Season.isin(HOLDOUT_SEASONS)].copy()
    holdout = holdout.merge(pred_a, on=["Season", "GameID"], how="left")
    holdout = holdout.merge(pred_b, on=["Season", "GameID"], how="left")
    holdout = holdout.merge(pred_d, on=["Season", "GameID"], how="left")
    holdout = holdout.merge(pred_e, on=["Season", "GameID"], how="left")
    holdout = holdout.merge(tab_seed, on="GameID", how="left")
    holdout = merge_hca_sensitivity(holdout, hca_profiles)
    holdout = add_meta_interactions(holdout)

    preds = predict_ensemble(
        meta_payload,
        holdout["ModelA_Pred"].values,
        holdout["ModelB_Pred"].values,
        holdout["ModelD_Pred"].values,
        holdout["ModelE_Pred"].values,
        holdout["Seed_Diff"].values,
        holdout["is_women"].values,
        holdout["Delta_HCA_Sensitivity"].values
    )
    
    y_true = holdout["Result"].values
    report_calibration(y_true, preds)

    ll = log_loss(y_true, preds)
    bs = brier_score_loss(y_true, preds)
    acc = ((preds > 0.5).astype(int) == y_true).mean()

    print("\n" + "=" * 60)
    print(f"  FINAL HOLDOUT: LL={ll:.4f}  Brier={bs:.4f}  Acc={acc:.1%}")
    print("=" * 60)

    print("\n  Per-Season Breakdown:")
    for season in sorted(HOLDOUT_SEASONS):
        mask = holdout.Season.values == season
        if mask.sum() == 0: continue
        s_ll = log_loss(y_true[mask], preds[mask])
        s_bs = brier_score_loss(y_true[mask], preds[mask])
        s_acc = ((preds[mask] > 0.5).astype(int) == y_true[mask]).mean()
        print(f"    {season}: LL={s_ll:.4f}  Brier={s_bs:.4f}  Acc={s_acc:.1%}  (n={mask.sum()})")
    print("=" * 60)

if __name__ == "__main__":
    run()
