"""
Evaluate Unified Model vs Ensemble

This script runs a true temporal holdout evaluation on both the existing
Late Fusion Ensemble and the new Early Fusion Unified Model to compare
their performance side-by-side.
"""

import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
import numpy as np

from src.config import TRAIN_ONLY_SEASONS, HOLDOUT_SEASONS
from src.ensemble.unified_model import load_unified_data, predict_holdout_unified, _unified_lgb_factory
from src.holdout_evaluate import run_holdout_evaluation

def evaluate_both():
    print("\n" + "=" * 60)
    print("  EVALUATING LATE FUSION ENSEMBLE (Baseline)")
    print("=" * 60)
    
    holdout_ensemble = run_holdout_evaluation()
    
    print("\n" + "=" * 60)
    print("  EVALUATING EARLY FUSION UNIFIED MODEL")
    print("=" * 60)
    
    print("\n── Loading Unified Data ──")
    unified_df, unified_features = load_unified_data()
    print(f"  Merged Data: {len(unified_df):,} rows")
    print(f"  Feature Count: {len(unified_features)}")
    
    print("\n── Predicting Holdout Seasons (2022-2025) ──")
    pred_unified, _ = predict_holdout_unified(
        df=unified_df,
        feature_cols=unified_features,
        target_col="Result",
        model_factory=_unified_lgb_factory,
        model_name="UnifiedLGB",
        train_seasons=TRAIN_ONLY_SEASONS,
        holdout_seasons=HOLDOUT_SEASONS
    )
    
    print("\n" + "=" * 60)
    print("  UNIFIED MODEL HOLDOUT RESULTS")
    print("=" * 60)
    
    overall_ll = log_loss(pred_unified.Result, pred_unified.UnifiedLGB_Pred)
    overall_bs = brier_score_loss(pred_unified.Result, pred_unified.UnifiedLGB_Pred)
    pred_unified["Correct"] = ((pred_unified.UnifiedLGB_Pred > 0.5).astype(int) == pred_unified.Result).astype(int)
    overall_acc = pred_unified.Correct.mean()

    print(f"\n  Overall Log Loss:   {overall_ll:.4f}")
    print(f"  Overall Brier:      {overall_bs:.4f}")
    print(f"  Overall Accuracy:   {overall_acc:.1%} "
          f"({pred_unified.Correct.sum()}/{len(pred_unified)})")
          
    # Compare against ensemble!
    ensemble_ll = log_loss(holdout_ensemble.Result, holdout_ensemble.Pred)
    ensemble_bs = brier_score_loss(holdout_ensemble.Result, holdout_ensemble.Pred)
    ensemble_acc = holdout_ensemble.Correct.mean()
    
    print("\n" + "=" * 60)
    print("  HEAD-TO-HEAD COMPARISON: ENSEMBLE vs UNIFIED")
    print("=" * 60)
    
    print(f"\n  Ensemble Log Loss: {ensemble_ll:.4f}")
    print(f"  Unified Log Loss:  {overall_ll:.4f}")
    diff_ll = overall_ll - ensemble_ll
    print(f"  Difference:        {diff_ll:+.4f} ({'Unified Better' if diff_ll < 0 else 'Ensemble Better'})")
    
    print(f"\n  Ensemble Brier:    {ensemble_bs:.4f}")
    print(f"  Unified Brier:     {overall_bs:.4f}")
    diff_bs = overall_bs - ensemble_bs
    print(f"  Difference:        {diff_bs:+.4f} ({'Unified Better' if diff_bs < 0 else 'Ensemble Better'})")
    
    print(f"\n  Ensemble Accuracy: {ensemble_acc:.1%}")
    print(f"  Unified Accuracy:  {overall_acc:.1%}")
    diff_acc = overall_acc - ensemble_acc
    print(f"  Difference:        {diff_acc:+.1%} ({'Unified Better' if diff_acc > 0 else 'Ensemble Better'})")


if __name__ == "__main__":
    evaluate_both()
