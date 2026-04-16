"""
Experiment Tracker — Log metrics from each pipeline run to compare
across versions and code changes.

Appends a row to `submissions/experiment_log.csv` each time you run it.
Each row captures: timestamp, experiment name, notes, and all key metrics.

Usage:
    # After generating a Stage 1 submission:
    conda activate madness
    python -m src.track --name "baseline_v1" --notes "Initial 3-model stack"

    # After making changes:
    python -m src.track --name "added_momentum" --notes "Added 30-day rank trend"

    # View history:
    python -m src.track --history
"""

import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path
from sklearn.metrics import log_loss, brier_score_loss

from src.config import SUBMISSIONS_DIR
from src.evaluate import load_actual_results


EXPERIMENT_LOG = SUBMISSIONS_DIR / "experiment_log.csv"

STAGE1_SEASONS = [2022, 2023, 2024, 2025]


# ═══════════════════════════════════════════════════════════════════════════
# COLLECT METRICS
# ═══════════════════════════════════════════════════════════════════════════

def collect_metrics(stage: int = 1) -> dict:
    """
    Score the current Stage 1 submission and return a flat dict of metrics.
    """
    sub = pd.read_csv(SUBMISSIONS_DIR / f"submission_stage{stage}.csv")
    actuals = load_actual_results(STAGE1_SEASONS)
    merged = actuals.merge(sub, left_on="GameID", right_on="ID", how="inner")

    metrics = {}

    # Overall
    metrics["log_loss"] = round(log_loss(merged.Result, merged.Pred), 4)
    metrics["brier"] = round(brier_score_loss(merged.Result, merged.Pred), 4)

    # Accuracy
    merged["Correct"] = ((merged.Pred > 0.5).astype(int) == merged.Result).astype(int)
    metrics["accuracy"] = round(merged.Correct.mean(), 4)
    metrics["n_games"] = len(merged)
    metrics["n_correct"] = int(merged.Correct.sum())
    metrics["n_wrong"] = int(len(merged) - merged.Correct.sum())

    # Per season
    for s in STAGE1_SEASONS:
        m = merged[merged.Season == s]
        if len(m) > 0:
            metrics[f"ll_{s}"] = round(log_loss(m.Result, m.Pred), 4)
            metrics[f"acc_{s}"] = round(m.Correct.mean(), 4)

    # By gender
    for label, is_w in [("men", 0), ("women", 1)]:
        m = merged[merged.is_women == is_w]
        if len(m) > 0:
            metrics[f"ll_{label}"] = round(log_loss(m.Result, m.Pred), 4)
            metrics[f"acc_{label}"] = round(m.Correct.mean(), 4)

    # Confidence buckets
    merged["Confidence"] = abs(merged.Pred - 0.5)
    for lo, hi, label in [(0, 0.1, "tossup"), (0.1, 0.2, "lean"),
                           (0.2, 0.3, "likely"), (0.3, 0.5, "strong")]:
        m = merged[(merged.Confidence >= lo) & (merged.Confidence < hi)]
        if len(m) > 0:
            metrics[f"acc_{label}"] = round(m.Correct.mean(), 4)
            metrics[f"n_{label}"] = int(len(m))

    # Confident but wrong
    metrics["n_confident_wrong"] = int(
        ((merged.Correct == 0) & (merged.Confidence > 0.2)).sum()
    )

    # Prediction spread
    metrics["pred_mean"] = round(merged.Pred.mean(), 4)
    metrics["pred_std"] = round(merged.Pred.std(), 4)
    metrics["pred_min"] = round(merged.Pred.min(), 4)
    metrics["pred_max"] = round(merged.Pred.max(), 4)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# LOG EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def log_experiment(name: str, notes: str = "", stage: int = 1) -> pd.DataFrame:
    """
    Collect metrics and append a row to the experiment log.
    """
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Collecting metrics for experiment: '{name}' …")
    metrics = collect_metrics(stage)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": name,
        "notes": notes,
        **metrics,
    }

    # Append to CSV
    row_df = pd.DataFrame([row])
    if EXPERIMENT_LOG.exists():
        existing = pd.read_csv(EXPERIMENT_LOG)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(EXPERIMENT_LOG, index=False)

    # Print summary
    print(f"\n{'='*55}")
    print(f"  EXPERIMENT: {name}")
    print(f"  {row['timestamp']}")
    print(f"  Notes: {notes}")
    print(f"{'='*55}")
    print(f"  Log Loss:  {metrics['log_loss']}")
    print(f"  Brier:     {metrics['brier']}")
    print(f"  Accuracy:  {metrics['accuracy']:.1%} "
          f"({metrics['n_correct']}/{metrics['n_games']})")
    print(f"  Confident Wrong: {metrics['n_confident_wrong']}")
    print(f"{'='*55}")
    print(f"  Saved → {EXPERIMENT_LOG}")

    return combined


# ═══════════════════════════════════════════════════════════════════════════
# VIEW HISTORY
# ═══════════════════════════════════════════════════════════════════════════

def show_history():
    """Print a comparison table of all logged experiments."""
    if not EXPERIMENT_LOG.exists():
        print("No experiments logged yet. Run with --name to log one.")
        return

    df = pd.read_csv(EXPERIMENT_LOG)

    # Key columns to compare
    key_cols = ["experiment", "log_loss", "brier", "accuracy",
                "n_confident_wrong", "acc_men", "acc_women",
                "ll_2022", "ll_2023", "ll_2024", "ll_2025"]
    display_cols = [c for c in key_cols if c in df.columns]

    print(f"\n{'='*80}")
    print(f"  EXPERIMENT HISTORY ({len(df)} runs)")
    print(f"{'='*80}\n")

    # Format as table
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df[display_cols].to_string(index=False))

    # Highlight best
    if len(df) > 1:
        best_ll = df.loc[df.log_loss.idxmin()]
        best_acc = df.loc[df.accuracy.idxmax()]
        print(f"\n  🏆 Best Log Loss:  {best_ll.experiment} ({best_ll.log_loss:.4f})")
        print(f"  🏆 Best Accuracy:  {best_acc.experiment} ({best_acc.accuracy:.1%})")

    print()


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Tracker")
    parser.add_argument("--name", type=str, help="Experiment name (e.g., 'baseline_v1')")
    parser.add_argument("--notes", type=str, default="", help="Short description of changes")
    parser.add_argument("--history", action="store_true", help="Show all past experiments")
    parser.add_argument("--stage", type=int, default=1, help="Submission stage to evaluate")
    args = parser.parse_args()

    if args.history:
        show_history()
    elif args.name:
        log_experiment(args.name, args.notes, args.stage)
    else:
        parser.print_help()
