"""
Master Pipeline Runner — Single entrypoint for the March Madness ensemble.

Usage:
    conda activate madness
    python run_pipeline.py                     # Run everything
    python run_pipeline.py --stage features    # Only build features
    python run_pipeline.py --stage models      # Only train models
    python run_pipeline.py --stage submit      # Only generate submission
"""

import argparse
import time


def stage_features():
    """Build all feature pipelines."""
    print("\n" + "=" * 60)
    print("STAGE 1: FEATURE ENGINEERING")
    print("=" * 60)

    from src.data_loader import build_tourney_labels, save_tourney_labels
    labels = build_tourney_labels()
    save_tourney_labels(labels)

    from src.tabular.feature_engineering import build_tabular_features
    build_tabular_features()

    from src.graph.feature_engineering import build_graph_features
    build_graph_features()

    from src.elo.feature_engineering import build_elo_features
    build_elo_features()


def stage_models():
    """Train base models + meta-learner."""
    print("\n" + "=" * 60)
    print("STAGE 2: MODEL TRAINING")
    print("=" * 60)

    from src.ensemble.base_models import train_all_base_models, train_final_models
    train_all_base_models()
    train_final_models(save=True)

    from src.ensemble.meta_learner import train_meta_learner
    import pandas as pd
    from src.config import PROCESSED_DIR
    oof = pd.read_parquet(PROCESSED_DIR / "oof_predictions.parquet")
    train_meta_learner(oof)


def stage_submit():
    """Generate submission CSVs."""
    print("\n" + "=" * 60)
    print("STAGE 3: SUBMISSION GENERATION")
    print("=" * 60)

    from src.submit import generate_submission
    generate_submission(stage=2)


def main():
    parser = argparse.ArgumentParser(description="March Madness 2026 Pipeline")
    parser.add_argument("--stage", choices=["features", "models", "submit", "all"],
                        default="all", help="Which stage to run")
    args = parser.parse_args()

    t0 = time.time()

    if args.stage in ("features", "all"):
        stage_features()
    if args.stage in ("models", "all"):
        stage_models()
    if args.stage in ("submit", "all"):
        stage_submit()

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
