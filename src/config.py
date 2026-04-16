"""
Central configuration for the March Madness 2026 pipeline.

All paths, constants, and hyperparameters live here so every module
imports from a single source of truth.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# ── Temporal Firewall ────────────────────────────────────────────────────
# Regular-season features must use ONLY days <= REGULAR_SEASON_CUTOFF.
# Tournament targets begin on TOURNEY_START_DAY.
REGULAR_SEASON_CUTOFF = 132   # Last day of regular season data we may use
TOURNEY_START_DAY     = 134   # First day of NCAA Tournament games

# ── Season Ranges ────────────────────────────────────────────────────────
# Detailed box-score data starts in 2003; compact results go back to 1985.
FIRST_DETAILED_SEASON = 2003
LAST_HISTORICAL_SEASON = 2025   # Most recent season with known tournament results
PREDICTION_SEASON      = 2026

# All seasons we train on (detailed results required for tabular pipeline)
TRAIN_SEASONS = list(range(FIRST_DETAILED_SEASON, LAST_HISTORICAL_SEASON + 1))

# Holdout seasons for true temporal evaluation (simulate unseen tournaments)
HOLDOUT_SEASONS = [2022, 2023, 2024, 2025]
TRAIN_ONLY_SEASONS = [s for s in TRAIN_SEASONS if s not in HOLDOUT_SEASONS]

# Expanding Window CV: minimum seasons of training data before predicting
MIN_TRAIN_SEASONS = 5

# ── Elo Rating System (MOVDA) ────────────────────────────────────────────
ELO_INIT     = 1500    # Starting rating for all teams
ELO_K        = 44.4    # Base K-factor (Optuna-tuned, decays over season)
ELO_K_DECAY  = 0.77    # Exponential decay rate for K over season
ELO_REGRESS  = 0.37    # Season-to-season regression toward mean
ELO_LAMBDA   = 0.45    # Weight for MOV differential term
# MOVDA α, β, δ (defaults; overwritten by curve_fit at runtime)
ELO_MOVDA_ALPHA = 12.0  # Asymptotic max margin from skill
ELO_MOVDA_BETA  = 0.003 # Steepness near Δelo=0
ELO_MOVDA_DELTA = 3.5   # Home-court advantage in points
FIRST_COMPACT_SEASON_M = 1985  # Men's compact data starts here
FIRST_COMPACT_SEASON_W = 1998  # Women's compact data starts here



# ── Prediction Clipping ─────────────────────────────────────────────────
# Guard against catastrophic Log Loss from extreme probabilities.
CLIP_LOW  = 0.025
CLIP_HIGH = 0.975

# ── File Names (processed outputs) ───────────────────────────────────────
TOURNEY_LABELS_FILE     = PROCESSED_DIR / "tourney_labels.parquet"
TABULAR_FEATURES_FILE   = PROCESSED_DIR / "tabular_features.parquet"
GRAPH_EMBEDDINGS_FILE   = PROCESSED_DIR / "graph_embeddings.parquet"
ELO_FEATURES_FILE       = PROCESSED_DIR / "elo_features.parquet"
ORDINALS_FEATURES_FILE  = PROCESSED_DIR / "ordinals_features.parquet"

# ── Massey Ordinals (Men's only) ─────────────────────────────────────────
MASSEY_SYSTEMS       = ["POM", "SAG", "MOR", "DOL", "COL"]
MASSEY_ANCHOR_DAY    = 133   # Selection Sunday snapshot
MASSEY_BENCHMARK_DAY = 103   # 30 days prior (for trend)

# ── Men's / Women's CSV Prefixes ─────────────────────────────────────────
# Men's files start with "M", Women's with "W".  We process both and tag
# each row with an `is_women` flag so the meta-learner can learn any
# structural differences between the two tournaments.
MEN_PREFIX   = "M"
WOMEN_PREFIX = "W"
