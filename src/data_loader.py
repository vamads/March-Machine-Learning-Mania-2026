"""
Data Loader — Load raw CSVs, build canonical matchup IDs, and
construct the tournament target table.

Key responsibilities:
  1. Load Men's and Women's data from RAW_DIR.
  2. Tag every row with `is_women` (0 or 1).
  3. Build a unified tournament labels DataFrame for training:
     one row per tournament game with canonical (TeamA < TeamB) ordering.
  4. Parse sample submissions into (Season, TeamA, TeamB) for inference.
"""

import pandas as pd
import numpy as np
from src.config import (
    RAW_DIR, PROCESSED_DIR,
    FIRST_DETAILED_SEASON, LAST_HISTORICAL_SEASON, PREDICTION_SEASON,
    TOURNEY_LABELS_FILE,
)


# ── Raw CSV Loading ──────────────────────────────────────────────────────

def load_csv(filename: str) -> pd.DataFrame:
    """Load a single CSV from RAW_DIR."""
    return pd.read_csv(RAW_DIR / filename)


def load_regular_season_detailed(prefix: str = "M") -> pd.DataFrame:
    """Load detailed regular-season results (box scores)."""
    return load_csv(f"{prefix}RegularSeasonDetailedResults.csv")


def load_regular_season_compact(prefix: str = "M") -> pd.DataFrame:
    """Load compact regular-season results (scores only)."""
    return load_csv(f"{prefix}RegularSeasonCompactResults.csv")


def load_tourney_compact(prefix: str = "M") -> pd.DataFrame:
    """Load compact NCAA tournament results."""
    return load_csv(f"{prefix}NCAATourneyCompactResults.csv")


def load_tourney_seeds(prefix: str = "M") -> pd.DataFrame:
    """Load tournament seeds."""
    return load_csv(f"{prefix}NCAATourneySeeds.csv")


def load_massey_ordinals() -> pd.DataFrame:
    """Load the (large) Massey Ordinals file.  Men's only — no Women's ordinals exist."""
    return load_csv("MMasseyOrdinals.csv")


def load_teams(prefix: str = "M") -> pd.DataFrame:
    """Load team metadata."""
    return load_csv(f"{prefix}Teams.csv")


def load_sample_submission(stage: int = 2) -> pd.DataFrame:
    """Load a sample submission file (stage 1 or 2)."""
    return load_csv(f"SampleSubmissionStage{stage}.csv")


from typing import List, Optional

# ── Canonical Game ID ────────────────────────────────────────────────────

def make_game_id(season: int, team_a: int, team_b: int) -> str:
    """
    Create the canonical Kaggle matchup ID.
    Convention: lower TeamID always comes first → '2024_1104_1242'.
    """
    lo, hi = sorted([team_a, team_b])
    return f"{season}_{lo}_{hi}"


# ── Tournament Labels ────────────────────────────────────────────────────

def _build_labels_from_compact(compact_df: pd.DataFrame,
                               is_women: int) -> pd.DataFrame:
    """
    Convert a compact-results DataFrame into canonical labelled matchups.

    For each game the winner is WTeamID and the loser is LTeamID.
    We create the canonical ordering (TeamA < TeamB) and set
        Result = 1  if TeamA won
        Result = 0  if TeamA lost (i.e. TeamB won)
    """
    rows = []
    for _, r in compact_df.iterrows():
        season = int(r["Season"])
        w, l = int(r["WTeamID"]), int(r["LTeamID"])
        lo, hi = sorted([w, l])
        
        # Row 1: Canonical (lower ID first)
        result_lo = 1 if w == lo else 0
        rows.append({
            "Season":   season,
            "TeamA":    lo,
            "TeamB":    hi,
            "GameID":   f"{season}_{lo}_{hi}",
            "Result":   result_lo,
            "is_women": is_women,
        })
        
        # Row 2: Swapped (higher ID first)
        rows.append({
            "Season":   season,
            "TeamA":    hi,
            "TeamB":    lo,
            "GameID":   f"{season}_{hi}_{lo}",
            "Result":   1 - result_lo,
            "is_women": is_women,
        })
    return pd.DataFrame(rows)


def build_tourney_labels(seasons: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Build the master tournament-labels table for Men's + Women's
    NCAA tournament games.

    Parameters
    ----------
    seasons : list of int, optional
        If given, restrict to these seasons.  Defaults to
        FIRST_DETAILED_SEASON .. LAST_HISTORICAL_SEASON.

    Returns
    -------
    pd.DataFrame with columns:
        Season, TeamA, TeamB, GameID, Result, is_women
    """
    if seasons is None:
        seasons = list(range(FIRST_DETAILED_SEASON,
                             LAST_HISTORICAL_SEASON + 1))

    dfs = []
    for prefix, is_w in [("M", 0), ("W", 1)]:
        compact = load_tourney_compact(prefix)
        compact = compact[compact["Season"].isin(seasons)]
        dfs.append(_build_labels_from_compact(compact, is_w))

    labels = pd.concat(dfs, ignore_index=True)
    labels = labels.sort_values(["Season", "GameID"]).reset_index(drop=True)
    return labels


def save_tourney_labels(labels: pd.DataFrame) -> None:
    """Persist the tournament labels to parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(TOURNEY_LABELS_FILE, index=False)
    print(f"  Saved {len(labels):,} tournament labels → {TOURNEY_LABELS_FILE}")


def load_tourney_labels() -> pd.DataFrame:
    """Load previously-saved tournament labels."""
    return pd.read_parquet(TOURNEY_LABELS_FILE)


# ── Submission Parsing ───────────────────────────────────────────────────

def parse_submission_ids(stage: int = 2) -> pd.DataFrame:
    """
    Parse a sample submission CSV into structured matchup rows.

    Returns
    -------
    pd.DataFrame with columns: Season, TeamA, TeamB, GameID
    """
    sub = load_sample_submission(stage)
    parts = sub["ID"].str.split("_", expand=True).astype(int)
    parsed = pd.DataFrame({
        "Season": parts[0],
        "TeamA":  parts[1],
        "TeamB":  parts[2],
        "GameID": sub["ID"],
    })
    return parsed


# ── Quick Sanity Check ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building tournament labels …")
    labels = build_tourney_labels()
    print(f"  Total games: {len(labels):,}")
    print(f"  Men's games: {(labels.is_women == 0).sum():,}")
    print(f"  Women's games: {(labels.is_women == 1).sum():,}")
    print(f"  Seasons: {labels.Season.min()} – {labels.Season.max()}")
    print(f"  Result balance: {labels.Result.mean():.3f} "
          f"(should be ~0.50)")
    save_tourney_labels(labels)

    print("\nParsing Stage 2 submission IDs …")
    sub2 = parse_submission_ids(stage=2)
    print(f"  {len(sub2):,} matchups to predict for {PREDICTION_SEASON}")
    print(sub2.head())
