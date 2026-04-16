"""
Seed Parser — Extract numeric seeds from the tournament seed strings.

Seed strings look like 'W01', 'X16a', 'Z11b', etc.
  - First character = region (W, X, Y, Z)
  - Next two digits = seed number (01–16)
  - Optional trailing letter (a/b) for play-in games
"""

import pandas as pd
from src.config import RAW_DIR, FIRST_DETAILED_SEASON, PREDICTION_SEASON


def parse_seed_number(seed_str: str) -> int:
    """Extract the numeric seed (1–16) from a seed string like 'W01' or 'X16a'."""
    return int(seed_str[1:3])


def build_seed_lookup() -> pd.DataFrame:
    """
    Build a (Season, TeamID) → SeedNum lookup table for both Men's and Women's.

    Returns
    -------
    pd.DataFrame with columns: Season, TeamID, SeedNum, is_women
    """
    dfs = []
    for prefix, is_w in [("M", 0), ("W", 1)]:
        seeds = pd.read_csv(RAW_DIR / f"{prefix}NCAATourneySeeds.csv")
        seeds["SeedNum"] = seeds["Seed"].apply(parse_seed_number)
        seeds["is_women"] = is_w
        seeds = seeds[["Season", "TeamID", "SeedNum", "is_women"]]
        dfs.append(seeds)

    return pd.concat(dfs, ignore_index=True)


def get_seed(seed_lookup: pd.DataFrame,
             season: int, team_id: int) -> int | None:
    """Look up a single team's seed.  Returns None if the team wasn't seeded."""
    row = seed_lookup[
        (seed_lookup.Season == season) & (seed_lookup.TeamID == team_id)
    ]
    if len(row) == 0:
        return None
    return int(row.iloc[0]["SeedNum"])


# ── Quick Sanity Check ───────────────────────────────────────────────────

if __name__ == "__main__":
    seeds = build_seed_lookup()
    print(f"Total seed entries: {len(seeds):,}")
    print(f"Men's:   {(seeds.is_women == 0).sum():,}")
    print(f"Women's: {(seeds.is_women == 1).sum():,}")
    print(f"Seed range: {seeds.SeedNum.min()} – {seeds.SeedNum.max()}")
    print(f"Seasons: {seeds.Season.min()} – {seeds.Season.max()}")
    assert seeds.SeedNum.between(1, 16).all(), "Seed numbers out of range!"
    print("✓ All seeds in [1, 16]")
    print(seeds.head(10))
