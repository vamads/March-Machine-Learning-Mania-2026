"""
Graph Metadata — Load and encode team metadata for GNN node features.

Builds per-team-per-season feature vectors from:
  - Conference membership (one-hot encoded, top conferences)
  - Coach tenure at current team (years)
  - Historical win rate from compact results
"""

import pandas as pd
import numpy as np
from collections import defaultdict

from src.config import (
    RAW_DIR, REGULAR_SEASON_CUTOFF,
)
from src.data_loader import load_regular_season_compact

# Top conferences to one-hot encode (rest get lumped into "other")
TOP_CONFERENCES = [
    "acc", "big_east", "big_ten", "big_twelve", "sec",
    "pac_ten", "pac_twelve", "aac", "mwc", "wcc",
    "a_ten", "mvc", "colonial", "mtn_west", "be",
]


def _load_conferences() -> pd.DataFrame:
    """Load team conference memberships for M + W."""
    dfs = []
    for prefix in ["M", "W"]:
        path = RAW_DIR / f"{prefix}TeamConferences.csv"
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=["Season", "TeamID", "ConfAbbrev"])
    return pd.concat(dfs, ignore_index=True)


def _load_coaches() -> pd.DataFrame:
    """Load coaching data (Men's only — no Women's coaching data)."""
    path = RAW_DIR / "MTeamCoaches.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Season", "TeamID", "CoachName",
                                      "FirstDayNum", "LastDayNum"])
    return pd.read_csv(path)


def _compute_coach_tenure(coaches: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Season, TeamID), compute how many consecutive seasons
    the current coach has been coaching that team.
    """
    # Get the coach active at end of regular season for each team-season
    active = coaches[coaches.LastDayNum >= REGULAR_SEASON_CUTOFF].copy()
    # If multiple coaches, take the one with latest FirstDayNum
    active = (active.sort_values("FirstDayNum", ascending=False)
              .drop_duplicates(["Season", "TeamID"], keep="first"))

    # Compute tenure
    tenure_rows = []
    team_groups = active.sort_values("Season").groupby("TeamID")

    for team_id, group in team_groups:
        prev_coach = None
        streak = 0
        for _, row in group.iterrows():
            if row.CoachName == prev_coach:
                streak += 1
            else:
                streak = 1
                prev_coach = row.CoachName
            tenure_rows.append({
                "Season": row.Season,
                "TeamID": team_id,
                "CoachTenure": streak,
            })

    return pd.DataFrame(tenure_rows)


def build_node_features(season: int, team_ids: list[int],
                        compact: pd.DataFrame) -> dict:
    """
    Build a feature vector for each team in a season graph.

    Returns
    -------
    dict: {team_id: np.array of features}
    Features: [conf_onehot (len=len(TOP_CONFERENCES)+1), coach_tenure, win_rate]
    """
    # Conference
    conferences = _load_conferences()
    conf_map = {}
    season_conf = conferences[conferences.Season == season]
    for _, row in season_conf.iterrows():
        conf_map[row.TeamID] = row.ConfAbbrev

    # Coach tenure
    coaches = _load_coaches()
    tenure_df = _compute_coach_tenure(coaches)
    tenure_map = {}
    season_tenure = tenure_df[tenure_df.Season == season]
    for _, row in season_tenure.iterrows():
        tenure_map[row.TeamID] = row.CoachTenure

    # Win rate from compact results this season
    games = compact[
        (compact.Season == season) &
        (compact.DayNum <= REGULAR_SEASON_CUTOFF)
    ]
    wins = defaultdict(int)
    total = defaultdict(int)
    for _, row in games.iterrows():
        wins[row.WTeamID] += 1
        total[row.WTeamID] += 1
        total[row.LTeamID] += 1

    n_conf = len(TOP_CONFERENCES)
    features = {}

    for tid in team_ids:
        # Conference one-hot
        conf = conf_map.get(tid, "other")
        conf_vec = np.zeros(n_conf + 1, dtype=np.float32)
        if conf in TOP_CONFERENCES:
            conf_vec[TOP_CONFERENCES.index(conf)] = 1.0
        else:
            conf_vec[n_conf] = 1.0  # "other"

        # Coach tenure (normalized)
        tenure = tenure_map.get(tid, 1) / 10.0

        # Win rate
        t = total.get(tid, 0)
        wr = wins.get(tid, 0) / max(t, 1)

        features[tid] = np.concatenate([conf_vec, [tenure, wr]])

    return features


def get_node_feature_dim() -> int:
    """Return the dimension of node feature vectors."""
    return len(TOP_CONFERENCES) + 1 + 2  # conf_onehot + tenure + win_rate
