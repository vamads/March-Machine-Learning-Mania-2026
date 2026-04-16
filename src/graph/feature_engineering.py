"""
Graph Feature Engineering — Season game graphs, PageRank, HITS & degree metrics.

Follows the spec in 03_graph_pipeline.md:
  1. Build a directed graph per season from CompactResults (Days ≤ 132).
     - Nodes = teams
     - Edges = games, directed from winner to loser
     - Edge weight = margin of victory (optionally time-decayed)
  2. Compute graph-based centrality features:
     - PageRank (global importance / "quality wins" propagation)
     - HITS hub & authority scores (hubs beat many good teams; auths are beaten by few)
     - Weighted in/out degree ratio
  3. Produce delta matchup features (A − B) for tournament games.

TEMPORAL FIREWALL: Only regular-season games (Day ≤ 132) are included.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

from src.config import (
    RAW_DIR, PROCESSED_DIR, GRAPH_EMBEDDINGS_FILE,
    REGULAR_SEASON_CUTOFF,
    FIRST_DETAILED_SEASON, LAST_HISTORICAL_SEASON, PREDICTION_SEASON,
    TRAIN_SEASONS,
)
from src.data_loader import (
    load_regular_season_compact, load_tourney_labels,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def _time_decay_weight(day_num: int, max_day: int = REGULAR_SEASON_CUTOFF,
                       decay: float = 0.005) -> float:
    """
    Exponential time-decay factor so late-season games weigh more.
    Range ≈ [e^(-decay*max_day), 1.0]  ≈  [0.52, 1.0] for default params.
    """
    return np.exp(-decay * (max_day - day_num))


def build_season_graph(compact: pd.DataFrame, season: int) -> nx.DiGraph:
    """
    Build a directed game graph for one season.

    Edge direction: winner → loser (winner "dominates" loser).
    Edge attributes:
      - margin: point differential
      - weight: margin × time_decay (captures recency + dominance)
      - loc: H/A/N
    """
    games = compact[
        (compact.Season == season) &
        (compact.DayNum <= REGULAR_SEASON_CUTOFF)
    ].copy()

    G = nx.DiGraph()

    for _, row in games.iterrows():
        w, l = int(row.WTeamID), int(row.LTeamID)
        margin = int(row.WScore - row.LScore)
        decay = _time_decay_weight(int(row.DayNum))
        loc = row.get("WLoc", "N")

        # If edge already exists (rematch), aggregate weights
        if G.has_edge(w, l):
            G[w][l]["margin"] += margin
            G[w][l]["weight"] += margin * decay
            G[w][l]["games"]  += 1
        else:
            G.add_edge(w, l, margin=margin, weight=margin * decay,
                       games=1, loc=loc)

        # Ensure both nodes exist even if a team only loses
        if w not in G:
            G.add_node(w)
        if l not in G:
            G.add_node(l)

    return G


# ═══════════════════════════════════════════════════════════════════════════
# 2.  GRAPH FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_graph_features(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute centrality features for all nodes in a season graph.

    Features per node:
      - PageRank
      - HITS hub score
      - HITS authority score
      - Weighted out-degree / weighted in-degree ratio (dominance ratio)
      - Total wins, total losses, win fraction
    """
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return pd.DataFrame()

    # PageRank (higher = stronger team topology)
    pr = nx.pagerank(G, weight="weight", alpha=0.85)

    # HITS
    try:
        hubs, auths = nx.hits(G, max_iter=500, normalized=True)
    except nx.PowerIterationFailedConvergence:
        hubs  = {n: 0.0 for n in nodes}
        auths = {n: 0.0 for n in nodes}

    # Degree metrics
    out_deg = dict(G.out_degree(weight="weight"))  # wins (weighted)
    in_deg  = dict(G.in_degree(weight="weight"))   # losses (weighted)
    wins    = dict(G.out_degree())                  # unweighted win count
    losses  = dict(G.in_degree())                   # unweighted loss count

    rows = []
    for n in nodes:
        total_games = wins.get(n, 0) + losses.get(n, 0)
        w_deg = out_deg.get(n, 0.0)
        l_deg = in_deg.get(n, 0.0)
        rows.append({
            "TeamID":       n,
            "PageRank":     pr.get(n, 0.0),
            "HITS_Hub":     hubs.get(n, 0.0),
            "HITS_Auth":    auths.get(n, 0.0),
            "Dominance":    np.log1p(w_deg) - np.log1p(l_deg),  # log-scaled ratio
            "Graph_Wins":   wins.get(n, 0),
            "Graph_Losses": losses.get(n, 0),
            "Graph_WinFrac": wins.get(n, 0) / max(total_games, 1),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  SEASON-LEVEL PROFILES
# ═══════════════════════════════════════════════════════════════════════════

def build_graph_profiles(seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Build graph-based features for all teams across seasons (M + W).
    Combines classic features (PageRank, HITS) with GNN embeddings.
    """
    from src.graph.gnn_model import compute_gnn_embeddings, GNN_EMBED_COLS

    if seasons is None:
        seasons = TRAIN_SEASONS + [PREDICTION_SEASON]

    all_profiles = []

    for prefix, is_w in [("M", 0), ("W", 1)]:
        compact = load_regular_season_compact(prefix)

        # Classic features (PageRank, HITS, Dominance)
        classic_dfs = []
        for season in tqdm(seasons, desc=f"{prefix} graph features"):
            G = build_season_graph(compact, season)
            if len(G.nodes()) == 0:
                continue
            feats = extract_graph_features(G)
            feats["Season"] = season
            feats["is_women"] = is_w
            classic_dfs.append(feats)

        if not classic_dfs:
            continue
        classic = pd.concat(classic_dfs, ignore_index=True)

        # GNN embeddings
        print(f"  Computing {prefix} GNN embeddings ({len(seasons)} seasons) …")
        gnn = compute_gnn_embeddings(compact, seasons, is_w)

        # Merge classic + GNN
        if len(gnn) > 0:
            merged = classic.merge(gnn[["Season", "TeamID"] + GNN_EMBED_COLS],
                                   on=["Season", "TeamID"], how="left")
            # Fill missing GNN embeddings with 0
            for col in GNN_EMBED_COLS:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0.0)
                else:
                    merged[col] = 0.0
        else:
            merged = classic
            for col in GNN_EMBED_COLS:
                merged[col] = 0.0

        all_profiles.append(merged)

    profiles = pd.concat(all_profiles, ignore_index=True)
    return profiles


# ═══════════════════════════════════════════════════════════════════════════
# 4.  DELTA FEATURES
# ═══════════════════════════════════════════════════════════════════════════

# Classic graph features use standard deltas
from src.graph.gnn_model import GNN_EMBED_COLS

_CLASSIC_GRAPH_COLS = [
    "PageRank", "HITS_Hub", "HITS_Auth",
    "Dominance", "Graph_WinFrac",
]

# Rotation-invariant GNN features (computed at matchup time)
_GNN_MATCHUP_COLS = ["GNN_EucDist", "GNN_CosSim"]

GRAPH_FEATURE_NAMES = (
    [f"Delta_{c}" for c in _CLASSIC_GRAPH_COLS]
    + _GNN_MATCHUP_COLS
)


def _merge_graph_to_matchups(
    matchups: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge graph profiles → matchups and compute features.

    Classic features (PageRank, HITS, etc.) use standard deltas (A - B).
    GNN embeddings use rotation-invariant metrics:
      - Euclidean distance: ||emb_A - emb_B||  (lower = more similar)
      - Cosine similarity: cos(emb_A, emb_B)   (higher = more similar)

    These bypass the cross-season alignment problem entirely since
    distances and angles are preserved under rotation/reflection.
    """
    # Merge classic features for Team A
    a_rename = {c: f"A_{c}" for c in _CLASSIC_GRAPH_COLS}
    matchups = matchups.merge(
        profiles[["Season", "TeamID"] + _CLASSIC_GRAPH_COLS].rename(
            columns={"TeamID": "TeamA", **a_rename}
        ),
        on=["Season", "TeamA"],
        how="left",
    )

    # Merge classic features for Team B
    b_rename = {c: f"B_{c}" for c in _CLASSIC_GRAPH_COLS}
    matchups = matchups.merge(
        profiles[["Season", "TeamID"] + _CLASSIC_GRAPH_COLS].rename(
            columns={"TeamID": "TeamB", **b_rename}
        ),
        on=["Season", "TeamB"],
        how="left",
    )

    # Classic deltas
    for col in _CLASSIC_GRAPH_COLS:
        matchups[f"Delta_{col}"] = matchups[f"A_{col}"] - matchups[f"B_{col}"]

    # ── GNN rotation-invariant features ──────────────────────────────
    # Merge raw GNN embeddings for distance computation
    a_gnn_rename = {c: f"A_{c}" for c in GNN_EMBED_COLS}
    matchups = matchups.merge(
        profiles[["Season", "TeamID"] + GNN_EMBED_COLS].rename(
            columns={"TeamID": "TeamA", **a_gnn_rename}
        ),
        on=["Season", "TeamA"],
        how="left",
    )

    b_gnn_rename = {c: f"B_{c}" for c in GNN_EMBED_COLS}
    matchups = matchups.merge(
        profiles[["Season", "TeamID"] + GNN_EMBED_COLS].rename(
            columns={"TeamID": "TeamB", **b_gnn_rename}
        ),
        on=["Season", "TeamB"],
        how="left",
    )

    # Compute rotation-invariant distance metrics
    a_gnn = matchups[[f"A_{c}" for c in GNN_EMBED_COLS]].values
    b_gnn = matchups[[f"B_{c}" for c in GNN_EMBED_COLS]].values

    # Euclidean distance
    diff = a_gnn - b_gnn
    matchups["GNN_EucDist"] = np.sqrt((diff ** 2).sum(axis=1))

    # Cosine similarity
    norm_a = np.sqrt((a_gnn ** 2).sum(axis=1))
    norm_b = np.sqrt((b_gnn ** 2).sum(axis=1))
    dot_product = (a_gnn * b_gnn).sum(axis=1)
    matchups["GNN_CosSim"] = dot_product / (norm_a * norm_b + 1e-10)

    # Drop raw GNN columns (not needed as features)
    drop_cols = [f"A_{c}" for c in GNN_EMBED_COLS] + [f"B_{c}" for c in GNN_EMBED_COLS]
    matchups = matchups.drop(columns=drop_cols, errors="ignore")

    return matchups


# ═══════════════════════════════════════════════════════════════════════════
# 5.  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def build_graph_features() -> pd.DataFrame:
    """
    End-to-end graph pipeline: build season graphs, extract features,
    merge with tournament labels, compute deltas, save to parquet.
    """
    print("▶ Building graph profiles …")
    profiles = build_graph_profiles()
    print(f"  {len(profiles):,} team-seasons with graph features")

    print("▶ Loading tournament labels …")
    labels = load_tourney_labels()

    print("▶ Computing graph delta features …")
    features = _merge_graph_to_matchups(labels, profiles)

    n_missing = features[GRAPH_FEATURE_NAMES].isna().any(axis=1).sum()
    if n_missing > 0:
        print(f"  ⚠ {n_missing} matchups missing graph features. Dropping.")
        features = features.dropna(subset=GRAPH_FEATURE_NAMES)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_parquet(GRAPH_EMBEDDINGS_FILE, index=False)
    print(f"  Saved {len(features):,} matchups → {GRAPH_EMBEDDINGS_FILE}")

    return features


# ── Quick Sanity Check ───────────────────────────────────────────────────

if __name__ == "__main__":
    features = build_graph_features()
    print(f"\nShape: {features.shape}")
    print(f"\nDelta feature ranges:")
    for col in GRAPH_FEATURE_NAMES:
        if col in features.columns:
            print(f"  {col:30s}  "
                  f"mean={features[col].mean():+.6f}  "
                  f"std={features[col].std():.6f}  "
                  f"[{features[col].min():+.6f}, {features[col].max():+.6f}]")
    print(f"\nResult balance: {features.Result.mean():.3f}")
