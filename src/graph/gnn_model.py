"""
GNN Model — GraphSAGE for learning team embeddings from season game graphs.

Architecture:
  - 2-layer GraphSAGE with node metadata (conference, coach tenure, win rate)
  - Edge weights from margin × time-decay
  - Trained per-season to reconstruct game outcomes (link prediction)
  - Outputs d-dimensional team embeddings

The embeddings augment the existing PageRank/HITS features in the graph pipeline.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from collections import defaultdict

from src.config import REGULAR_SEASON_CUTOFF
from src.graph.metadata import build_node_features, get_node_feature_dim


# ═══════════════════════════════════════════════════════════════════════════
# GNN ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

class TeamGraphSAGE(torch.nn.Module):
    """
    2-layer GraphSAGE for learning team embeddings from game graphs.
    
    Trained via edge-level binary classification: predict game outcomes
    from team embeddings (link prediction).
    """
    def __init__(self, in_channels: int, hidden_channels: int = 32,
                 out_channels: int = 16):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        # Edge predictor: takes concatenated embeddings of two teams
        self.edge_pred = torch.nn.Sequential(
            torch.nn.Linear(out_channels * 2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )

    def encode(self, x, edge_index):
        """Generate team embeddings."""
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        return h

    def decode(self, z, edge_index):
        """Predict game outcomes from team embeddings."""
        src, dst = edge_index
        cat = torch.cat([z[src], z[dst]], dim=-1)
        return self.edge_pred(cat).squeeze(-1)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)


# ═══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

def _time_decay_weight(day_num, max_day=REGULAR_SEASON_CUTOFF, decay=0.005):
    return np.exp(-decay * (max_day - day_num))


def build_pyg_graph(compact: pd.DataFrame, season: int,
                    node_features: dict) -> Data | None:
    """
    Build a PyG Data object from compact results.

    Edges: both directions for each game (winner→loser as positive,
    loser→winner as negative for training).
    Node features: from metadata.
    """
    games = compact[
        (compact.Season == season) &
        (compact.DayNum <= REGULAR_SEASON_CUTOFF)
    ].copy()

    if len(games) == 0:
        return None

    # Collect all teams
    all_teams = sorted(set(games.WTeamID.unique()) | set(games.LTeamID.unique()))
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)

    # Build node feature matrix
    feat_dim = get_node_feature_dim()
    x = np.zeros((n_teams, feat_dim), dtype=np.float32)
    for team_id, idx in team_to_idx.items():
        if team_id in node_features:
            x[idx] = node_features[team_id]
        else:
            # Default features for teams without metadata
            x[idx, -1] = 0.5  # default win rate

    # Build edges: winner→loser (label=1 for winner perspective)
    edge_src, edge_dst, edge_labels = [], [], []

    for _, row in games.iterrows():
        w_idx = team_to_idx[row.WTeamID]
        l_idx = team_to_idx[row.LTeamID]

        # Winner → Loser (positive edge)
        edge_src.append(w_idx)
        edge_dst.append(l_idx)
        edge_labels.append(1.0)

        # Loser → Winner (negative edge, for training balance)
        edge_src.append(l_idx)
        edge_dst.append(w_idx)
        edge_labels.append(0.0)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)

    # For message passing, use ALL edges (undirected connectivity)
    # but train on directional labels
    mp_src = edge_src + edge_dst
    mp_dst = edge_dst + edge_src
    mp_edge_index = torch.tensor([mp_src, mp_dst], dtype=torch.long)

    data = Data(
        x=torch.tensor(x),
        edge_index=mp_edge_index,   # for message passing
        train_edge_index=edge_index, # for loss computation
        edge_labels=edge_labels,
        team_ids=all_teams,
        team_to_idx=team_to_idx,
    )

    return data


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_gnn_season(data: Data, hidden_dim: int = 32, embed_dim: int = 16,
                     epochs: int = 100, lr: float = 0.01,
                     prev_state_dict: dict | None = None) -> tuple[dict, dict]:
    """
    Train a GraphSAGE on one season's game graph and extract embeddings.

    Parameters
    ----------
    prev_state_dict : optional state dict from previous season's model.
        If provided, the model is warm-started from those weights for
        temporal coherence instead of random initialization.

    Returns
    -------
    tuple: (embeddings dict, model state_dict)
        embeddings: {team_id: np.array of shape (embed_dim,)}
        state_dict: for warm-starting the next season
    """
    if data is None:
        return {}, None

    in_dim = data.x.shape[1]
    model = TeamGraphSAGE(in_dim, hidden_dim, embed_dim)

    # Warm-start from previous season if available
    if prev_state_dict is not None:
        try:
            model.load_state_dict(prev_state_dict)
        except RuntimeError:
            pass  # Architecture mismatch — start fresh

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        pred = model.decode(z, data.train_edge_index)
        loss = F.binary_cross_entropy_with_logits(pred, data.edge_labels)
        loss.backward()
        optimizer.step()

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        embeddings = z.numpy()

    # Map back to team IDs
    result = {}
    for team_id, idx in data.team_to_idx.items():
        result[team_id] = embeddings[idx]

    return result, model.state_dict()


def compute_gnn_embeddings(compact: pd.DataFrame, seasons: list[int],
                           is_women: int, embed_dim: int = 16) -> pd.DataFrame:
    """
    Compute GNN embeddings for all teams across multiple seasons,
    with warm-started training for temporal coherence.

    Each season's GNN is initialized from the previous season's final
    weights rather than random initialization. This provides immediate
    temporal coherence — the embedding space naturally evolves smoothly
    rather than undergoing random rotations/reflections.

    The feature engineering layer then computes rotation-invariant
    metrics (Euclidean distance, cosine similarity) at matchup time.

    Returns DataFrame with columns: Season, TeamID, GNN_0, ..., GNN_{embed_dim-1}
    """
    all_rows = []
    prev_state = None  # Previous season's model weights

    for season in sorted(seasons):
        # Build node features
        node_features = build_node_features(season,
            list(set(compact[compact.Season == season].WTeamID.unique()) |
                 set(compact[compact.Season == season].LTeamID.unique())),
            compact)

        # Build PyG graph
        data = build_pyg_graph(compact, season, node_features)
        if data is None:
            continue

        # Train with warm-start and extract embeddings
        embeddings, state_dict = train_gnn_season(
            data, embed_dim=embed_dim, prev_state_dict=prev_state
        )

        # Save state for next season's warm-start
        if state_dict is not None:
            prev_state = state_dict

        for team_id, emb in embeddings.items():
            row = {"Season": season, "TeamID": int(team_id), "is_women": is_women}
            for d in range(embed_dim):
                row[f"GNN_{d}"] = float(emb[d])
            all_rows.append(row)

    return pd.DataFrame(all_rows)


GNN_EMBED_DIM = 16
GNN_EMBED_COLS = [f"GNN_{d}" for d in range(GNN_EMBED_DIM)]
