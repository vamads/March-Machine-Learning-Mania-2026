# Graph Pipeline (Strength of Schedule & Transitivity)
This model uses Graph Neural Networks (GNN) to capture how teams perform against common opponents and across different conferences.

**Graph Structure:**
- **Nodes:** Teams (e.g., Michigan State, UConn). Node features should include basic program identifiers and their season-average tabular stats.
- **Edges:** Directed edges representing a game played between two nodes. 

**Edge Feature Engineering:**
An edge should not just be a binary 1/0 for a game played. It needs context:
- **Weight/Score:** Margin of victory, or better yet, the possession-adjusted offensive efficiency margin for that specific game. 
- **Location Encoding:** A categorical feature on the edge indicating if the game was Home, Away, or Neutral. A Michigan State win on the road carries a different topological weight than a win at home.
- **Time Decay:** Games played later in the season (closer to Day 132) should have a slightly higher edge weight than games played on Day 1.

**Output:** The GNN (e.g., GraphSAGE or GCN) should output a fixed-length embedding vector for each node. We will calculate the cosine similarity or distance between Team A's embedding and Team B's embedding to feed into the final ensemble.