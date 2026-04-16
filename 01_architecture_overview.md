# Project Overview: March Mania Ensemble
We are building a stacking ensemble to predict the outcome of the 2026 Kaggle March Machine Learning Mania tournament. 

**The Base Models (Level 0):**
1. **Tabular Model (XGBoost/LightGBM):** Learns team efficiency, possession-based metrics, and variance/upset potential from detailed regular season box scores.
2. **Graph Model (GNN):** Learns transitive properties and strength of schedule by modeling teams as nodes and games as edges.
3. **Ordinals Model (Logistic Regression/Ridge):** Captures human-voted "eye test" and intangible momentum using top Massey Ordinals.

**Data Splitting Strategy (The Most Critical Rule):**
To ensure the model learns how to predict tournament environments rather than regular-season environments, we must split the data strictly across the Day 132 boundary:
- **Training Features (X):** Regular season stats, graph embeddings, and ordinals (Days 1–132) for historical years (2003 through 2025).
- **Training Targets (y):** The historical **NCAA Tournament games** (Days 134+) for those exact same years.
- **Testing/Inference Set:** The upcoming **2026 Tournament matchups**, utilizing only the 2026 regular-season features.

**Output & Evaluation (Log Loss Optimization):**
Kaggle evaluates this competition using **Log Loss**. We are predicting *probabilities*, not binary winners. 
- The Level 1 Meta-Learner must output a float between 0.0 and 1.0 representing the probability that Team A beats Team B.

$$\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]$$

- **Handling Upsets:** Log Loss heavily penalizes overconfidence. If a chaotic 15-seed plays a highly structured 2-seed, the model should recognize high-variance features and assign the underdog a realistic probability (e.g., 18%). If the upset occurs, the model is heavily rewarded for capturing that vulnerability. Models that confidently predict a 1% chance for the underdog will suffer massive Log Loss penalties.