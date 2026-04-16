# Tabular Pipeline (Detailed Box Scores & Variance)
This model utilizes the `RegularSeasonDetailedResults.csv` (2003-present) to build mathematical team profiles.

**Data Preparation Requirements:**

**1. Symmetry:** Every game must be represented twice in the training set (Once where Team 1 is Team A, and once where Team 1 is Team B) to perfectly balance the target variable (1 for win, 0 for loss).

**2. Aggregation (The Baseline Averages):** Aggregate regular season games into season-long averages for each team up to Day 132. Calculate the "Four Factors" for each team's season average:
- Effective Field Goal Percentage (eFG%)
- Turnover Percentage (TOV%)
- Offensive Rebound Percentage (ORB%)
- Free Throw Rate (FTR)

**3. Upset & Variance Features (The "Madness" Metrics):**
To capture a team's unpredictability and upset potential, calculate these specific metrics alongside the standard season averages:
- **Game-to-Game Variance:** The standard deviation of a team's game-by-game offensive efficiency. High variance indicates a high-ceiling/low-floor team.
- **Live-by-the-3 Metric (3PAr):** `Total 3-Point Attempts / Total Field Goal Attempts`. Captures the mathematical variance introduced by heavy three-point reliance.
- **Chaos Creation (Defensive TOV%):** The percentage of opponent possessions that end in a turnover. Captures an underdog's ability to disrupt structured offenses.

**4. Delta Features:** Tree-based models struggle to calculate direct comparisons across columns. The model must be trained on the *differences* between the teams. 
- E.g., `Delta_eFG = Team_A_eFG - Team_B_eFG`
- E.g., `Delta_Variance = Team_A_Variance - Team_B_Variance`