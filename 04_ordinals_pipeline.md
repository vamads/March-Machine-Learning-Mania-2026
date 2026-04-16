# Ordinals Pipeline (Massey Rankings & Momentum)
This pipeline processes the `MasseyOrdinals.csv` file to extract predictive, forward-looking ranking systems and calculate late-season momentum ("the eye test" and "hot streaks").

**1. System Filtering & Selection:**
- Filter the massive dataset to include only the most predictive systems: `POM` (KenPom), `SAG` (Sagarin), `BPI` (ESPN), `TOR` (Torvik), and `MOR` (Moore). 
- Drop all other ranking systems to prevent multicollinearity and noise.

**2. Time-Series Feature Engineering (The Momentum Features):**
Instead of just taking a single snapshot, calculate how a team's ranking has changed over the final stretch of the season. 
- **The Anchor (Day 133):** Extract the final pre-tournament ranking for each team on Day 133 (Selection Sunday). 
- **The Benchmark (Day 103):** Extract the team's ranking from 30 days prior (Day 103). 
- **The Trend:** Calculate the slope/difference: `Trend = Rank_Day_133 - Rank_Day_103`. 
  - *Note for the model:* A negative trend value is actually positive momentum (e.g., moving from rank 25 to rank 10 means a trend of -15).

**3. Consensus Metric Generation:**
To smooth out individual system biases, calculate the **median** rank across the selected systems (POM, SAG, BPI, TOR, MOR) for both the Day 133 Anchor and the Trend feature.
- Generate `Consensus_Rank_Day_133`
- Generate `Consensus_Trend`

**4. Delta Matchup Features:**
Transform the data into symmetrical matchup differences so the Level 1 meta-learner can easily process the gaps between two opponents (e.g., an MSU vs. UConn matchup).
- `Delta_Consensus_Rank = Team_A_Consensus_Day_133 - Team_B_Consensus_Day_133`
- `Delta_Consensus_Trend = Team_A_Consensus_Trend - Team_B_Consensus_Trend`

**5. STRICT LEAKAGE GUARDRAIL:**
Under no circumstances can this pipeline process or merge Massey Ordinal data from Day 134 or beyond for the season being predicted. The tournament starts after Day 133.