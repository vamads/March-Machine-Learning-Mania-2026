"""
HCA Sensitivity Analysis — Identifying "Home Bullies".

Computes (Home WinPct) - (Away/Neutral WinPct) for every team-season
to identify teams that are disproportionately reliant on home-court advantage.
"""

import pandas as pd
import numpy as np
from src.config import PROCESSED_DIR, REGULAR_SEASON_CUTOFF
from src.data_loader import load_regular_season_compact

def build_hca_profiles():
    print("▶ Building HCA sensitivity profiles …")
    
    all_profiles = []
    for (prefix, gender) in [("M", "Men"), ("W", "Women")]:
        compact = load_regular_season_compact(prefix)
        # Limit to regular season
        compact = compact[compact.DayNum <= REGULAR_SEASON_CUTOFF].copy()
        
        # Team Win Stats by Location
        # WTeamID wins at WLoc
        # LTeamID loses at WLoc (inverted)
        
        wins = compact[["Season", "WTeamID", "WLoc"]].rename(columns={"WTeamID": "TeamID", "WLoc": "Loc"})
        wins["Win"] = 1
        
        losses = compact[["Season", "LTeamID", "WLoc"]].rename(columns={"LTeamID": "TeamID", "WLoc": "Loc"})
        # Invert location for loss
        losses["Loc"] = losses["Loc"].map({"H": "A", "A": "H", "N": "N"})
        losses["Win"] = 0
        
        games = pd.concat([wins, losses], ignore_index=True)
        
        # Group by Season, TeamID, Loc
        stats = games.groupby(["Season", "TeamID", "Loc"]).agg(
            GP = ("Win", "count"),
            Wins = ("Win", "sum")
        ).reset_index()
        
        # Pivot to get Home vs Away/Neutral
        piv = stats.pivot_table(index=["Season", "TeamID"], columns="Loc", values=["GP", "Wins"], fill_value=0)
        piv.columns = [f"{col[1]}_{col[0]}" for col in piv.columns]
        piv = piv.reset_index()
        
        # Combine Away + Neutral
        piv["AwayNeutral_GP"] = piv.get("A_GP", 0) + piv.get("N_GP", 0)
        piv["AwayNeutral_Wins"] = piv.get("A_Wins", 0) + piv.get("N_Wins", 0)
        
        piv["Home_WinPct"] = np.where(piv.H_GP > 0, piv.H_Wins / piv.H_H_GP if "H_H_GP" in piv.columns else piv.H_Wins / piv.H_GP, 0)
        # Fix column names from pivot if needed (GP_H, GP_A, GP_N etc)
        # Let's be safer with columns
        
    # Re-doing pivot logic for clarity
    all_hca = []
    for prefix in ["M", "W"]:
        compact = load_regular_season_compact(prefix)
        compact = compact[compact.DayNum <= REGULAR_SEASON_CUTOFF].copy()
        
        for (season, team_id), grp in compact.groupby(["Season", "WTeamID"]):
            pass # Too slow to iterate
            
    # Vectorized approach
    def get_team_stats(df, prefix):
        compact = load_regular_season_compact(prefix)
        compact = compact[compact.DayNum <= REGULAR_SEASON_CUTOFF].copy()
        
        # W perspective
        w = compact[['Season', 'WTeamID', 'WLoc']].rename(columns={'WTeamID':'TeamID', 'WLoc':'Loc'})
        w['Win'] = 1
        # L perspective
        l = compact[['Season', 'LTeamID', 'WLoc']].rename(columns={'LTeamID':'TeamID', 'WLoc':'Loc'})
        l['Loc'] = l['Loc'].map({'H':'A', 'A':'H', 'N':'N'})
        l['Win'] = 0
        
        comb = pd.concat([w, l])
        
        # Home stats
        home = comb[comb.Loc == 'H'].groupby(['Season', 'TeamID'])['Win'].agg(['count', 'sum']).rename(columns={'count':'H_GP', 'sum':'H_Wins'})
        # Away/Neutral stats
        an = comb[comb.Loc != 'H'].groupby(['Season', 'TeamID'])['Win'].agg(['count', 'sum']).rename(columns={'count':'AN_GP', 'sum':'AN_Wins'})
        
        merged = home.join(an, how='outer').fillna(0).reset_index()
        merged['Home_WinPct'] = merged['H_Wins'] / merged['H_GP'].replace(0, 1)
        merged['AN_WinPct'] = merged['AN_Wins'] / merged['AN_GP'].replace(0, 1)
        merged['HCA_Sensitivity'] = merged['Home_WinPct'] - merged['AN_WinPct']
        
        return merged

    m_hca = get_team_stats(None, "M")
    w_hca = get_team_stats(None, "W")
    
    all_hca = pd.concat([m_hca, w_hca], ignore_index=True)
    
    out_file = PROCESSED_DIR / "hca_sensitivity.parquet"
    all_hca.to_parquet(out_file, index=False)
    print(f"  Saved {len(all_hca):,} team-seasons → {out_file}")
    
    # Show Top 10 Bullies overall
    bullies = all_hca[all_hca.H_GP >= 5].sort_values("HCA_Sensitivity", ascending=False).head(10)
    print("\n  Top 10 Home Bullies (Home WinPct - Road WinPct):")
    for _, row in bullies.iterrows():
        print(f"    Season {int(row.Season)} Team {int(row.TeamID)}: Delta={row.HCA_Sensitivity:.3f} (H:{row.Home_WinPct:.2f}, R:{row.AN_WinPct:.2f})")
    
    return all_hca

if __name__ == "__main__":
    build_hca_profiles()
