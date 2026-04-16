"""
Injury Profiles — 2026 Tournament.

Maps manually reported late-season injuries to TeamIDs and calculates
an 'Injury_Risk' score to be used for ensemble hedging.
"""

import pandas as pd
from src.config import PROCESSED_DIR

def build_injury_profiles():
    print("▶ Building 2026 Injury Profiles …")
    
    # Men's Injuries (M)
    mens_data = [
        {"TeamID": 1104, "Team": "Alabama", "Severity": 0.4}, # Aden Holloway (Q), Bristow (Q), Hannah (Q)
        {"TeamID": 1116, "Team": "Arkansas", "Severity": 0.3}, # Karter Knox (Q)
        {"TeamID": 1140, "Team": "BYU", "Severity": 1.0},      # Saunders (Out), Baker (Out) - BIG IMPACT
        {"TeamID": 1155, "Team": "Clemson", "Severity": 0.5},  # Carter Welling (Out)
        {"TeamID": 1163, "Team": "UConn", "Severity": 0.3},    # Demary Jr. (Q)
        {"TeamID": 1181, "Team": "Duke", "Severity": 0.8},     # Caleb Foster (Out), Ngongba (Q)
        {"TeamID": 1211, "Team": "Gonzaga", "Severity": 0.3},  # Braden Huff (Q)
        {"TeamID": 1246, "Team": "Kentucky", "Severity": 0.3}, # Quaintance (Q)
        {"TeamID": 1257, "Team": "Louisville", "Severity": 0.5}, # Mikel Brown Jr. (Out)
        {"TeamID": 1276, "Team": "Michigan", "Severity": 0.6}, # L.J. Cason (Out)
        {"TeamID": 1314, "Team": "North Carolina", "Severity": 0.9}, # Caleb Wilson (Out), James Brown (Out)
        {"TeamID": 1326, "Team": "Ohio State", "Severity": 0.3}, # Chatman (Q)
        {"TeamID": 1400, "Team": "Texas", "Severity": 0.3},    # Traore (Q)
        {"TeamID": 1403, "Team": "Texas Tech", "Severity": 0.6}, # Toppin (Out), others playing
        {"TeamID": 1417, "Team": "UCLA", "Severity": 0.1},     # Expected to play
        {"TeamID": 1437, "Team": "Villanova", "Severity": 0.7}, # Matt Hodge (Out)
        {"TeamID": 1458, "Team": "Wisconsin", "Severity": 0.3}, # Nolan Winter (Q)
    ]
    
    # Women's Injuries (W)
    womens_data = [
        {"TeamID": 3161, "Team": "Colorado State", "Severity": 0.8}, # Bargesser (Out)
        {"TeamID": 3393, "Team": "Syracuse", "Severity": 0.6},       # Darius (Out)
        {"TeamID": 3163, "Team": "UConn", "Severity": 1.0},          # Brady, Cheli, El Alfy (All Out) - EXTREME
        {"TeamID": 3376, "Team": "South Carolina", "Severity": 1.0},   # Watkins, Kitts (All Out) - EXTREME
        {"TeamID": 3235, "Team": "Iowa State", "Severity": 0.7},      # Addy Brown (Out)
        {"TeamID": 3425, "Team": "USC", "Severity": 0.8},             # JuJu Watkins (Out/Rehab) - EXTREME
    ]
    
    df_m = pd.DataFrame(mens_data)
    df_w = pd.DataFrame(womens_data)
    
    df = pd.concat([df_m, df_w], ignore_index=True)
    df["Season"] = 2026
    
    out_file = PROCESSED_DIR / "injury_risk.parquet"
    df.to_parquet(out_file, index=False)
    print(f"  Saved {len(df)} injury profiles → {out_file}")
    
    return df

if __name__ == "__main__":
    build_injury_profiles()
