"""
Predict Stage 1 & 1.5 Pure Pre-Draft Translation for 2023, 2024, and 2025 WR Classes.
Applies frozen pre-draft college baseline and athletic modulation with zero draft capital.
Outputs results to data/processed/stage1_recent_prospects.parquet and stage1_recent_prospects.csv.

Usage:
    python3 src/predict_recent_classes.py
    # or from inside src/:
    python3 predict_recent_classes.py
"""

import math
import os
from pathlib import Path

# Dual-engine support: uses Polars if installed, falls back to Pandas
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    import pandas as pd
    HAS_POLARS = False

SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "data").exists():
    DATA_DIR = SCRIPT_DIR / "data"
elif (SCRIPT_DIR.parent / "data").exists():
    DATA_DIR = SCRIPT_DIR.parent / "data"
elif Path("data").exists():
    DATA_DIR = Path("data").resolve()
else:
    DATA_DIR = (SCRIPT_DIR.parent / "data").resolve()

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT_PARQUET = PROCESSED_DIR / "stage1_recent_prospects.parquet"
OUT_CSV = PROCESSED_DIR / "stage1_recent_prospects.csv"

def calculate_hass(weight: float, forty: float, height_in: float) -> float:
    if weight is None or forty is None or forty <= 4.1 or forty >= 5.3:
        return 100.0
    speed_score = (weight * 200.0) / (forty ** 4)
    return speed_score * (height_in / 73.0)

def calculate_burst(vertical: float, broad: float) -> float:
    if vertical is None or broad is None or vertical <= 0 or broad <= 0:
        return 100.0
    raw_burst = vertical + (broad / 10.0)
    return (raw_burst / 47.6) * 100.0

def get_stage1_baseline(c_grade: float):
    """
    Deterministic Stage 1 College Baseline Quantiles.
    Calibrated exponent (1.75) grounds mid-tier grades while preserving alpha upside.
    """
    g_norm = max(0.0, (c_grade - 45.0) / 50.0)
    base_p50 = round(150.0 + 1850.0 * (g_norm ** 1.75))
    base_p90 = min(4850, round(650.0 + 3500.0 * (g_norm ** 1.15)))
    base_p10 = max(25.0, round(25.0 + 350.0 * (g_norm ** 1.60)))
    return base_p10, base_p50, base_p90

def calculate_calibrated_probabilities(c_grade: float, ceil_mult: float, med_mult: float):
    """
    Calibrated logistic probability equations anchored to historical base rates:
    - Quality Starter: 24.8%
    - Boom / WR1 Alpha: 12.1%
    - Bust / Flop: 50.0%
    """
    z_starter = -1.18 + (c_grade - 74.0) * 0.052 + (ceil_mult - 1.0) * 1.3
    starter_prob = min(0.92, max(0.04, 1.0 / (1.0 + math.exp(-z_starter))))

    z_boom = -2.12 + (c_grade - 74.0) * 0.058 + (ceil_mult - 1.0) * 1.6
    boom_prob = min(0.85, max(0.015, 1.0 / (1.0 + math.exp(-z_boom))))

    z_bust = 0.00 - (c_grade - 74.0) * 0.048 - (med_mult - 1.0) * 1.4
    bust_prob = min(0.95, max(0.03, 1.0 / (1.0 + math.exp(-z_bust))))

    return round(starter_prob, 3), round(boom_prob, 3), round(bust_prob, 3)

def score_prospect(name, year, college, round_pick, grade, forty, vert, broad, cone, shuttle, ht_in, wt_lbs, age=21.5, c_ratio=0.88):
    has_test = (forty is not None)
    base_p10, base_p50, base_p90 = get_stage1_baseline(grade)
    hass = calculate_hass(wt_lbs, forty, ht_in)
    burst = calculate_burst(vert, broad)
    agility_sum = (cone + shuttle) if (cone and shuttle) else None
    
    if not has_test:
        if grade >= 91.0:
            ceil_mult = 1.00
            med_mult = 1.00
        else:
            ceil_mult = 0.93
            med_mult = 0.94
        if c_ratio < 0.72 and grade < 89.0:
            ceil_mult *= 0.95
            med_mult *= 0.95
        elif c_ratio >= 0.90:
            ceil_mult *= 1.02
            med_mult *= 1.02
    else:
        hass_delta = (hass - 100.0) / 100.0
        burst_delta = (burst - 100.0) / 100.0
        athletic_index = (hass_delta * 0.7) + (burst_delta * 0.3)
        if athletic_index >= 0:
            ceil_mult = 1.0 + min(0.22, athletic_index * 0.75)
            med_mult = 1.0 + min(0.12, athletic_index * 0.35)
        else:
            ceil_mult = 1.0 + max(-0.22, athletic_index * 0.85)
            med_mult = 1.0 + max(-0.18, athletic_index * 0.70)
            
        if agility_sum is not None:
            if agility_sum <= 11.05:
                ag_boost = min(0.12, (11.05 - agility_sum) * 0.35)
                ceil_mult += ag_boost
                med_mult += ag_boost * 0.7
            elif agility_sum >= 11.55:
                ag_penalty = min(0.15, (agility_sum - 11.55) * 0.40)
                ceil_mult -= ag_penalty
                med_mult -= ag_penalty * 0.60
                
        phys_pen = 0.0
        if forty is not None and forty >= 4.62:
            if agility_sum is not None and agility_sum <= 11.15:
                phys_pen += 0.02
            else:
                phys_pen += 0.07
        if ht_in < 69.0 and (forty is None or forty > 4.46):
            phys_pen += 0.07
        if wt_lbs < 178.0 and (forty is None or forty > 4.42):
            phys_pen += 0.05
        if (86.0 <= grade < 91.0) and athletic_index <= -0.06:
            phys_pen += 0.08
        phys_pen = min(0.18, phys_pen)
        ceil_mult = max(0.70, ceil_mult * (1.0 - phys_pen))
        med_mult = max(0.72, med_mult * (1.0 - phys_pen))
        
    if age <= 21.2:
        ceil_mult *= 1.05
        med_mult *= 1.04
    elif age >= 22.3 and grade < 92.0:
        ceil_mult *= 0.94
        med_mult *= 0.94
        
    raw_p90 = round(base_p90 * ceil_mult)
    if grade >= 91.0:
        raw_p90 = max(raw_p90, 3100)
    p90 = min(4850, max(round(base_p50 * 1.3), raw_p90))
    p50 = round(base_p50 * med_mult)
    p10 = max(25.0, round(base_p10 * min(1.0, med_mult * 1.05)))
    
    p_starter, p_boom, p_bust = calculate_calibrated_probabilities(grade, ceil_mult, med_mult)

    if grade >= 92.0 and p_boom >= 0.28:
        profile = "Blue-Chip Alpha"
    elif p_starter >= 0.38 and p_bust <= 0.32:
        profile = "High-Floor Starter"
    elif p_boom >= 0.20:
        profile = "Boom-or-Bust Flier"
    else:
        profile = "High Bust Risk"

    return {
        "name": name,
        "draft_year": year,
        "college": college,
        "draft_slot_info": round_pick,
        "college_grade": grade,
        "forty_sec": forty,
        "hass": round(hass, 1),
        "burst_score": round(burst, 1),
        "agility_sum": agility_sum,
        "nfl_proj_floor": int(p10),
        "nfl_proj_median": int(p50),
        "nfl_proj_ceiling": int(p90),
        "prob_starter": p_starter,
        "prob_boom": p_boom,
        "prob_bust": p_bust,
        "prospect_profile": profile,
    }

RECENT_PROSPECTS = [
    # 2023 Draftees
    ("Jaxon Smith-Njigba", 2023, "Ohio State", "R1 Pick 20", 93.5, 4.52, 35.0, 125.0, 6.57, 3.93, 73.0, 196.0, 21.2),
    ("Zay Flowers", 2023, "Boston College", "R1 Pick 22", 91.5, 4.42, 35.5, 127.0, None, None, 69.2, 182.0, 22.6),
    ("Jordan Addison", 2023, "USC / Pitt", "R1 Pick 23", 91.2, 4.49, 34.0, 122.0, 7.05, 4.19, 71.1, 173.0, 21.3),
    ("Jayden Reed", 2023, "Michigan State", "R2 Pick 50", 89.0, 4.45, 33.5, 122.0, 7.05, 4.29, 71.0, 187.0, 22.9),
    ("Rashee Rice", 2023, "SMU", "R2 Pick 55", 88.5, 4.51, 41.0, 128.0, 7.02, 4.23, 73.0, 204.0, 22.9),
    ("Josh Downs", 2023, "North Carolina", "R3 Pick 79", 90.0, 4.48, 38.5, 131.0, 7.08, 4.24, 68.8, 171.0, 21.7),
    ("Tank Dell", 2023, "Houston", "R3 Pick 69", 89.0, 4.49, None, None, 7.07, 4.25, 68.3, 165.0, 23.4),
    ("Puka Nacua", 2023, "BYU", "R5 Pick 177", 89.5, 4.57, 33.0, 121.0, 7.32, 4.36, 73.5, 201.0, 21.9),
    ("Quentin Johnston", 2023, "TCU", "R1 Pick 21", 82.0, 4.51, 40.5, 134.0, 7.28, 4.28, 75.0, 208.0, 21.6),
    ("Jonathan Mingo", 2023, "Ole Miss", "R2 Pick 39", 81.5, 4.46, 39.5, 129.0, 7.37, 4.38, 73.7, 220.0, 22.0),
    ("Jalin Hyatt", 2023, "Tennessee", "R3 Pick 73", 83.0, 4.40, 40.0, 135.0, None, None, 72.1, 176.0, 21.6),
    ("Cedric Tillman", 2023, "Tennessee", "R3 Pick 74", 82.5, 4.54, 37.0, 128.0, 7.32, 4.36, 75.2, 213.0, 23.0),
    ("Dontayvion Wicks", 2023, "Virginia", "R5 Pick 159", 85.0, 4.62, 39.0, 130.0, 7.25, 4.30, 73.4, 206.0, 21.8),
    ("Demario Douglas", 2023, "Liberty", "R6 Pick 210", 82.0, 4.44, 39.5, 130.0, 7.05, 4.29, 68.0, 179.0, 22.3),

    # 2024 Draftees
    ("Marvin Harrison Jr.", 2024, "Ohio State", "R1 Pick 4", 98.2, 4.42, 38.0, 128.0, 6.85, 4.15, 75.2, 209.0, 21.8),
    ("Malik Nabers", 2024, "LSU", "R1 Pick 6", 97.5, 4.35, 42.0, 131.0, 6.80, 4.10, 72.0, 200.0, 20.8),
    ("Rome Odunze", 2024, "Washington", "R1 Pick 9", 96.0, 4.45, 39.0, 124.0, 6.88, 4.03, 75.0, 212.0, 21.9),
    ("Brian Thomas Jr.", 2024, "LSU", "R1 Pick 23", 92.5, 4.33, 38.5, 126.0, None, None, 74.9, 209.0, 21.6),
    ("Ladd McConkey", 2024, "Georgia", "R2 Pick 34", 91.0, 4.39, 36.0, 124.0, 6.72, 4.04, 71.7, 186.0, 22.4),
    ("Xavier Worthy", 2024, "Texas", "R1 Pick 28", 89.0, 4.21, 41.0, 131.0, 7.05, 4.25, 71.0, 165.0, 21.0),
    ("Ricky Pearsall", 2024, "Florida", "R1 Pick 31", 87.0, 4.41, 42.0, 129.0, 6.64, 4.05, 73.0, 189.0, 23.6),
    ("Xavier Legette", 2024, "South Carolina", "R1 Pick 32", 87.5, 4.39, 40.0, 126.0, 7.30, 4.25, 73.0, 221.0, 23.2),
    ("Keon Coleman", 2024, "Florida State", "R2 Pick 33", 84.5, 4.61, 38.0, 127.0, 7.06, 4.20, 75.2, 213.0, 21.0),
    ("Ja'Lynn Polk", 2024, "Washington", "R2 Pick 37", 86.0, 4.52, 37.5, 129.0, 7.15, 4.25, 73.2, 203.0, 22.1),
    ("Adonai Mitchell", 2024, "Texas", "R2 Pick 52", 88.0, 4.34, 39.5, 136.0, 7.05, 4.25, 74.2, 205.0, 21.6),
    ("Jermaine Burton", 2024, "Alabama", "R3 Pick 80", 87.0, 4.45, 38.5, 133.0, None, None, 72.2, 196.0, 22.8),
    ("Jalen McMillan", 2024, "Washington", "R3 Pick 92", 86.5, 4.47, 37.0, 127.0, 6.94, 4.18, 73.0, 197.0, 22.4),
    ("Troy Franklin", 2024, "Oregon", "R4 Pick 102", 88.0, 4.41, 39.0, 124.0, 6.90, 4.25, 73.9, 176.0, 21.3),
    ("Malik Washington", 2024, "Virginia", "R6 Pick 184", 89.0, 4.47, 42.5, 126.0, 6.87, 4.15, 68.5, 191.0, 23.5),

    # 2025 Draft Prospects
    ("Travis Hunter", 2025, "Colorado", "Top 5 Projected", 97.0, 4.40, 39.0, 128.0, 6.75, 4.05, 73.0, 185.0, 21.9),
    ("Tetairoa McMillan", 2025, "Arizona", "Top 10 Projected", 96.5, 4.52, 36.0, 124.0, 6.95, 4.18, 77.0, 212.0, 21.9),
    ("Luther Burden III", 2025, "Missouri", "Top 15 Projected", 95.0, 4.44, 37.5, 126.0, 6.85, 4.12, 71.5, 208.0, 21.4),
    ("Emeka Egbuka", 2025, "Ohio State", "Round 1 Projected", 92.0, 4.48, 36.5, 123.0, 6.80, 4.08, 73.0, 205.0, 22.5),
    ("Tre Harris", 2025, "Ole Miss", "Round 1-2 Projected", 93.0, 4.50, 36.0, 122.0, 6.95, 4.20, 74.0, 210.0, 23.1),
    ("Isaiah Bond", 2025, "Texas", "Round 1-2 Projected", 90.0, 4.36, 38.0, 128.0, 6.82, 4.15, 71.0, 182.0, 21.1),
    ("Matthew Golden", 2025, "Texas", "Day 2 Projected", 88.5, 4.42, 37.0, 125.0, 6.90, 4.18, 72.0, 195.0, 21.8),
    ("Elic Ayomanor", 2025, "Stanford", "Day 2 Projected", 89.0, 4.47, 38.0, 126.0, 7.02, 4.22, 74.0, 210.0, 21.5),
    ("Xavier Restrepo", 2025, "Miami", "Day 2-3 Projected", 88.0, 4.56, 35.0, 120.0, 6.98, 4.19, 70.0, 198.0, 22.8),
    ("Tai Felton", 2025, "Maryland", "Day 2-3 Projected", 88.0, 4.45, 36.5, 124.0, 7.05, 4.24, 73.5, 186.0, 22.3),
    ("Jack Bech", 2025, "TCU", "Day 2-3 Projected", 87.5, 4.52, 36.0, 122.0, 7.00, 4.20, 74.0, 220.0, 22.6),
]

def run_recent_predictions():
    records = [score_prospect(*p) for p in RECENT_PROSPECTS]
    
    if HAS_POLARS:
        df = pl.DataFrame(records)
        df.write_parquet(OUT_PARQUET)
        df.write_csv(OUT_CSV)
    else:
        pdf = pd.DataFrame(records)
        pdf.to_parquet(OUT_PARQUET)
        pdf.to_csv(OUT_CSV, index=False)
        
    print(f"Successfully processed {len(records)} prospects across 2023, 2024, and 2025 classes.")
    print(f"Saved to: {OUT_PARQUET}")
    print(f"Saved to: {OUT_CSV}")

    for yr in [2023, 2024, 2025]:
        sub_records = [r for r in records if r["draft_year"] == yr]
        sub_records.sort(key=lambda x: x["college_grade"], reverse=True)
        print("\n" + "=" * 92)
        print(f"CLASS OF {yr} PRE-DRAFT PROJECTIONS (ZERO DRAFT CAPITAL)")
        print("=" * 92)
        print(f"{'Prospect':<22} {'College':<16} {'Grade':<6} {'HaSS':<6} {'p10':<5} {'p50':<5} {'p90':<5} {'P(Starter)':<11} {'P(Boom)':<9} {'P(Bust)':<8} {'Profile'}")
        print("-" * 92)
        for r in sub_records:
            p_s = f"{r['prob_starter']*100:.1f}%"
            p_b = f"{r['prob_boom']*100:.1f}%"
            p_bu = f"{r['prob_bust']*100:.1f}%"
            print(f"{r['name']:<22} {r['college']:<16} {r['college_grade']:<6.1f} {r['hass']:<6.1f} {r['nfl_proj_floor']:<5d} {r['nfl_proj_median']:<5d} {r['nfl_proj_ceiling']:<5d} {p_s:<11} {p_b:<9} {p_bu:<8} {r['prospect_profile']}")

if __name__ == "__main__":
    run_recent_predictions()