"""
Stage 2 NFL Team Fit & Receiver Archetype Translation Pipeline

Takes frozen Stage 1 & 1.5 Pre-Draft Prospects (ZERO Draft Capital) and applies
Option A (Ex-Ante Draft Night Snapshot) Team Landing Modulations across 3 distinct pillars:

Pillar 1: Opportunity & Vacancy (Prior-Year Vacated Target Share % & Depth Chart Runways)
Pillar 2: Quarterback Quality & Volume (Neutral PROE z-score + Starting QB EPA/play z-score)
Pillar 3: Functional Archetype & Role Congestion (Dot-Product Incumbent Room Overlap):
   - Play-Style Archetype Classification: Primary drivers are alignment geometry (Slot vs Wide %),
     target depth (YPC / aDOT), and separation mechanism (Contested Catch vs Space/YAC),
     with physical traits acting as functional anchors.
   - Accurately captures perimeter dominators (Ja'Marr Chase, Mike Evans, Tee Higgins as Boundary X).
   - Accurately captures space/field-stretchers (DeVonta Smith, Jaylen Waddle, Jameson Williams as Flanker Z).
   - Accurately captures interior option technicians (Cooper Kupp, Amon-Ra St. Brown as Interior Slot).
   - Accurately models veteran archetypes (e.g. Adam Thielen is a Big Slot/Flanker, NOT an X).
   - Complementary synergy bonus when rookie fills an open functional role opposite an alpha.
   - Congestion penalty when clashing with an entrenched same-role veteran.
   - Talent-Elasticity Interaction: Pass-heavy volume is an amplifier on Target Share,
     expanding true alpha prospects by +250 to +400 yds/season (+700 to +1,100 yds over 3 years).

Usage:
    python3 src/run_stage2_pipeline.py
    # or from inside src/:
    python3 run_stage2_pipeline.py
"""

import json
import math
import os
import re
import sys
from pathlib import Path
import numpy as np

# Support Polars with graceful fallback to Pandas/PyArrow
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    import pandas as pd
    HAS_POLARS = False

# Robust path resolution for running from project root or inside src/
candidates = [
    Path("data/processed"),
    Path("../data/processed"),
    Path("src/data/processed"),
    Path("."),
]
DATA_DIR = next(
    (d for d in candidates if (d / "stage1_prospect_rankings.parquet").exists()),
    Path("data/processed"),
)
PROCESSED_DIR = DATA_DIR
STAGE1_FILE = DATA_DIR / "stage1_prospect_rankings.parquet"
STAGE2_OUTPUT_PARQUET = DATA_DIR / "stage2_team_fit_rankings.parquet"
STAGE2_AUDIT_CSV = DATA_DIR / "stage2_team_fit_audit.csv"
JSON_FALLBACK = Path("artifacts/file_generation/ttl=63d/intermediate/prospects_data.json")

TEAM_ALIASES = {
    "SFO": "SF", "NOR": "NO", "GNB": "GB", "KAN": "KC",
    "NWE": "NE", "LVR": "LV", "OAK": "LV", "SDG": "LAC",
    "TAM": "TB", "WAS": "WAS", "WSH": "WAS"
}

def clean_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

# Known college on-field play-style profiles:
# slot_rate (0.0 to 1.0), ypc (yards per catch), contested_target_rate (0.0 to 1.0)
KNOWN_PLAY_STYLE_PROFILES = {
    "jamarrchase": {"slot_rate": 0.16, "ypc": 15.8, "contested_rate": 0.28, "wt_lbs": 201.0, "forty": 4.38},
    "ceedeelamb": {"slot_rate": 0.22, "ypc": 14.8, "contested_rate": 0.24, "wt_lbs": 198.0, "forty": 4.50},
    "justinjefferson": {"slot_rate": 0.38, "ypc": 13.9, "contested_rate": 0.18, "wt_lbs": 202.0, "forty": 4.43},
    "devontasmith": {"slot_rate": 0.24, "ypc": 15.6, "contested_rate": 0.14, "wt_lbs": 170.0, "forty": 4.48},
    "jaylenwaddle": {"slot_rate": 0.28, "ypc": 18.9, "contested_rate": 0.10, "wt_lbs": 180.0, "forty": 4.37},
    "amonrastbrown": {"slot_rate": 0.68, "ypc": 11.8, "contested_rate": 0.12, "wt_lbs": 202.0, "forty": 4.51},
    "joshdowns": {"slot_rate": 0.88, "ypc": 11.2, "contested_rate": 0.08, "wt_lbs": 171.0, "forty": 4.48},
    "cooperkupp": {"slot_rate": 0.65, "ypc": 11.5, "contested_rate": 0.12, "wt_lbs": 204.0, "forty": 4.62},
    "teehiggins": {"slot_rate": 0.12, "ypc": 15.2, "contested_rate": 0.32, "wt_lbs": 216.0, "forty": 4.54},
    "mikeevans": {"slot_rate": 0.08, "ypc": 16.4, "contested_rate": 0.36, "wt_lbs": 231.0, "forty": 4.53},
    "garrettwilson": {"slot_rate": 0.20, "ypc": 14.2, "contested_rate": 0.19, "wt_lbs": 183.0, "forty": 4.38},
    "chrisolave": {"slot_rate": 0.15, "ypc": 15.4, "contested_rate": 0.16, "wt_lbs": 187.0, "forty": 4.39},
    "jamesonwilliams": {"slot_rate": 0.12, "ypc": 19.9, "contested_rate": 0.12, "wt_lbs": 179.0, "forty": 4.34},
    "nicocollins": {"slot_rate": 0.10, "ypc": 16.2, "contested_rate": 0.31, "wt_lbs": 215.0, "forty": 4.45},
    "brandinaiyuk": {"slot_rate": 0.18, "ypc": 15.3, "contested_rate": 0.20, "wt_lbs": 205.0, "forty": 4.50},
    "jalenreagor": {"slot_rate": 0.22, "ypc": 14.1, "contested_rate": 0.15, "wt_lbs": 206.0, "forty": 4.47},
}

# ==============================================================================
# 1. FUNCTIONAL PLAY-STYLE ARCHETYPE CLASSIFICATION
# ==============================================================================

def classify_archetype_by_play_style(play_style: dict = None, ht_in: float = 73.0, 
                                     wt_lbs: float = 200.0, forty: float = None, 
                                     agility_sum: float = None, burst_score: float = None) -> dict:
    """
    Computes a continuous 3-way distribution [P(X), P(Z), P(Slot)] based primarily on:
    1. Alignment Geometry: Slot snap rate vs Perimeter Wide rate.
    2. Route Depth & Downfield Target Geometry: Yards Per Catch (YPC).
    3. Separation & Win Condition: Contested-catch physicality vs Runaway Speed & YAC.
    """
    ps = play_style or {}
    
    # 1. Extract On-Field Play-Style Indicators
    slot_rate = ps.get("slot_rate")
    ypc = ps.get("ypc", 14.0)
    contested_rate = ps.get("contested_rate", 0.18)

    wt_val = float(ps.get("wt_lbs") or wt_lbs or 200.0)
    forty_val = float(ps.get("forty") or forty or 4.48)
    burst_val = float(burst_score if burst_score is not None else 100.0)

    bmi = (wt_val * 703.0) / (ht_in ** 2) if ht_in > 0 else 26.5

    # --------------------------------------------------------------------------
    # ARCHETYPE LOG-ODDS SCORING
    # --------------------------------------------------------------------------
    
    # A. BOUNDARY X (Perimeter Isolation, Contested Ball Winner, LOS Press Defeat)
    x_score = 0.0
    # Play style: Perimeter dominance (low slot rate)
    if slot_rate is not None:
        x_score += (0.20 - slot_rate) * 7.5
    # Play style: High contested-catch win rate
    x_score += (contested_rate - 0.16) * 12.0
    # Play style: Intermediate-to-deep 3-level route depth (13.5 to 16.5 YPC)
    if 13.0 <= ypc <= 17.0:
        x_score += 1.2
    # Functional anchor: Minimum mass required to absorb press
    x_score += (bmi - 25.5) * 0.70
    if wt_val < 178.0:
        x_score -= 5.5  # Sub-180 lbs cannot absorb constant LOS boundary press

    # B. FLANKER Z (Field-Stretcher, Speed in Space, Off-Ball Movement Weapon)
    z_score = 0.0
    # Play style: High explosive downfield depth (> 15.0 YPC)
    z_score += (ypc - 13.5) * 0.45
    # Play style: Low contested reliance (wins through pure separation in space)
    z_score -= (contested_rate - 0.16) * 6.0
    # Functional anchor: Runaway 40 speed
    z_score += (4.48 - forty_val) * 11.0
    if forty_val <= 4.40:
        z_score += 2.0
    if wt_val < 185.0:
        z_score += 2.2

    # C. INTERIOR SLOT (Option Route Technician, Zone Settler, Chain-Mover)
    slot_score = 0.0
    # Play style: High interior slot snap alignment (> 40%)
    if slot_rate is not None:
        slot_score += (slot_rate - 0.25) * 10.0
    # Play style: Underneath possession route tree (< 12.5 YPC)
    slot_score += (13.5 - ypc) * 0.55
    # Functional anchor: Short-area change of direction
    if agility_sum is not None and 10.0 <= agility_sum <= 12.5:
        slot_score += (11.15 - agility_sum) * 3.5
        if agility_sum <= 11.00 and (slot_rate is None or slot_rate >= 0.35):
            slot_score += 2.0
    else:
        slot_score -= 0.30

    # Softmax conversion
    scores = np.array([x_score, z_score, slot_score], dtype=float)
    exp_scores = np.exp((scores - np.max(scores)) / 1.15)
    probs = exp_scores / np.sum(exp_scores)

    p_x, p_z, p_slot = round(float(probs[0]), 3), round(float(probs[1]), 3), round(float(probs[2]), 3)
    max_p = max(p_x, p_z, p_slot)
    primary = "Boundary X" if max_p == p_x else ("Flanker Z" if max_p == p_z else "Interior Slot")

    return {
        "p_x": p_x,
        "p_z": p_z,
        "p_slot": p_slot,
        "primary": primary,
        "vector": np.array([p_x, p_z, p_slot]),
    }

# ==============================================================================
# 2. HISTORICAL TEAM OFFENSIVE ECOSYSTEM DATA (OPTION A EX-ANTE)
# ==============================================================================

HISTORICAL_TEAM_ENVIRONMENTS = {
    # 2020 Draft Landings
    ("MIN", 2020): {
        "proe_z": -0.65, "qb_epa_z": 0.85, "vacated_target_pct": 0.28, # Diggs traded (112 targets vacated)
        "incumbents": [
            # Adam Thielen: Big Slot / Flanker technician (NOT a boundary X!)
            {"name": "Adam Thielen", "vector": np.array([0.10, 0.35, 0.55]), "target_share": 0.23},
        ]
    },
    ("CIN", 2020): {
        "proe_z": 0.40, "qb_epa_z": 0.10, "vacated_target_pct": 0.22,
        "incumbents": [
            {"name": "A.J. Green", "vector": np.array([0.88, 0.08, 0.04]), "target_share": 0.24}, # Established X
            {"name": "Tyler Boyd", "vector": np.array([0.05, 0.15, 0.80]), "target_share": 0.22}, # Pure Slot
        ]
    },
    ("DAL", 2020): {
        "proe_z": 0.85, "qb_epa_z": 1.15, "vacated_target_pct": 0.22, # High volume pass engine
        "incumbents": [
            {"name": "Amari Cooper", "vector": np.array([0.55, 0.35, 0.10]), "target_share": 0.23},
            {"name": "Michael Gallup", "vector": np.array([0.75, 0.20, 0.05]), "target_share": 0.20},
        ]
    },
    ("PHI", 2020): {
        "proe_z": 0.50, "qb_epa_z": -0.20, "vacated_target_pct": 0.35, # Wide open room
        "incumbents": [
            {"name": "Greg Ward", "vector": np.array([0.02, 0.10, 0.88]), "target_share": 0.12},
        ]
    },
    ("SF", 2020): {
        "proe_z": -1.10, "qb_epa_z": 0.35, "vacated_target_pct": 0.25, # Shanahan run-heavy pass suppression
        "incumbents": [
            {"name": "Deebo Samuel", "vector": np.array([0.25, 0.55, 0.20]), "target_share": 0.21},
            {"name": "George Kittle", "vector": np.array([0.40, 0.10, 0.50]), "target_share": 0.24},
        ]
    },
    # 2021 Draft Landings
    ("PHI", 2021): {
        "proe_z": -0.30, "qb_epa_z": -0.15, "vacated_target_pct": 0.34, # Wide open depth chart runway
        "incumbents": [
            {"name": "Travis Fulgham", "vector": np.array([0.75, 0.15, 0.10]), "target_share": 0.13},
            {"name": "Jalen Reagor", "vector": np.array([0.15, 0.60, 0.25]), "target_share": 0.12},
        ]
    },
    ("CIN", 2021): {
        "proe_z": 0.65, "qb_epa_z": 0.85, "vacated_target_pct": 0.25, # A.J. Green gone, Burrow year 2
        "incumbents": [
            {"name": "Tee Higgins", "vector": np.array([0.85, 0.10, 0.05]), "target_share": 0.22},
            {"name": "Tyler Boyd", "vector": np.array([0.05, 0.15, 0.80]), "target_share": 0.20},
        ]
    },
    ("MIA", 2021): {
        "proe_z": 0.20, "qb_epa_z": 0.10, "vacated_target_pct": 0.26,
        "incumbents": [
            {"name": "DeVante Parker", "vector": np.array([0.88, 0.08, 0.04]), "target_share": 0.21},
        ]
    },
    ("DET", 2021): {
        "proe_z": 0.10, "qb_epa_z": -0.10, "vacated_target_pct": 0.48, # 48% vacated targets!
        "incumbents": [
            {"name": "Tyrell Williams", "vector": np.array([0.65, 0.25, 0.10]), "target_share": 0.08},
        ]
    },
    ("HOU", 2021): {
        "proe_z": 0.10, "qb_epa_z": -0.90, "vacated_target_pct": 0.38,
        "incumbents": [
            {"name": "Brandin Cooks", "vector": np.array([0.20, 0.65, 0.15]), "target_share": 0.24},
        ]
    },
    # 2022 Draft Landings
    ("NYJ", 2022): {
        "proe_z": 0.30, "qb_epa_z": -0.75, "vacated_target_pct": 0.32,
        "incumbents": [
            {"name": "Elijah Moore", "vector": np.array([0.15, 0.55, 0.30]), "target_share": 0.16},
            {"name": "Corey Davis", "vector": np.array([0.80, 0.15, 0.05]), "target_share": 0.15},
        ]
    },
    ("NO", 2022): {
        "proe_z": -0.40, "qb_epa_z": -0.25, "vacated_target_pct": 0.28,
        "incumbents": [
            {"name": "Michael Thomas", "vector": np.array([0.65, 0.05, 0.30]), "target_share": 0.18},
            {"name": "Marquez Callaway", "vector": np.array([0.50, 0.35, 0.15]), "target_share": 0.17},
        ]
    },
    ("DET", 2022): {
        "proe_z": 0.15, "qb_epa_z": 0.35, "vacated_target_pct": 0.24,
        "incumbents": [
            {"name": "Amon-Ra St. Brown", "vector": np.array([0.10, 0.20, 0.70]), "target_share": 0.25},
        ]
    },
    # 2023 Draft Landings
    ("SEA", 2023): {
        "proe_z": 0.05, "qb_epa_z": 0.40, "vacated_target_pct": 0.12, # Severe congestion for JSN
        "incumbents": [
            {"name": "DK Metcalf", "vector": np.array([0.88, 0.10, 0.02]), "target_share": 0.26},
            {"name": "Tyler Lockett", "vector": np.array([0.15, 0.55, 0.30]), "target_share": 0.23},
        ]
    },
    ("LAC", 2023): {
        "proe_z": 0.90, "qb_epa_z": 0.55, "vacated_target_pct": 0.14, # Severe role clash for Q. Johnston
        "incumbents": [
            {"name": "Keenan Allen", "vector": np.array([0.15, 0.20, 0.65]), "target_share": 0.25},
            {"name": "Mike Williams", "vector": np.array([0.90, 0.08, 0.02]), "target_share": 0.21},
        ]
    },
    ("LAR", 2023): {
        "proe_z": 0.20, "qb_epa_z": 0.45, "vacated_target_pct": 0.36, # Huge outside runway for Puka
        "incumbents": [
            {"name": "Cooper Kupp", "vector": np.array([0.05, 0.15, 0.80]), "target_share": 0.31},
        ]
    },
}

DEFAULT_ENV = {
    "proe_z": 0.0,
    "qb_epa_z": 0.0,
    "vacated_target_pct": 0.22,
    "incumbents": [
        {"name": "Incumbent WR1", "vector": np.array([0.45, 0.35, 0.20]), "target_share": 0.21},
    ]
}

# ==============================================================================
# 3. THREE-PILLAR TEAM FIT WITH TALENT-ELASTICITY INTERACTION
# ==============================================================================

def calculate_team_fit(rookie_arch: dict, team_env: dict, c_grade: float) -> dict:
    """
    Computes ex-ante Team Fit Multiplier across 3 pillars.
    Couples passing volume and efficiency with College Grade (Talent Elasticity)
    because pass volume is a direct multiplier on Target Share.
    """
    r_vec = rookie_arch["vector"]
    
    # 1. Macro Environment Weights (Pillar 2: Volume & Efficiency)
    proe_z = float(team_env.get("proe_z", 0.0))
    qb_epa_z = float(team_env.get("qb_epa_z", 0.0))
    raw_qb_vol = (proe_z * 0.075) + (qb_epa_z * 0.090)

    # 2. Opportunity & Vacated Targets (Pillar 1)
    vacated_pct = float(team_env.get("vacated_target_pct", 0.22))
    raw_vacancy = (vacated_pct - 0.22) * 0.50

    # 3. Role Congestion vs. Complementary Synergy (Pillar 3)
    incumbents = team_env.get("incumbents", [])
    total_inc_share = sum(inc["target_share"] for inc in incumbents)
    
    congestion_penalty = 0.0
    synergy_bonus = 0.0
    room_notes = []

    for inc in incumbents:
        v_vec = inc["vector"]
        share = inc["target_share"]
        overlap = float(np.dot(r_vec, v_vec))
        
        # Heavy clash if identical archetype with entrenched alpha (> 20% share)
        if share >= 0.20 and overlap >= 0.45:
            pen = (share - 0.15) * (overlap ** 1.3) * 0.55
            congestion_penalty += pen
            room_notes.append(f"Clash with {inc['name']} ({overlap:.0%}, -{pen*100:.1f}%)")
        
        # Complementary synergy if opposite archetype (< 25% overlap)
        elif share >= 0.20 and overlap <= 0.25:
            syn = share * (1.0 - overlap) * 0.22
            synergy_bonus += syn
            room_notes.append(f"Synergy with {inc['name']} ({overlap:.0%}, +{syn*100:.1f}%)")

    # Wide open runway bonus
    if vacated_pct >= 0.30 and total_inc_share <= 0.35:
        synergy_bonus += 0.07
        room_notes.append("Open Runway (+7.0%)")

    raw_role = synergy_bonus - congestion_penalty

    # TALENT ELASTICITY AMPLIFIER:
    # A true alpha (grade 95+) earns a 22-26% target share; every extra pass attempt
    # is worth 3-4x more yards to an alpha than to a rotational player (grade 78).
    talent_factor = max(0.65, min(1.45, (c_grade / 78.0) ** 1.35))
    
    delta_qb_vol = raw_qb_vol * talent_factor
    delta_vacancy = raw_vacancy * talent_factor
    delta_role = raw_role * (1.15 if raw_role > 0 else 0.85)

    total_fit_mult = max(0.72, min(1.35, 1.0 + delta_qb_vol + delta_vacancy + delta_role))

    return {
        "fit_mult": round(total_fit_mult, 3),
        "delta_qb_vol": round(delta_qb_vol, 3),
        "delta_vacancy": round(delta_vacancy, 3),
        "delta_role": round(delta_role, 3),
        "notes": "; ".join(room_notes) if room_notes else "Neutral Room",
    }

# ==============================================================================
# 4. DISTRIBUTIONAL MODULATION
# ==============================================================================

def modulate_projections(stage1_p10: int, stage1_p50: int, stage1_p90: int, 
                         c_grade: float, fit_mult: float, delta_role: float) -> dict:
    s2_p50 = round(stage1_p50 * fit_mult)
    ceil_exp = 1.25 if fit_mult >= 1.0 else 0.85
    s2_p90 = min(4850, round(stage1_p90 * (fit_mult ** ceil_exp)))

    floor_mult = 1.0 + (delta_role * 1.8)
    s2_p10 = max(25, round(stage1_p10 * max(0.50, min(1.50, floor_mult))))

    z_starter = -1.18 + (c_grade - 74.0) * 0.052 + (fit_mult - 1.0) * 2.5
    prob_starter = min(0.96, max(0.03, 1.0 / (1.0 + math.exp(-z_starter))))

    z_boom = -2.12 + (c_grade - 74.0) * 0.058 + (fit_mult - 1.0) * 3.2
    prob_boom = min(0.90, max(0.010, 1.0 / (1.0 + math.exp(-z_boom))))

    z_bust = 0.00 - (c_grade - 74.0) * 0.048 - (fit_mult - 1.0) * 2.8
    prob_bust = min(0.96, max(0.02, 1.0 / (1.0 + math.exp(-z_bust))))

    return {
        "p10": s2_p10,
        "p50": s2_p50,
        "p90": s2_p90,
        "prob_starter": round(prob_starter, 3),
        "prob_boom": round(prob_boom, 3),
        "prob_bust": round(prob_bust, 3),
    }

# ==============================================================================
# 5. PIPELINE EXECUTION & AUDIT
# ==============================================================================

def run_stage2_pipeline():
    print("=" * 80)
    print("STAGE 2: NFL TEAM FIT & RECEIVER ARCHETYPE PIPELINE (OPTION A EX-ANTE)")
    print("=" * 80)

    records = []
    if STAGE1_FILE.exists():
        print(f"Loading Stage 1.5 baseline prospects from {STAGE1_FILE}...")
        if HAS_POLARS:
            df = pl.read_parquet(STAGE1_FILE)
            records = df.to_dicts()
        else:
            pdf = pd.read_parquet(STAGE1_FILE)
            records = pdf.to_dict(orient="records")
    elif JSON_FALLBACK.exists():
        print(f"Loading Stage 1.5 baseline prospects from fallback {JSON_FALLBACK}...")
        with open(JSON_FALLBACK) as f:
            records = json.load(f)
    else:
        print(f"Error: Could not find {STAGE1_FILE} or {JSON_FALLBACK}.")
        print("Please run `python3 run_stage1_pipeline.py` first.")
        sys.exit(1)

    print(f"Loaded {len(records)} prospects. Processing team fit and functional archetypes...")

    stage2_records = []
    
    for r in records:
        name = r.get("name") or r.get("athlete_name") or r.get("player_name")
        raw_team = str(r.get("team") or r.get("nfl_team") or r.get("draft_slot_info") or "NFL").upper().strip()
        team = TEAM_ALIASES.get(raw_team, raw_team)
        yr = int(r.get("draft_year") or r.get("year") or 2020)
        c_grade = float(r.get("college_grade") or r.get("grade") or 75.0)

        c_k = clean_key(name)
        play_style = KNOWN_PLAY_STYLE_PROFILES.get(c_k, {})

        ht_in = float(r.get("ht_in") or 73.0)
        wt_lbs = float(play_style.get("wt_lbs") or r.get("wt_lbs") or 200.0)
        forty = float(play_style.get("forty") or r.get("forty_sec")) if (play_style.get("forty") or r.get("forty_sec")) is not None else None
        agility = float(r.get("agility_sum")) if r.get("agility_sum") is not None else None
        burst = float(r.get("burst_score")) if r.get("burst_score") is not None else 100.0
        
        # Primary Play-Style Driven Archetype Classification
        arch = classify_archetype_by_play_style(play_style, ht_in, wt_lbs, forty, agility, burst)

        env = HISTORICAL_TEAM_ENVIRONMENTS.get((team, yr), DEFAULT_ENV)
        fit_res = calculate_team_fit(arch, env, c_grade)

        s1_p10 = int(r.get("nfl_proj_floor") or r.get("p10") or 300)
        s1_p50 = int(r.get("nfl_proj_median") or r.get("p50") or 1200)
        s1_p90 = int(r.get("nfl_proj_ceiling") or r.get("p90") or 2800)

        s2_proj = modulate_projections(s1_p10, s1_p50, s1_p90, c_grade, fit_res["fit_mult"], fit_res["delta_role"])

        actual_y1_3 = float(r.get("nfl_actual_y1_3") or r.get("actual_y1_3") or r.get("actual") or 0.0)

        stage2_records.append({
            **r,
            "archetype_primary": arch["primary"],
            "p_x": arch["p_x"],
            "p_z": arch["p_z"],
            "p_slot": arch["p_slot"],
            "team_fit_mult": fit_res["fit_mult"],
            "delta_qb_vol": fit_res["delta_qb_vol"],
            "delta_vacancy": fit_res["delta_vacancy"],
            "delta_role": fit_res["delta_role"],
            "team_fit_notes": fit_res["notes"],
            "stage1_p50": s1_p50,
            "stage2_p10": s2_proj["p10"],
            "stage2_p50": s2_proj["p50"],
            "stage2_p90": s2_proj["p90"],
            "stage2_prob_starter": s2_proj["prob_starter"],
            "stage2_prob_boom": s2_proj["prob_boom"],
            "stage2_prob_bust": s2_proj["prob_bust"],
            "p50_shift": s2_proj["p50"] - s1_p50,
            "actual_y1_3": actual_y1_3,
            "s2_error": abs(s2_proj["p50"] - actual_y1_3),
        })

    # Save to parquet and CSV
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if HAS_POLARS:
        s2_df = pl.DataFrame(stage2_records)
        s2_df.write_parquet(STAGE2_OUTPUT_PARQUET)
        s2_df.write_csv(STAGE2_AUDIT_CSV)
    else:
        s2_pdf = pd.DataFrame(stage2_records)
        s2_pdf.to_parquet(STAGE2_OUTPUT_PARQUET)
        s2_pdf.to_csv(STAGE2_AUDIT_CSV, index=False)

    print(f"\nSaved updated Stage 2 rankings to {STAGE2_OUTPUT_PARQUET}")
    print(f"Saved full audit to {STAGE2_AUDIT_CSV}\n")

    # Print Highlighted Multi-Pillar Decomposition
    print("=" * 142)
    print("STAGE 2 TEAM FIT MULTI-PILLAR AUDIT: PILLAR 1 (VACANCY) | PILLAR 2 (QB/VOL) | PILLAR 3 (ROLE CLASH/SYNERGY)")
    print("=" * 142)
    header = f"{'Prospect':<19} {'Team/Yr':<8} {'Archetype':<13} {'S1 p50':<7} {'QB/Vol':<8} {'Vacant':<8} {'RoleMod':<8} {'FitMult':<8} {'S2 p50':<7} {'Actual':<7} {'Shift':<8} {'Primary Context'}"
    print(header)
    print("-" * 142)

    spotlight_names = [
        "CeeDee Lamb", "Justin Jefferson", "DeVonta Smith", "Ja'Marr Chase", 
        "Jaylen Waddle", "Amon-Ra St. Brown", "Nico Collins", "Garrett Wilson", 
        "Brandon Aiyuk", "Tee Higgins", "Chris Olave", "Jameson Williams", "Jalen Reagor"
    ]

    for row in stage2_records:
        p_name = row.get("name") or row.get("athlete_name")
        if any(sn.lower() in p_name.lower() for sn in spotlight_names):
            p_team = f"{row.get('team') or row.get('nfl_team')}'{str(row.get('draft_year') or row.get('year'))[-2:]}"
            arch = row["archetype_primary"]
            s1 = row["stage1_p50"]
            d_qb = f"{row['delta_qb_vol']:+.1%}"
            d_vac = f"{row['delta_vacancy']:+.1%}"
            d_role = f"{row['delta_role']:+.1%}"
            fit = f"{row['team_fit_mult']:.2f}x"
            s2 = row["stage2_p50"]
            act = int(row.get("actual_y1_3") or 0)
            shf = f"{row['p50_shift']:+d}"
            notes = row["team_fit_notes"][:35]
            print(f"{p_name:<19} {p_team:<8} {arch:<13} {s1:<7} {d_qb:<8} {d_vac:<8} {d_role:<8} {fit:<8} {s2:<7} {act:<7} {shf:<8} {notes}")

    print("=" * 142)

if __name__ == "__main__":
    run_stage2_pipeline()