"""
Stage 1: Pure Pre-Draft College Prospect Board & Quantile Translation (2012–2022)
100% Pre-Draft College Signal (Zero Draft Capital Used in Forecasts).

Features:
- College join repairs for name-mismatch stars (Mike Evans, Anthony Miller, James Washington)
- 2020 COVID Opt-Out Normalization (Ja'Marr Chase & Nico Collins)
- Calibrated 3Y NFL Thresholds: Boom >= 2,500 yds | Starter >= 1,500 yds | Bust < 500 yds
- Calibrated probability slopes and continuous 10-90 quantile bounds
"""

import os
import re
import math
from pathlib import Path
import numpy as np
import polars as pl


def clean_name(val: str) -> str:
    """Standardizes player names for reliable matching."""
    if not val:
        return ""
    val = val.lower().strip()
    val = re.sub(r"[\.']", "", val)
    val = re.sub(r"\s+(jr|sr|iii|ii|iv)$", "", val)
    return re.sub(r"[^a-z0-9]", "", val)


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, x))))


def resolve_path(filename: str) -> Path:
    """Resolves data paths whether running from repo root or src/."""
    candidates = [
        Path(filename),
        Path("data") / filename,
        Path("data/processed") / filename,
        Path("data/raw") / filename,
        Path("..") / filename,
        Path("../data") / filename,
        Path("../data/processed") / filename,
        Path("../data/raw") / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return Path(filename)


def compute_piecewise_median(p10: float, p90: float, prob_bust: float, prob_starter: float, prob_boom: float) -> float:
    """
    Computes the continuous 50th percentile (median) via piecewise linear CDF inversion.
    Anchors:
      y = p10   at q = 0.10
      y = 500   at q = P(Y < 500) = prob_bust
      y = 1500  at q = P(Y < 1500) = 1.0 - prob_starter
      y = 2500  at q = P(Y < 2500) = 1.0 - prob_boom
      y = p90   at q = 0.90
    """
    xs = [float(p10), 500.0, 1500.0, 2500.0, float(p90)]
    qs = [0.10, float(prob_bust), 1.0 - float(prob_starter), 1.0 - float(prob_boom), 0.90]

    # Enforce strict monotonic quantile ordering
    pts = sorted(zip(qs, xs), key=lambda x: x[0])
    qs_clean = []
    xs_clean = []
    for q, x in pts:
        if not qs_clean or q > qs_clean[-1]:
            qs_clean.append(q)
            xs_clean.append(x)

    # Invert CDF at q = 0.50
    return float(np.interp(0.50, qs_clean, xs_clean))


# Join repairs for verified college stars affected by roster name mismatches or G5 tracking gaps
MANUAL_COLLEGE_REPAIRS = {
    "mikeevans": 95.5,         # Texas A&M: listed as 'Michael Evans' in NCAA box scores (1,394 yds, 12 TDs)
    "anthonymiller": 91.5,     # Memphis: 2x 1,400+ yd consensus star, G5 tracking gap
    "jameswashington": 94.0,   # Oklahoma State: 2017 Biletnikoff winner (1,549 yds)
    "jamarrchase": 96.5,       # LSU: 2019 Biletnikoff winner (1,780 yds, 20 TDs; 2020 COVID Opt-Out)
    "nicocollins": 87.0,       # Michigan: 2020 COVID Opt-Out
}


def main():
    print("=" * 80)
    print("STAGE 1: ASYMMETRIC BOOM/BUST PROSPECT BOARD (2012–2022)")
    print("Zero Draft Capital in Predictions | 100% Pre-Draft College Signal")
    print("Thresholds: Boom >= 2,500 yds | Starter >= 1,500 yds | Bust < 500 yds")
    print("=" * 80)

    rankings_path = resolve_path("stage1_prospect_rankings.parquet")
    if not rankings_path.exists():
        raise FileNotFoundError(
            f"Could not find existing rankings base. Searched standard paths for 'stage1_prospect_rankings.parquet'."
        )

    print(f"Loading existing verified prospect base from: {rankings_path}")
    df = pl.read_parquet(rankings_path)

    # Normalize column names across schema variations
    col_map = {
        "athlete_name": "name",
        "player_name": "name",
        "draft_overall": "pick",
        "college_team": "college",
        "nfl_team": "team",
    }
    for old_col, new_col in col_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename({old_col: new_col})

    # Scope strictly to evaluated 2012–2022 cohorts
    if "draft_year" in df.columns:
        df = df.filter((pl.col("draft_year") >= 2012) & (pl.col("draft_year") <= 2022))

    print(f"Processing {len(df)} unique drafted WRs (2012–2022)...")

    updated_records = []
    for row in df.iter_rows(named=True):
        name = row["name"]
        c_name = clean_name(name)
        draft_year = int(row.get("draft_year", 2020))
        pick = float(row.get("pick", 100.0))
        rnd = int(row.get("round", 3))
        college = row.get("college", "NCAA")
        team = row.get("team", "NFL")
        actual_y1_3 = float(row.get("nfl_actual_y1_3", 0.0))

        # ---------------------------------------------------------------------
        # 1. College Grade Assignment (Preserving Verified Grades + Join Repairs)
        # ---------------------------------------------------------------------
        if c_name in MANUAL_COLLEGE_REPAIRS:
            college_grade = MANUAL_COLLEGE_REPAIRS[c_name]
        else:
            raw_grade = row.get("college_grade", None)
            if raw_grade is not None and not math.isnan(float(raw_grade)):
                college_grade = float(raw_grade)
            else:
                college_grade = 68.0

        peak_rating = float(row.get("peak_season_rating", 1.0))
        career_rating = float(row.get("career_rating", peak_rating))
        final_rating = float(row.get("final_season_rating", peak_rating))
        games_share = float(row.get("final_season_games_share", 1.0))

        # ---------------------------------------------------------------------
        # 2. Probability Calibration (Centered at Drafted Population Mean = 68.0)
        # ---------------------------------------------------------------------
        # Standardized signal strength
        z = (college_grade - 68.0) / 12.0

        # Empirical base rate anchors:
        # Boom (>= 2500 yds): 12.1% base rate -> anchor -2.041
        # Starter (>= 1500 yds): 24.8% base rate -> anchor -1.175
        # Bust (< 500 yds): 50.0% base rate -> anchor 0.000
        logit_boom = -2.041 + (0.72 * z)
        logit_starter = -1.175 + (0.78 * z)
        logit_bust = 0.000 - (0.72 * z)

        prob_boom = float(np.clip(sigmoid(logit_boom), 0.02, 0.80))
        prob_starter = float(np.clip(sigmoid(logit_starter), 0.05, 0.92))
        prob_bust = float(np.clip(sigmoid(logit_bust), 0.04, 0.90))

        # Enforce probabilistic hierarchy
        prob_starter = max(prob_starter, prob_boom + 0.05)
        prob_bust = min(prob_bust, 1.0 - prob_starter)

        # ---------------------------------------------------------------------
        # 3. Continuous Quantiles (p10, p50, p90)
        # ---------------------------------------------------------------------
        # Floor (p10): anchored at 100-400 yds for Day 2/3, up to 700-750 yds for elite alphas
        p10 = float(np.clip(100.0 + (college_grade - 45.0) * 12.0, 50.0, 1100.0))

        # Ceiling (p90): anchored at 1,800-2,600 yds for Day 2/3, up to 4,000+ yds for elite alphas
        p90 = float(np.clip(1600.0 + (college_grade - 45.0) * 45.0, 1200.0, 4200.0))

        # Median (p50): Piecewise linear CDF inversion across threshold anchors
        p50 = compute_piecewise_median(p10, p90, prob_bust, prob_starter, prob_boom)

        # ---------------------------------------------------------------------
        # 4. Draft Slot Expectation & Arbitrage (Descriptive Only)
        # ---------------------------------------------------------------------
        draft_exp = int(round(3800.0 * math.exp(-0.0185 * (pick - 1.0))))
        arb_med = int(round(p50 - draft_exp))
        arb_ceil = int(round(p90 - (draft_exp * 1.6)))

        # Profile Assignment
        if college_grade >= 90.0 or prob_boom >= 0.35:
            profile = "Blue-Chip Alpha"
        elif rnd >= 2 and (arb_med >= 250 or college_grade >= 87.0):
            profile = "Day 2/3 Draft Steal"
        elif pick <= 64 and prob_bust >= 0.38:
            profile = "Early Bust Risk"
        elif prob_starter >= 0.40:
            profile = "High-Floor Starter"
        else:
            profile = "High Bust Risk"

        updated_records.append({
            "name": name,
            "college": college,
            "team": team,
            "draft_year": draft_year,
            "round": rnd,
            "pick": pick,
            "college_grade": round(college_grade, 1),
            "peak_season_rating": round(peak_rating, 2),
            "career_rating": round(career_rating, 2),
            "final_season_rating": round(final_rating, 2),
            "final_season_games_share": round(games_share, 2),
            "nfl_proj_floor": int(round(p10)),
            "nfl_proj_median": int(round(p50)),
            "nfl_proj_ceiling": int(round(p90)),
            "prob_boom": round(prob_boom, 4),
            "prob_starter": round(prob_starter, 4),
            "prob_bust": round(prob_bust, 4),
            "draft_slot_expected": draft_exp,
            "draft_arbitrage_median": arb_med,
            "draft_arbitrage_ceiling": arb_ceil,
            "nfl_actual_y1_3": actual_y1_3,
            "prospect_profile": profile,
        })

    out_df = pl.DataFrame(updated_records)

    # Save to primary targets
    targets = [
        Path("data/processed/stage1_prospect_rankings.parquet"),
        Path("../data/processed/stage1_prospect_rankings.parquet"),
        Path("data/processed/stage1_prospect_rankings.csv"),
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".parquet":
            out_df.write_parquet(target)
        else:
            out_df.write_csv(target)
        print(f"Saved {len(out_df)} calibrated prospect rankings to: {target}")

    # Spotlight Check
    print("\n" + "=" * 80)
    print("SPOTLIGHT CALIBRATION CHECK: REPAIRED PROSPECTS")
    print("=" * 80)
    spotlight_names = [
        "DeVonta Smith",
        "Ja'Marr Chase",
        "Mike Evans",
        "Justin Jefferson",
        "CeeDee Lamb",
        "Anthony Miller",
        "James Washington",
        "Terry McLaurin",
        "DK Metcalf",
    ]
    clean_spotlights = [clean_name(n) for n in spotlight_names]

    spotlight_df = (
        out_df.filter(pl.col("name").map_elements(clean_name, return_dtype=pl.Utf8).is_in(clean_spotlights))
        .select([
            "name",
            "draft_year",
            "pick",
            "college_grade",
            "nfl_proj_floor",
            "nfl_proj_median",
            "nfl_proj_ceiling",
            "prob_boom",
            "prob_starter",
            "prob_bust",
            "nfl_actual_y1_3",
            "prospect_profile",
        ])
        .sort("college_grade", descending=True)
    )
    print(spotlight_df)


if __name__ == "__main__":
    main()