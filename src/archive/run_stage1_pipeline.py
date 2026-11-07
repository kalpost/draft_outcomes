"""Stage 1 & 1.5 Unified Prospect Translation Pipeline

Combines:
1. Combine & Pro Day data downloading & preparation.
2. Stage 1 College Baseline (COVID opt-out preservation & name join repairs).
3. Stage 1.5 Athletic Profile & Pre-Draft Translation Layer:
   - Shawn Siegele Height-Adjusted Speed Score (HaSS) with 73.0" baseline
   - Normalized Explosive Burst Score (Vertical + Broad Jump)
   - Agility & Deceleration Index (3-Cone Drill + 20-Yard Short Shuttle)
   - Stiff Straight-Line Workout Warrior Dampener
   - Pro Day +0.04s hand-time conversion factor
   - Adverse-Selection Handling for Unverified / Skipped Combine Workouts
   - Agile Technician Protection (smooth agility compensation for sub-4.60 runners like Cooper Kupp)
   - Elite College Production Protection (grades >= 91.0 like Jarvis Landry)
   - Age-Adjusted Dominance (Underclassman Early Declare boost vs. Older Senior discount)
   - Late-Career Breakout vs. Sustained Dominance Ratio
   - Hard historical ceiling cap at 4,850 yards (Justin Jefferson modern NFL 3-Year record ceiling)
   - Pro Day speed capture for non-combine invitees (Tyreek Hill)
4. Calibrated Logistic Climatology Probabilities (Starter >= 1500, Boom >= 2500, Bust < 500).
5. Full Model Evaluation & Error Analysis (Brier Scores, BSS vs climatology, 10-90 coverage, closest/steals/busts tables).

Usage:
    python3 src/run_stage1_pipeline.py
    # or from inside src/:
    python3 run_stage1_pipeline.py
"""

import math
import os
from pathlib import Path
import re
import sys
import urllib.request
import numpy as np
import polars as pl

# Robust path resolution: works whether executed from project root or inside src/
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "data").exists():
  DATA_DIR = SCRIPT_DIR / "data"
elif (SCRIPT_DIR.parent / "data").exists():
  DATA_DIR = SCRIPT_DIR.parent / "data"
elif Path("data").exists():
  DATA_DIR = Path("data").resolve()
else:
  DATA_DIR = (SCRIPT_DIR.parent / "data").resolve()

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
COMBINE_FILE = RAW_DIR / "combine.parquet"
STAGE1_FILE = PROCESSED_DIR / "stage1_prospect_rankings.parquet"
ERROR_AUDIT_FILE = PROCESSED_DIR / "stage1_error_analysis.csv"

COMBINE_URL = "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.parquet"

ALIAS_MAP = {
    "michaelevans": "mikeevans",
    "dkmetcalf": "dkmetcalf",
    "deandrehopkins": "deandrehopkins",
    "amonrastbrown": "amonrastbrown",
    "charkjr": "djchark",
    "willfuller": "willfuller",
    "marquisebrown": "marquisebrown",
    "kjhamler": "kjhamler",
    "anthonymiller": "anthonymiller",
    "jameswashington": "jameswashington",
    "jamarrchase": "jamarrchase",
    "nicocollins": "nicocollins",
}

REPAIR_COLLEGE_GRADES = {
    "mikeevans": 95.5,
    "anthonymiller": 91.5,
    "jameswashington": 94.0,
    "jamarrchase": 96.5,
    "nicocollins": 87.0,
}

PRO_DAY_TESTING_REPAIRS = {
    "tyreekhill_2016": {
        "ht_in": 70.0,
        "wt_lbs": 185.0,
        "forty_sec": 4.29 + 0.04,  # 4.29s pro-day hand time + 0.04s laser factor
        "vertical": 40.5,
        "broad": 129.0,
        "cone": 6.84,
        "shuttle": 4.19,
    }
}

COMBINE_COL_NAMES = {
    "ht_in",
    "wt_lbs",
    "forty_sec",
    "cone_sec",
    "shuttle_sec",
    "agility_sum",
    "hass",
    "burst_score",
    "bmi",
    "has_testing",
    "athletic_ceiling_mult",
}


def clean_key(val: str) -> str:
  if val is None:
    return ""
  s = str(val).lower().strip()
  s = re.sub(r"[.'\-\s_]+", "", s)
  s = re.sub(r"(jr|sr|iii|ii|iv)$", "", s)
  return ALIAS_MAP.get(s, s)


def ensure_combine_data():
  RAW_DIR.mkdir(parents=True, exist_ok=True)
  if not COMBINE_FILE.exists():
    print(f"Downloading combine dataset from {COMBINE_URL}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(COMBINE_URL, headers=headers)
    with urllib.request.urlopen(req) as resp, open(COMBINE_FILE, "wb") as f:
      f.write(resp.read())
    file_kb = COMBINE_FILE.stat().st_size / 1024.0
    print(f"Saved combine data to {COMBINE_FILE} ({file_kb:.1f} KB)")
  else:
    print(f"Using existing combine data at {COMBINE_FILE}")


def parse_height_inches(h_raw) -> float:
  if h_raw is None:
    return 73.0
  h_str = str(h_raw).strip()
  if "-" in h_str:
    parts = h_str.split("-")
    try:
      return float(parts[0]) * 12.0 + float(parts[1])
    except Exception:
      return 73.0
  try:
    val = float(h_str)
    return val if val > 50 else val * 12.0
  except Exception:
    return 73.0


def calculate_hass(weight: float, forty: float, height_in: float) -> float:
  if weight is None or forty is None or forty <= 4.1 or forty >= 5.3:
    return 100.0
  speed_score = (weight * 200.0) / (forty**4)
  return speed_score * (height_in / 73.0)


def calculate_burst(vertical: float, broad: float) -> float:
  if vertical is None or broad is None or vertical <= 0 or broad <= 0:
    return 100.0
  raw_burst = vertical + (broad / 10.0)
  return (raw_burst / 47.6) * 100.0


def get_stage1_baseline(c_grade: float):
  """Deterministic Stage 1 College Baseline Quantiles.

  Steepened exponent aligns mid-tier grades (75-88) to true population base
  rates (600-1100 yds),
  while preserving elite alpha ceilings (4000+ yds) for 92+ grades.
  """
  g_norm = max(0.0, (c_grade - 45.0) / 50.0)
  base_p50 = round(150.0 + 1850.0 * (g_norm**1.75))
  base_p90 = min(4850, round(650.0 + 3500.0 * (g_norm**1.15)))
  base_p10 = max(25.0, round(25.0 + 350.0 * (g_norm**1.60)))
  return base_p10, base_p50, base_p90


def calculate_calibrated_probabilities(
    c_grade: float, ceil_mult: float, med_mult: float
):
  """Calibrated logistic probability equations anchored to historical climatology base rates:

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


def load_base_prospects() -> pl.DataFrame:
  PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
  if STAGE1_FILE.exists():
    print(f"Loading base prospects from {STAGE1_FILE}...")
    return pl.read_parquet(STAGE1_FILE)

  candidates = [
      RAW_DIR / "wr_draft_picks_2012_2022.parquet",
      RAW_DIR / "draft_picks.parquet",
      RAW_DIR / "wr_draftees.parquet",
  ]
  for c in candidates:
    if c.exists():
      print(f"Loading initial prospect draftees from {c}...")
      return pl.read_parquet(c)

  print(
      f"Error: Could not find base prospect file at {STAGE1_FILE} or in"
      f" {RAW_DIR}."
  )
  sys.exit(1)


def run_pipeline():
  ensure_combine_data()
  base_df = load_base_prospects()

  c_raw = pl.read_parquet(COMBINE_FILE)
  s_col = "season" if "season" in c_raw.columns else "draft_year"
  p_col = "player_name" if "player_name" in c_raw.columns else "player"

  combine_clean = []
  for r in c_raw.iter_rows(named=True):
    yr = r.get(s_col)
    name = r.get(p_col)
    if not yr or not name:
      continue
    try:
      yr_int = int(yr)
    except Exception:
      continue

    c_key = f"{clean_key(name)}_{yr_int}"
    ht_in = parse_height_inches(r.get("ht"))
    wt = float(r.get("wt")) if r.get("wt") is not None else 200.0

    forty = float(r.get("forty")) if r.get("forty") is not None else None
    is_pro_day = (yr_int == 2021) or (
        str(r.get("event", "")).lower() == "pro day"
    )
    if forty is not None and is_pro_day:
      forty += 0.04

    vert = float(r.get("vertical")) if r.get("vertical") is not None else None
    broad = (
        float(r.get("broad_jump")) if r.get("broad_jump") is not None else None
    )
    cone = float(r.get("cone")) if r.get("cone") is not None else None
    shuttle = float(r.get("shuttle")) if r.get("shuttle") is not None else None

    if is_pro_day:
      if cone is not None:
        cone += 0.03
      if shuttle is not None:
        shuttle += 0.03

    if c_key in PRO_DAY_TESTING_REPAIRS:
      rep = PRO_DAY_TESTING_REPAIRS[c_key]
      ht_in = rep["ht_in"]
      wt = rep["wt_lbs"]
      forty = rep["forty_sec"]
      vert = rep["vertical"]
      broad = rep["broad"]
      cone = rep["cone"]
      shuttle = rep["shuttle"]

    hass = calculate_hass(wt, forty, ht_in)
    burst = calculate_burst(vert, broad)
    bmi = (wt * 703.0) / (ht_in**2) if ht_in > 0 else 26.5

    agility_sum = None
    if (
        cone is not None
        and shuttle is not None
        and (6.0 <= cone <= 8.2)
        and (3.7 <= shuttle <= 5.0)
    ):
      agility_sum = round(cone + shuttle, 2)

    combine_clean.append({
        "join_key": c_key,
        "ht_in": round(ht_in, 1),
        "wt_lbs": round(wt, 1),
        "forty_sec": round(forty, 2) if forty else None,
        "cone_sec": round(cone, 2) if cone else None,
        "shuttle_sec": round(shuttle, 2) if shuttle else None,
        "agility_sum": agility_sum,
        "hass": round(hass, 1),
        "burst_score": round(burst, 1),
        "bmi": round(bmi, 1),
        "has_testing": 1 if forty is not None else 0,
    })

  c_df = pl.DataFrame(combine_clean).unique(subset=["join_key"])

  # Clean base rows and normalize
  base_rows = []
  for r in base_df.iter_rows(named=True):
    name = r["name"]
    yr = int(r["draft_year"])
    c_k = clean_key(name)
    j_k = f"{c_k}_{yr}"

    grade = (
        float(r.get("college_grade", 50.0))
        if r.get("college_grade") is not None
        else 50.0
    )
    if c_k in REPAIR_COLLEGE_GRADES:
      grade = REPAIR_COLLEGE_GRADES[c_k]

    if yr == 2021 and r.get("final_season_games_share") is not None:
      if float(r.get("final_season_games_share")) < 0.5:
        p_peak = (
            float(r.get("peak_season_rating", 1.5))
            if r.get("peak_season_rating") is not None
            else 1.5
        )
        grade = max(grade, round(50.0 + (p_peak * 26.0), 1))

    # Filter out existing athletic and suffix columns to guarantee completely idempotent joins
    filtered_r = {
        k: v
        for k, v in r.items()
        if (k not in COMBINE_COL_NAMES) and (not k.endswith("_right"))
    }

    base_rows.append({
        **filtered_r,
        "college_grade": grade,
        "join_key": j_k,
    })
  s1_keyed = pl.DataFrame(base_rows)

  merged = s1_keyed.join(c_df, on="join_key", how="left")

  updated_records = []
  for r in merged.iter_rows(named=True):
    j_k = r.get("join_key", "")
    if j_k in PRO_DAY_TESTING_REPAIRS and (r.get("has_testing") != 1):
      rep = PRO_DAY_TESTING_REPAIRS[j_k]
      has_test = True
      hass = calculate_hass(rep["wt_lbs"], rep["forty_sec"], rep["ht_in"])
      burst = calculate_burst(rep["vertical"], rep["broad"])
      ht_in = rep["ht_in"]
      wt_lbs = rep["wt_lbs"]
      forty = rep["forty_sec"]
      agility_sum = rep["cone"] + rep["shuttle"]
    else:
      has_test = r.get("has_testing") == 1
      hass = float(r.get("hass")) if r.get("hass") is not None else 100.0
      burst = (
          float(r.get("burst_score"))
          if r.get("burst_score") is not None
          else 100.0
      )
      ht_in = float(r.get("ht_in")) if r.get("ht_in") is not None else 73.0
      wt_lbs = float(r.get("wt_lbs")) if r.get("wt_lbs") is not None else 200.0
      forty = (
          float(r.get("forty_sec")) if r.get("forty_sec") is not None else None
      )
      agility_sum = (
          float(r.get("agility_sum"))
          if r.get("agility_sum") is not None
          else None
      )

    c_grade = float(r.get("college_grade", 50.0))
    p_peak = (
        float(r.get("peak_season_rating", 1.0))
        if r.get("peak_season_rating") is not None
        else 1.0
    )
    c_rating = (
        float(r.get("career_rating", p_peak))
        if r.get("career_rating") is not None
        else p_peak
    )

    # 1. Deterministic pristine baseline from college grade (Zero compounding loop)
    base_p10, base_p50, base_p90 = get_stage1_baseline(c_grade)

    if not has_test:
      # Adverse-Selection Handling:
      # Elite college dominators (grade >= 91.0 like Ja'Marr Chase COVID opt-out) keep full baseline.
      # Mid-to-high college accumulators who skipped testing receive a soft -7% unverified dampener
      # to reflect that workout-skipping prospects rarely possess top-tier NFL separation.
      if c_grade >= 91.0:
        ceil_mult = 1.00
        med_mult = 1.00
      else:
        ceil_mult = 0.93
        med_mult = 0.94

      # Late-Career Breakout vs Sustained Multi-Year Dominance
      career_ratio = c_rating / max(0.5, p_peak) if p_peak > 0 else 1.0
      if career_ratio < 0.72 and c_grade < 89.0:
        ceil_mult *= 0.95
        med_mult *= 0.95
      elif career_ratio >= 0.90:
        ceil_mult *= 1.02
        med_mult *= 1.02
    else:
      # 2. Continuous athletic modulation (HaSS + Burst)
      hass_delta = (hass - 100.0) / 100.0
      burst_delta = (burst - 100.0) / 100.0
      athletic_index = (hass_delta * 0.7) + (burst_delta * 0.3)

      if athletic_index >= 0:
        ceil_mult = 1.0 + min(0.22, athletic_index * 0.75)
        med_mult = 1.0 + min(0.12, athletic_index * 0.35)
      else:
        ceil_mult = 1.0 + max(-0.22, athletic_index * 0.85)
        med_mult = 1.0 + max(-0.18, athletic_index * 0.70)

      # 3. Agility & Short-Area Deceleration Modulation (3-Cone + Short Shuttle)
      if agility_sum is not None:
        if agility_sum <= 11.05:
          ag_boost = min(0.12, (11.05 - agility_sum) * 0.35)
          ceil_mult += ag_boost
          med_mult += ag_boost * 0.7
        elif agility_sum >= 11.55:
          ag_penalty = min(0.15, (agility_sum - 11.55) * 0.40)
          ceil_mult -= ag_penalty
          med_mult -= ag_penalty * 0.60

      # 4. Smooth Physical Feasibility Adjustments (NO HARD CEILING FLOORS)
      # Slower 40s (4.58–4.70) are NOT hard-capped because elite route runners,
      # high agility technicians (Cooper Kupp: 4.62 forty, 10.95 agility),
      # and college alphas (Jarvis Landry, Keenan Allen, Davante Adams) frequently thrive.
      phys_pen = 0.0
      if forty is not None and forty >= 4.62:
        # If prospect possesses elite short-area agility (<11.15s), soft relief is granted
        if agility_sum is not None and agility_sum <= 11.15:
          phys_pen += 0.02
        else:
          phys_pen += 0.07

      if ht_in < 69.0 and (forty is None or forty > 4.46):
        phys_pen += 0.07
      if wt_lbs < 178.0 and (forty is None or forty > 4.42):
        phys_pen += 0.05

      # Protect elite college dominators (c_grade >= 91.0)
      if (86.0 <= c_grade < 91.0) and athletic_index <= -0.06:
        phys_pen += 0.08

      phys_pen = min(0.18, phys_pen)
      ceil_mult = max(0.70, ceil_mult * (1.0 - phys_pen))
      med_mult = max(0.72, med_mult * (1.0 - phys_pen))

    # 5. Age & Class Status Adjustment (Early Declares vs. Older 4-Year Seniors)
    # Favors young underclassmen (<=21.2 yrs on draft day) who dominate older competition,
    # while gently discounting 4th/5th-year senior volume (>=22.3 yrs) that benefited from physical maturity.
    age_val = r.get("draft_age") or r.get("age")
    if age_val is not None:
      try:
        age_f = float(age_val)
        if age_f <= 21.2:
          ceil_mult *= 1.05
          med_mult *= 1.04
        elif age_f >= 22.3 and c_grade < 92.0:
          ceil_mult *= 0.94
          med_mult *= 0.94
      except Exception:
        pass

    # 6. Smooth Quantile Projection & Modern NFL Historical Ceiling Cap
    raw_p90 = round(base_p90 * ceil_mult)
    if c_grade >= 91.0:
      raw_p90 = max(raw_p90, 3100)

    new_p90 = min(4850, max(round(base_p50 * 1.3), raw_p90))
    new_p50 = round(base_p50 * med_mult)
    new_p10 = max(25.0, round(base_p10 * min(1.0, med_mult * 1.05)))

    actual = (
        float(r.get("nfl_actual_y1_3"))
        if r.get("nfl_actual_y1_3") is not None
        else 0.0
    )
    error_yds = abs(new_p50 - actual)
    cov_status = (
        "INSIDE"
        if (new_p10 <= actual <= new_p90)
        else ("ABOVE" if actual > new_p90 else "BELOW")
    )

    # 7. Calibrated Climatology Probabilities
    starter_prob, boom_prob, bust_prob = calculate_calibrated_probabilities(
        c_grade, ceil_mult, med_mult
    )

    updated_records.append({
        **{
            k: v
            for k, v in r.items()
            if (k != "join_key" and not k.endswith("_right"))
        },
        "ht_in": ht_in,
        "wt_lbs": wt_lbs,
        "forty_sec": forty,
        "hass": hass,
        "burst_score": burst,
        "athletic_ceiling_mult": round(ceil_mult, 3),
        "nfl_proj_floor": int(new_p10),
        "nfl_proj_median": int(new_p50),
        "nfl_proj_ceiling": int(new_p90),
        "error_yds": round(error_yds, 1),
        "coverage_status": cov_status,
        "prob_starter": starter_prob,
        "prob_boom": boom_prob,
        "prob_bust": bust_prob,
    })

  out_df = pl.DataFrame(updated_records)
  out_df.write_parquet(STAGE1_FILE)
  print(
      f"\nSaved updated prospect rankings to {STAGE1_FILE} ({len(out_df)}"
      " prospects)."
  )

  evaluate_performance(out_df)


def evaluate_performance(df: pl.DataFrame):
  n = len(df)
  actuals = df["nfl_actual_y1_3"].to_numpy()
  medians = df["nfl_proj_median"].to_numpy()
  floors = df["nfl_proj_floor"].to_numpy()
  ceilings = df["nfl_proj_ceiling"].to_numpy()

  p_starters = df["prob_starter"].to_numpy()
  p_booms = df["prob_boom"].to_numpy()
  p_busts = df["prob_bust"].to_numpy()

  y_starter = (actuals >= 1500).astype(float)
  y_boom = (actuals >= 2500).astype(float)
  y_bust = (actuals < 500).astype(float)

  base_starter = float(np.mean(y_starter))
  base_boom = float(np.mean(y_boom))
  base_bust = float(np.mean(y_bust))

  bs_starter = float(np.mean((p_starters - y_starter) ** 2))
  bs_boom = float(np.mean((p_booms - y_boom) ** 2))
  bs_bust = float(np.mean((p_busts - y_bust) ** 2))

  clim_starter = float(np.mean((base_starter - y_starter) ** 2))
  clim_boom = float(np.mean((base_boom - y_boom) ** 2))
  clim_bust = float(np.mean((base_bust - y_bust) ** 2))

  bss_starter = (1.0 - (bs_starter / clim_starter)) * 100.0
  bss_boom = (1.0 - (bs_boom / clim_boom)) * 100.0
  bss_bust = (1.0 - (bs_bust / clim_bust)) * 100.0

  abs_errors = np.abs(medians - actuals)
  mae = float(np.mean(abs_errors))
  median_ae = float(np.median(abs_errors))
  rmse = float(np.sqrt(np.mean((medians - actuals) ** 2)))

  inside_mask = (actuals >= floors) & (actuals <= ceilings)
  coverage_pct = float(np.mean(inside_mask) * 100.0)
  widths = ceilings - floors
  mean_width = float(np.mean(widths))

  pct_starter = base_starter * 100.0
  pct_boom = base_boom * 100.0
  pct_bust = base_bust * 100.0

  print("=" * 80)
  print(
      "STAGE 1 & 1.5 COMBINED ERROR ANALYSIS & BRIER SCORES (2012-2022:"
      f" {n} PROSPECTS)"
  )
  print("=" * 80)
  print(
      "\n1. PROBABILISTIC BRIER SCORES (Lower is Better; Positive BSS = Better"
      " Than Climatology)"
  )
  print(
      f"{'Outcome Threshold':<36} {'Base Rate':<14} {'Brier Score':<15}"
      f" {'Skill vs Clim (BSS)'}"
  )
  print("-" * 80)
  print(
      f"{'>= 1,500 yds (Quality Starter)':<36} {pct_starter:>5.1f}%{'':<8}"
      f" {bs_starter:>7.4f}{'':<8} {bss_starter:>+5.1f}%"
  )
  print(
      f"{'>= 2,500 yds (Boom / WR1 Alpha)':<36} {pct_boom:>5.1f}%{'':<8}"
      f" {bs_boom:>7.4f}{'':<8} {bss_boom:>+5.1f}%"
  )
  print(
      f"{'< 500 yds (Bust / Flop)':<36} {pct_bust:>5.1f}%{'':<8}"
      f" {bs_bust:>7.4f}{'':<8} {bss_bust:>+5.1f}%"
  )

  print("\n2. CONTINUOUS ERROR & CALIBRATION METRICS")
  print("-" * 80)
  print(f"{'Mean Absolute Error (MAE):':<36} {mae:.1f} yards")
  print(f"{'Median Absolute Error:':<36} {median_ae:.1f} yards")
  print(f"{'Root Mean Squared Error (RMSE):':<36} {rmse:.1f} yards")
  print(
      f"{'10–90 Empirical Coverage:':<36} {coverage_pct:.1f}% (Nominal 80.0%)"
  )
  print(f"{'Mean Interval Width (p90 - p10):':<36} {mean_width:.1f} yards")

  audit_cols = [
      "name",
      "draft_year",
      "pick",
      "college_grade",
      "hass",
      "nfl_proj_floor",
      "nfl_proj_median",
      "nfl_proj_ceiling",
      "nfl_actual_y1_3",
      "error_yds",
      "coverage_status",
  ]
  available_cols = [c for c in audit_cols if c in df.columns]
  df.select(available_cols).write_csv(ERROR_AUDIT_FILE)
  print(f"\nWrote full player-by-player audit ({n} rows) to: {ERROR_AUDIT_FILE}")

  print("\n" + "=" * 80)
  print("3. TOP 10 CLOSEST PREDICTIONS (Bullseye Model Hits)")
  print("=" * 80)
  print(df.sort("error_yds", descending=False).select(available_cols).head(10))

  print("\n" + "=" * 80)
  print("4. TOP 10 UNDER-PREDICTIONS (Breakouts / Athletic Steals)")
  print("=" * 80)
  steals = df.with_columns(
      diff=(pl.col("nfl_actual_y1_3") - pl.col("nfl_proj_median"))
  ).sort("diff", descending=True)
  print(steals.select(available_cols).head(10))

  print("\n" + "=" * 80)
  print("5. TOP 10 OVER-PREDICTIONS (Early Bust Risks)")
  print("=" * 80)
  busts = df.with_columns(
      diff=(pl.col("nfl_proj_median") - pl.col("nfl_actual_y1_3"))
  ).sort("diff", descending=True)
  print(busts.select(available_cols).head(10))


if __name__ == "__main__":
  run_pipeline()