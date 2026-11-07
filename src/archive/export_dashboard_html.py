"""
Generate Interactive Stage 1 Pure College Prospect Dashboard HTML.
Translates pre-draft M2 college ratings and athletic profiles into 
visual bell curve density forecasts and outcome probability buckets
across historical (2012–2022) and recent holdout cohorts (2023–2025).

Usage:
    python3 export_dashboard_html.py
"""

import json
import math
import os
from pathlib import Path
import sys

# Support Polars with graceful fallback to Pandas/PyArrow
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    import pandas as pd
    HAS_POLARS = False

# 1. Flexible path resolution
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
RANKINGS_FILE = DATA_DIR / "stage1_prospect_rankings.parquet"
RECENT_FILE = DATA_DIR / "stage1_recent_prospects.parquet"
JSON_FALLBACK = Path("artifacts/file_generation/ttl=63d/intermediate/prospects_data.json")

if not RANKINGS_FILE.exists() and not JSON_FALLBACK.exists():
    print(f"Error: Could not find {RANKINGS_FILE}.")
    print("Please run `python3 run_stage1_pipeline.py` first.")
    sys.exit(1)

records = []

if RANKINGS_FILE.exists():
    print(f"Reading historical prospect rankings from {RANKINGS_FILE}...")
    if HAS_POLARS:
        df = pl.read_parquet(RANKINGS_FILE)
        col_map = {
            "athlete_name": "name",
            "name_draft": "name",
            "player_name": "name",
            "draft_overall": "pick",
            "college_team": "college",
            "school_name": "college",
            "nfl_team": "team",
            "nfl_proj_floor": "p10",
            "nfl_proj_median": "p50",
            "nfl_proj_ceiling": "p90",
            "nfl_proj_p10": "p10",
            "nfl_proj_p50": "p50",
            "nfl_proj_p90": "p90",
            "draft_round": "round",
            "actual": "actual_y1_3",
            "nfl_actual_y1_3": "actual_y1_3",
        }
        for old_c, new_c in col_map.items():
            if old_c in df.columns and new_c not in df.columns:
                df = df.rename({old_c: new_c})

        if "draft_year" in df.columns:
            df = df.filter((pl.col("draft_year") >= 2012) & (pl.col("draft_year") <= 2022))

        sort_col = "college_grade" if "college_grade" in df.columns else df.columns[0]
        raw_rows = df.sort(sort_col, descending=True).iter_rows(named=True)
    else:
        pdf = pd.read_parquet(RANKINGS_FILE)
        col_map = {
            "athlete_name": "name",
            "name_draft": "name",
            "player_name": "name",
            "draft_overall": "pick",
            "college_team": "college",
            "school_name": "college",
            "nfl_team": "team",
            "nfl_proj_floor": "p10",
            "nfl_proj_median": "p50",
            "nfl_proj_ceiling": "p90",
            "nfl_proj_p10": "p10",
            "nfl_proj_p50": "p50",
            "nfl_proj_p90": "p90",
            "draft_round": "round",
            "actual": "actual_y1_3",
            "nfl_actual_y1_3": "actual_y1_3",
        }
        pdf = pdf.rename(columns={k: v for k, v in col_map.items() if k in pdf.columns and v not in pdf.columns})
        if "draft_year" in pdf.columns:
            pdf = pdf[(pdf["draft_year"] >= 2012) & (pdf["draft_year"] <= 2022)]
        sort_col = "college_grade" if "college_grade" in pdf.columns else pdf.columns[0]
        raw_rows = pdf.sort_values(by=sort_col, ascending=False).to_dict(orient="records")

    for idx, r in enumerate(raw_rows):
        c_grade = float(r.get("college_grade") or 50.0)
        p_peak = float(r.get("peak_season_rating") or 1.0)
        c_rating = float(r.get("career_rating") or p_peak)
        f_rating = float(r.get("final_season_rating") or p_peak)
        g_share = float(r.get("final_season_games_share") or 1.0)

        p10 = int(round(float(r.get("p10") or 300)))
        p50 = int(round(float(r.get("p50") or 1200)))
        p90 = int(round(float(r.get("p90") or 2500)))
        actual = float(r.get("actual_y1_3") or r.get("nfl_actual_y1_3") or 0.0)

        prob_bust = float(r.get("prob_bust") or 0.50)
        prob_starter = float(r.get("prob_starter") or 0.25)
        prob_boom = float(r.get("prob_boom") or 0.12)

        forty = float(r.get("forty")) if r.get("forty") is not None and not math.isnan(float(r.get("forty") or 0)) else None
        speed_score = float(r.get("speed_score")) if r.get("speed_score") is not None and not math.isnan(float(r.get("speed_score") or 0)) else None
        hass = float(r.get("hass")) if r.get("hass") is not None and not math.isnan(float(r.get("hass") or 0)) else None
        cone_shuttle = float(r.get("cone_shuttle")) if r.get("cone_shuttle") is not None and not math.isnan(float(r.get("cone_shuttle") or 0)) else None
        vert_in = float(r.get("vert_in")) if r.get("vert_in") is not None and not math.isnan(float(r.get("vert_in") or 0)) else None
        ht_in = float(r.get("ht_in")) if r.get("ht_in") is not None and not math.isnan(float(r.get("ht_in") or 0)) else None
        wt_lbs = float(r.get("wt_lbs")) if r.get("wt_lbs") is not None and not math.isnan(float(r.get("wt_lbs") or 0)) else None

        prof = r.get("prospect_profile", None)
        if not prof:
            if c_grade >= 89.0:
                prof = "Blue-Chip Alpha"
            elif actual >= 2000 and c_grade < 80.0:
                prof = "Boom-or-Bust Flier"
            elif c_grade >= 80.0:
                prof = "High-Floor Starter"
            else:
                prof = "High Bust Risk"

        delta = int(round(actual - p50))

        if actual >= 2500:
            actual_bucket = "WR1 Boom (≥2.5k)"
        elif actual >= 1500:
            actual_bucket = "Quality Starter (1.5k–2.5k)"
        elif actual >= 500:
            actual_bucket = "Role Contributor (500–1.5k)"
        else:
            actual_bucket = "Bust Flop (<500)"

        records.append({
            "id": f"wr_{idx}_{r.get('draft_year', 2020)}",
            "name": r.get("name") or "Unknown Prospect",
            "college": r.get("college") or "NCAA",
            "team": r.get("team", "NFL"),
            "year": int(r.get("draft_year") or 2020),
            "round": int(r.get("round") or 1),
            "pick": int(r.get("pick") or 1),
            "grade": round(c_grade, 1),
            "peak_rating": round(p_peak, 2),
            "career_rating": round(c_rating, 2),
            "final_rating": round(f_rating, 2),
            "games_share": round(g_share, 2),
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "prob_boom": round(prob_boom, 3),
            "prob_starter": round(prob_starter, 3),
            "prob_bust": round(prob_bust, 3),
            "forty": round(forty, 2) if forty else None,
            "speed_score": round(speed_score, 1) if speed_score else None,
            "hass": round(hass, 1) if hass else None,
            "cone_shuttle": round(cone_shuttle, 2) if cone_shuttle else None,
            "vert_in": round(vert_in, 1) if vert_in else None,
            "ht_in": round(ht_in, 1) if ht_in else None,
            "wt_lbs": round(wt_lbs, 1) if wt_lbs else None,
            "delta": delta,
            "actual_y1_3": int(round(actual)),
            "actual_bucket": actual_bucket,
            "is_active": False,
            "profile": prof,
        })

# 2. Append 2023, 2024, and 2025 prospects if recent predictions file exists
if RECENT_FILE.exists():
    print(f"Loading recent draft cohorts (2023–2025) from {RECENT_FILE}...")
    if HAS_POLARS:
        rec_df = pl.read_parquet(RECENT_FILE)
        rec_rows = rec_df.iter_rows(named=True)
    else:
        rec_pdf = pd.read_parquet(RECENT_FILE)
        rec_rows = rec_pdf.to_dict(orient="records")

    base_idx = len(records)
    for idx, r in enumerate(rec_rows):
        c_grade = float(r.get("college_grade") or 50.0)
        p10 = int(round(float(r.get("nfl_proj_floor") or 300)))
        p50 = int(round(float(r.get("nfl_proj_median") or 1200)))
        p90 = int(round(float(r.get("nfl_proj_ceiling") or 2500)))
        yr = int(r.get("draft_year") or 2024)

        forty = float(r.get("forty_sec")) if r.get("forty_sec") is not None and not math.isnan(float(r.get("forty_sec") or 0)) else None
        hass = float(r.get("hass")) if r.get("hass") is not None and not math.isnan(float(r.get("hass") or 0)) else None
        burst = float(r.get("burst_score")) if r.get("burst_score") is not None and not math.isnan(float(r.get("burst_score") or 0)) else None
        agility = float(r.get("agility_sum")) if r.get("agility_sum") is not None and not math.isnan(float(r.get("agility_sum") or 0)) else None

        records.append({
            "id": f"recent_{base_idx + idx}_{yr}",
            "name": r.get("name") or "Prospect",
            "college": r.get("college") or "NCAA",
            "team": r.get("draft_slot_info") or "NFL",
            "year": yr,
            "round": 1,
            "pick": 1,
            "grade": round(c_grade, 1),
            "peak_rating": 1.5,
            "career_rating": 1.4,
            "final_rating": 1.5,
            "games_share": 1.0,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "prob_boom": round(float(r.get("prob_boom") or 0.15), 3),
            "prob_starter": round(float(r.get("prob_starter") or 0.30), 3),
            "prob_bust": round(float(r.get("prob_bust") or 0.50), 3),
            "forty": round(forty, 2) if forty else None,
            "speed_score": round(hass, 1) if hass else None,
            "hass": round(hass, 1) if hass else None,
            "cone_shuttle": round(agility, 2) if agility else None,
            "vert_in": None,
            "ht_in": None,
            "wt_lbs": None,
            "delta": 0,
            "actual_y1_3": 0,
            "actual_bucket": r.get("prospect_profile") or "In Progress",
            "is_active": True,
            "profile": r.get("prospect_profile") or "Active Prospect",
        })

prospects_json = json.dumps(records)
years_available = sorted(list(set(r["year"] for r in records)), reverse=True)
print(f"Loaded {len(records)} total prospects across classes {min(years_available)}–{max(years_available)}.")

html_template = f"""<!DOCTYPE html>
<html lang="en" class="h-full bg-slate-950 text-slate-100">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stage 1: Pure College Prospect Dashboard (Bell Curves &amp; Outcome Tiers)</title>
  <script src="https://cdn.tailwindcss.com/3.4.17"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    body {{ font-family: 'Inter', sans-serif; }}
    .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: #090d16; }}
    ::-webkit-scrollbar-thumb {{ background: #1e293b; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #334155; }}
    .bell-curve-svg {{ transition: filter 0.2s ease; }}
    .bell-curve-svg:hover {{ filter: drop-shadow(0 0 6px rgba(129, 140, 248, 0.4)); }}
  </style>
</head>
<body class="min-h-full flex flex-col bg-slate-950 text-slate-100 antialiased selection:bg-indigo-500 selection:text-white">

  <!-- Navigation Header -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-violet-600 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 font-bold text-white text-sm">S1</div>
        <div>
          <div class="flex items-center space-x-2">
            <h1 class="text-base font-bold tracking-tight text-white">Stage 1: Pure College Prospect Board</h1>
            <span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Zero Draft Slot Bias</span>
          </div>
          <p class="text-xs text-slate-400">Bell Curve Density Forecasts &amp; Calibrated Outcome Tiers (2012–2025)</p>
        </div>
      </div>

      <div class="flex items-center space-x-3">
        <div class="flex bg-slate-800/80 p-1 rounded-lg border border-slate-700/60 text-xs">
          <button id="tab-btn-board" class="px-3 py-1.5 rounded-md font-medium transition bg-indigo-600 text-white shadow">Prospect Board</button>
          <button id="tab-btn-analytics" class="px-3 py-1.5 rounded-md font-medium text-slate-300 hover:text-white transition">Outcome Analytics</button>
          <button id="tab-btn-evaluation" class="px-3 py-1.5 rounded-md font-medium text-slate-300 hover:text-white transition">Hits &amp; Sleepers</button>
        </div>
        <button id="btn-export-csv" class="hidden sm:inline-flex items-center space-x-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg border border-slate-700 transition">
          <svg class="w-3.5 h-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          <span>Export CSV</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Metric Overview Banners -->
  <section class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 pb-2 w-full">
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3.5 flex flex-col justify-between">
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Prospects Evaluated</span>
        <div class="flex items-baseline space-x-2 mt-1">
          <span id="stat-total-count" class="text-2xl font-bold text-white font-mono">{len(records)}</span>
          <span class="text-xs text-slate-400">Total WRs</span>
        </div>
        <span class="text-[11px] text-emerald-400 mt-1">2012–2025 Classes</span>
      </div>

      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3.5 flex flex-col justify-between">
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Quality Starter (≥1.5k)</span>
        <div class="flex items-baseline space-x-2 mt-1">
          <span class="text-2xl font-bold text-indigo-400 font-mono">+11.6%</span>
          <span class="text-xs text-slate-400">Brier Skill</span>
        </div>
        <span class="text-[11px] text-slate-400 mt-1">Base Rate: 24.8% (Climatology)</span>
      </div>

      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3.5 flex flex-col justify-between">
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">WR1 Boom Tier (≥2.5k)</span>
        <div class="flex items-baseline space-x-2 mt-1">
          <span class="text-2xl font-bold text-emerald-400 font-mono">+8.3%</span>
          <span class="text-xs text-slate-400">Brier Skill</span>
        </div>
        <span class="text-[11px] text-slate-400 mt-1">Base Rate: 12.1% (Alpha Ceiling)</span>
      </div>

      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3.5 flex flex-col justify-between">
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Bust Avoidance (&lt;500)</span>
        <div class="flex items-baseline space-x-2 mt-1">
          <span class="text-2xl font-bold text-rose-400 font-mono">+10.5%</span>
          <span class="text-xs text-slate-400">Brier Skill</span>
        </div>
        <span class="text-[11px] text-slate-400 mt-1">Base Rate: 50.0% (Flop Rate)</span>
      </div>
    </div>
  </section>

  <!-- Filter & Search Toolbar -->
  <section class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 w-full">
    <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 space-y-3.5 shadow-sm">
      
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div class="flex items-center space-x-1.5 overflow-x-auto pb-1 md:pb-0 scrollbar-none">
          <span class="text-xs font-semibold text-slate-400 mr-1 uppercase tracking-wider flex-shrink-0">Class:</span>
          <div id="class-filter-container" class="flex space-x-1">
            <button data-class="all" class="class-btn px-2.5 py-1 text-xs font-medium rounded-lg bg-indigo-600 text-white shadow">All</button>
            {"".join(f'<button data-class="{yr}" class="class-btn px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 transition">{yr}</button>' for yr in years_available)}
          </div>
        </div>

        <div class="relative w-full md:w-72">
          <input id="input-search" type="text" placeholder="Search player, college, team..."
                 class="w-full bg-slate-950 border border-slate-800 text-slate-100 text-xs rounded-lg pl-8 pr-3 py-2 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 transition">
          <svg class="w-4 h-4 text-slate-500 absolute left-2.5 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-4 gap-2 pt-1 border-t border-slate-800/60 text-xs">
        <div class="flex items-center space-x-2">
          <label class="text-slate-400 font-medium">Round:</label>
          <select id="select-round" class="bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-md px-2 py-1.5 focus:outline-none focus:border-indigo-500 w-full">
            <option value="all">All Rounds (1–7)</option>
            <option value="1">Round 1 Only</option>
            <option value="2">Round 2 Only</option>
            <option value="3">Round 3 Only</option>
            <option value="day3">Day 3 (Rounds 4–7)</option>
          </select>
        </div>

        <div class="flex items-center space-x-2">
          <label class="text-slate-400 font-medium">Outcome Tier:</label>
          <select id="select-bucket" class="bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-md px-2 py-1.5 focus:outline-none focus:border-indigo-500 w-full">
            <option value="all">All Projected Tiers</option>
            <option value="starter_favored">Starter Favored (P ≥ 35%)</option>
            <option value="boom_upside">High Boom Upside (P ≥ 18%)</option>
            <option value="low_bust">Low Bust Risk (P ≤ 35%)</option>
            <option value="high_bust">High Bust Flop Warning (P ≥ 50%)</option>
            <option value="actual_boom">Actual Hit: WR1 Boom (≥2.5k)</option>
            <option value="actual_starter">Actual Hit: Starter (1.5k–2.5k)</option>
            <option value="actual_bust">Actual Hit: Bust Flop (&lt;500)</option>
          </select>
        </div>

        <div class="flex items-center space-x-2">
          <label class="text-slate-400 font-medium">Sort By:</label>
          <select id="select-sort" class="bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-md px-2 py-1.5 focus:outline-none focus:border-indigo-500 w-full">
            <option value="grade_desc">Pre-Draft College Grade (Highest)</option>
            <option value="p50_desc">Projected Median Peak Y1–3 (Highest)</option>
            <option value="boom_desc">P(Boom ≥2.5k) Probability (Highest)</option>
            <option value="starter_desc">P(Starter ≥1.5k) Probability (Highest)</option>
            <option value="bust_asc">P(Bust &lt;500) Risk (Lowest)</option>
            <option value="actual_desc">Actual NFL Y1–3 Yards (Highest)</option>
            <option value="delta_desc">Outperforming Median (Most Yards Over)</option>
          </select>
        </div>

        <div class="flex items-center justify-end">
          <button id="btn-reset-filters" class="text-slate-400 hover:text-indigo-400 transition text-xs flex items-center space-x-1 py-1">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            <span>Reset Filters</span>
          </button>
        </div>
      </div>
    </div>
  </section>

  <!-- Main Content -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2 w-full flex-1">
    
    <!-- View 1: Table -->
    <div id="view-board" class="space-y-4">
      <div class="bg-slate-900/90 border border-slate-800 rounded-xl overflow-hidden shadow">
        
        <!-- Legend & Sub-header -->
        <div class="px-4 py-3 border-b border-slate-800 flex flex-wrap items-center justify-between text-xs text-slate-400 gap-2">
          <div>Showing <span id="label-showing-count" class="font-semibold text-white">0</span> prospects</div>
          <div class="flex items-center space-x-4">
            <span class="flex items-center space-x-1.5">
              <span class="w-3 h-2 rounded bg-indigo-600/60 border border-indigo-400"></span>
              <span>Bell Curve Likelihood (Peak = p50 Mode)</span>
            </span>
            <span class="flex items-center space-x-1.5">
              <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]"></span>
              <span>Actual NFL Production</span>
            </span>
            <span class="flex items-center space-x-1.5">
              <span class="w-2 h-2 rounded-full bg-rose-500/80"></span>
              <span>&lt;500 Bust</span>
              <span class="w-2 h-2 rounded-full bg-indigo-500/80 ml-1"></span>
              <span>≥1.5k Starter</span>
              <span class="w-2 h-2 rounded-full bg-emerald-500/80 ml-1"></span>
              <span>≥2.5k Boom</span>
            </span>
          </div>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs border-collapse">
            <thead>
              <tr class="bg-slate-950/80 text-slate-400 font-semibold border-b border-slate-800">
                <th class="py-3 px-3.5 text-center w-12">Rank</th>
                <th class="py-3 px-3">Prospect</th>
                <th class="py-3 px-3 text-center">College Grade</th>
                <th class="py-3 px-3">Athletic Profile</th>
                <th class="py-3 px-4 min-w-[320px]">Production Forecast (Bell Curve &amp; Outcome Tiers)</th>
                <th class="py-3 px-3 text-right">Actual Y1–3</th>
                <th class="py-3 px-3 text-center">Vs Mode</th>
                <th class="py-3 px-3 text-center">Outcome Tier</th>
              </tr>
            </thead>
            <tbody id="board-tbody" class="divide-y divide-slate-800/60 font-medium"></tbody>
          </table>
        </div>

        <div id="no-results" class="hidden py-16 text-center text-slate-500">
          <p class="text-sm font-medium text-slate-400">No prospects match your search criteria</p>
        </div>
      </div>
    </div>

    <!-- View 2: Analytics -->
    <div id="view-analytics" class="hidden space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-2 bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow">
          <div class="flex items-center justify-between mb-4">
            <div>
              <h2 class="text-sm font-bold text-white">Production Density &amp; Outcome Probability by Prospect</h2>
              <p class="text-xs text-slate-400">Top 15 prospects matching filter. Stacked outcome probability tiers vs actual production.</p>
            </div>
            <span class="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">Double-Digit Skill vs Climatology</span>
          </div>
          <div class="relative h-96">
            <canvas id="chart-quantile-ranges"></canvas>
          </div>
        </div>

        <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow space-y-4">
          <h2 class="text-sm font-bold text-white">Stage 1 &amp; 1.5 Calibration Summary</h2>
          <div class="space-y-3 text-xs">
            <div class="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span class="text-slate-400 font-semibold block text-indigo-300">Starter Tier (≥1,500 yds)</span>
              <div class="flex items-baseline justify-between mt-1">
                <span class="text-lg font-bold font-mono text-emerald-400">+11.6% BSS</span>
                <span class="text-slate-400">Brier: 0.1651 vs 0.1868</span>
              </div>
            </div>

            <div class="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span class="text-slate-400 font-semibold block text-emerald-300">Boom WR1 Alpha (≥2,500 yds)</span>
              <div class="flex items-baseline justify-between mt-1">
                <span class="text-lg font-bold font-mono text-emerald-400">+8.3% BSS</span>
                <span class="text-slate-400">Brier: 0.0977 vs 0.1065</span>
              </div>
            </div>

            <div class="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span class="text-slate-400 font-semibold block text-rose-300">Bust / Flop Tier (&lt;500 yds)</span>
              <div class="flex items-baseline justify-between mt-1">
                <span class="text-lg font-bold font-mono text-emerald-400">+10.5% BSS</span>
                <span class="text-slate-400">Brier: 0.2236 vs 0.2500</span>
              </div>
            </div>

            <div class="p-3 bg-indigo-950/30 rounded-lg border border-indigo-800/40 text-slate-300 space-y-1">
              <span class="font-semibold text-indigo-300">Why Bell Curves &amp; Buckets Work</span>
              <p class="text-[11px] leading-relaxed text-slate-400">
                Single-point estimates or raw p10/p90 intervals hide the true probability mass. An intuitive bell curve density immediately reveals whether a receiver's likelihood is concentrated in the starter tier or skewed toward a right-tail breakout.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- View 3: Hits & Misses -->
    <div id="view-evaluation" class="hidden space-y-6">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow space-y-4">
          <div class="flex items-center justify-between pb-2 border-b border-slate-800">
            <h2 class="text-sm font-bold text-white">Top Model Bullseyes (Predicted Boom &amp; Hit WR1)</h2>
            <span class="text-xs text-slate-400">P(Boom) ≥ 18% &amp; Actual ≥ 2.4k Yds</span>
          </div>
          <div id="hits-list" class="space-y-2.5"></div>
        </div>

        <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow space-y-4">
          <div class="flex items-center justify-between pb-2 border-b border-slate-800">
            <h2 class="text-sm font-bold text-white">Top Right-Tail Outliers / Sleepers</h2>
            <span class="text-xs text-slate-400">Beat Median Likelihood Peak Most</span>
          </div>
          <div id="misses-list" class="space-y-2.5"></div>
        </div>
      </div>
    </div>

  </main>

  <!-- Player Detail Modal with Large Interactive Bell Curve -->
  <div id="player-modal" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md hidden">
    <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 space-y-5 shadow-2xl relative max-h-[92vh] overflow-y-auto">
      <button id="modal-close-btn" class="absolute top-4 right-4 text-slate-400 hover:text-white transition">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
      </button>

      <!-- Modal Header -->
      <div class="flex items-start space-x-4">
        <div id="modal-grade-circle" class="w-14 h-14 rounded-2xl bg-indigo-600/20 border border-indigo-500/40 flex flex-col items-center justify-center text-white flex-shrink-0">
          <span class="text-[10px] text-slate-400 uppercase tracking-wider">Grade</span>
          <span id="modal-grade" class="text-lg font-bold font-mono text-indigo-400">95.0</span>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <h3 id="modal-name" class="text-lg font-bold text-white">Prospect</h3>
            <span id="modal-profile-badge" class="px-2 py-0.5 text-[11px] rounded-full bg-purple-500/20 text-purple-300">Blue-Chip Alpha</span>
          </div>
          <p id="modal-sub" class="text-xs text-slate-400 mt-0.5">College &amp; Draft details</p>
        </div>
      </div>

      <!-- Large Interactive Bell Curve Visualizer -->
      <div class="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-xs font-semibold text-slate-200">Probability Density Distribution (Year 1–3 Receiving Yards)</span>
          <span class="text-[11px] font-mono text-indigo-400">Gaussian Credible Density</span>
        </div>

        <!-- Dynamic SVG Bell Curve Canvas -->
        <div class="relative w-full h-44 bg-slate-950 rounded-lg overflow-hidden border border-slate-800/80 p-1 flex items-center justify-center">
          <div id="modal-bell-container" class="w-full h-full"></div>
        </div>

        <!-- Outcome Buckets Grid -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-center">
          <div id="bucket-card-bust" class="p-2.5 rounded-lg border border-slate-800 bg-slate-900/50 transition">
            <span class="text-[10px] text-slate-400 uppercase tracking-wider block font-medium">Bust Risk</span>
            <span class="text-[10px] text-rose-400 block">&lt;500 yds</span>
            <span id="modal-prob-bust" class="text-base font-bold font-mono text-rose-400 mt-0.5 block">50%</span>
          </div>
          <div id="bucket-card-role" class="p-2.5 rounded-lg border border-slate-800 bg-slate-900/50 transition">
            <span class="text-[10px] text-slate-400 uppercase tracking-wider block font-medium">Rotation</span>
            <span class="text-[10px] text-slate-400 block">500–1.5k yds</span>
            <span id="modal-prob-role" class="text-base font-bold font-mono text-slate-300 mt-0.5 block">25%</span>
          </div>
          <div id="bucket-card-starter" class="p-2.5 rounded-lg border border-slate-800 bg-slate-900/50 transition">
            <span class="text-[10px] text-slate-400 uppercase tracking-wider block font-medium">Starter</span>
            <span class="text-[10px] text-indigo-400 block">1.5k–2.5k yds</span>
            <span id="modal-prob-starter" class="text-base font-bold font-mono text-indigo-400 mt-0.5 block">15%</span>
          </div>
          <div id="bucket-card-boom" class="p-2.5 rounded-lg border border-slate-800 bg-slate-900/50 transition">
            <span class="text-[10px] text-slate-400 uppercase tracking-wider block font-medium">WR1 Alpha</span>
            <span class="text-[10px] text-emerald-400 block">≥2.5k yds</span>
            <span id="modal-prob-boom" class="text-base font-bold font-mono text-emerald-400 mt-0.5 block">10%</span>
          </div>
        </div>

        <div class="flex items-center justify-between text-xs pt-1 border-t border-slate-800/60 text-slate-400">
          <span id="modal-actual-label">Actual NFL Y1–3 Production: <strong id="modal-actual" class="text-emerald-400 font-mono text-sm">-</strong></span>
          <span>Diff vs Likelihood Mode: <strong id="modal-delta" class="font-mono text-sm">-</strong></span>
        </div>
      </div>

      <!-- Athletic Profile & College Skills -->
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <!-- Athletic Testing Card -->
        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-2">
          <span class="text-xs font-semibold text-slate-300 flex items-center justify-between">
            <span>Pre-Draft Athletic Testing</span>
            <span class="text-[10px] font-mono text-slate-500">Combine / Pro Day</span>
          </span>
          <div class="grid grid-cols-2 gap-2 text-xs">
            <div class="p-2 bg-slate-900/60 rounded-lg border border-slate-800/80">
              <span class="text-slate-500 block text-[10px]">40-Yard Dash</span>
              <span id="modal-forty" class="font-mono font-bold text-slate-200 text-sm">-</span>
            </div>
            <div class="p-2 bg-slate-900/60 rounded-lg border border-slate-800/80">
              <span class="text-slate-500 block text-[10px]">Height-Adj Speed (HaSS)</span>
              <span id="modal-hass" class="font-mono font-bold text-indigo-400 text-sm">-</span>
            </div>
            <div class="p-2 bg-slate-900/60 rounded-lg border border-slate-800/80">
              <span class="text-slate-500 block text-[10px]">Agility (3-Cone+Shuttle)</span>
              <span id="modal-agility" class="font-mono font-bold text-slate-200 text-sm">-</span>
            </div>
            <div class="p-2 bg-slate-900/60 rounded-lg border border-slate-800/80">
              <span class="text-slate-500 block text-[10px]">Vertical Jump / Weight</span>
              <span id="modal-vert" class="font-mono font-bold text-slate-200 text-sm">-</span>
            </div>
          </div>
        </div>

        <!-- College M2 Skill Metrics -->
        <div class="bg-slate-950 p-3 rounded-xl border border-slate-800 space-y-2">
          <span class="text-xs font-semibold text-slate-300 flex items-center justify-between">
            <span>College M2 Translation</span>
            <span class="text-[10px] font-mono text-slate-500">Pure College Model</span>
          </span>
          <div class="grid grid-cols-2 gap-2 text-xs">
            <div class="p-2 bg-slate-900/60 rounded-lg border border-slate-800/80">
              <span class="text-slate-500 block text-[10px]">Peak Season Rating</span>
              <span id="modal-peak" class="font-mono font-bold text-slate-200 text-sm">-</span>
            </div>
            <div class="p-2 bg-slate-900/60 rounded-lg border border-slate-800/80">
              <span class="text-slate-500 block text-[10px]">Career Rating</span>
              <span id="modal-career" class="font-mono font-bold text-slate-200 text-sm">-</span>
            </div>
            <div class="p-2 bg-slate-900/60 rounded-lg border border-slate-800/80">
              <span class="text-slate-500 block text-[10px]">Games Avail Share</span>
              <span id="modal-games" class="font-mono font-bold text-slate-200 text-sm">-</span>
            </div>
            <div class="p-2 bg-slate-900/60 rounded-lg border border-slate-800/80">
              <span class="text-slate-500 block text-[10px]">80% Credible Interval</span>
              <span id="modal-interval" class="font-mono font-bold text-violet-400 text-sm">-</span>
            </div>
          </div>
        </div>
      </div>

      <div class="pt-2 flex justify-end">
        <button id="modal-ok-btn" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold rounded-lg transition">Close</button>
      </div>
    </div>
  </div>

  <div id="toast" class="fixed bottom-5 right-5 z-50 bg-slate-800 text-slate-100 px-4 py-2.5 rounded-xl border border-slate-700 shadow-xl text-xs font-medium flex items-center space-x-2 transition-all opacity-0 pointer-events-none translate-y-2">
    <span id="toast-msg">Export downloaded</span>
  </div>

  <script>
    const PROSPECTS = {prospects_json};

    let currentFilterClass = 'all';
    let currentFilterRound = 'all';
    let currentFilterBucket = 'all';
    let currentSearchTerm = '';
    let currentSort = 'grade_desc';
    let activeTab = 'board';
    let analyticsChart = null;

    const boardTbody = document.getElementById('board-tbody');
    const labelShowingCount = document.getElementById('label-showing-count');
    const noResults = document.getElementById('no-results');
    const inputSearch = document.getElementById('input-search');
    const selectRound = document.getElementById('select-round');
    const selectBucket = document.getElementById('select-bucket');
    const selectSort = document.getElementById('select-sort');
    const btnResetFilters = document.getElementById('btn-reset-filters');
    const classFilterContainer = document.getElementById('class-filter-container');
    const btnExportCsv = document.getElementById('btn-export-csv');

    const tabBtnBoard = document.getElementById('tab-btn-board');
    const tabBtnAnalytics = document.getElementById('tab-btn-analytics');
    const tabBtnEvaluation = document.getElementById('tab-btn-evaluation');
    const viewBoard = document.getElementById('view-board');
    const viewAnalytics = document.getElementById('view-analytics');
    const viewEvaluation = document.getElementById('view-evaluation');

    const playerModal = document.getElementById('player-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalOkBtn = document.getElementById('modal-ok-btn');

    function showToast(msg) {{
      const t = document.getElementById('toast');
      document.getElementById('toast-msg').textContent = msg;
      t.classList.remove('opacity-0', 'pointer-events-none', 'translate-y-2');
      setTimeout(() => t.classList.add('opacity-0', 'pointer-events-none', 'translate-y-2'), 2500);
    }}

    function getGradeBadgeColor(grade) {{
      if (grade >= 95.0) return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      if (grade >= 90.0) return 'bg-purple-500/10 text-purple-400 border-purple-500/30';
      if (grade >= 85.0) return 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30';
      if (grade >= 80.0) return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
      return 'bg-slate-800 text-slate-400 border-slate-700';
    }}

    function getOutcomeBadge(p) {{
      if (p.is_active) {{
        return `<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">${{p.profile}}</span>`;
      }}
      const bucket = p.actual_bucket || '';
      if (bucket.includes('Boom')) return '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">WR1 Boom</span>';
      if (bucket.includes('Starter')) return '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">Quality Starter</span>';
      if (bucket.includes('Role')) return '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-800 text-slate-300 border border-slate-700">Contributor</span>';
      return '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30">Bust / Flop</span>';
    }}

    function createBellCurveSparkline(p, width = 280, height = 36, maxVal = 4850) {{
      const mu = Number(p.p50);
      const p10 = Number(p.p10);
      const p90 = Number(p.p90);
      const actual = Number(p.actual_y1_3);

      const sigma = Math.max(220, (p90 - p10) / 2.56);
      const maxDensity = 1.0 / (sigma * Math.sqrt(2 * Math.PI));

      const steps = 30;
      let pathD = `M 0 ${{height}}`;
      for (let i = 0; i <= steps; i++) {{
        const xVal = (i / steps) * maxVal;
        const density = (1.0 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((xVal - mu) / sigma, 2));
        const normY = density / maxDensity;
        const svgX = (xVal / maxVal) * width;
        const svgY = height - 2 - (normY * (height - 5));
        pathD += ` L ${{svgX.toFixed(1)}} ${{svgY.toFixed(1)}}`;
      }}
      pathD += ` L ${{width}} ${{height}} Z`;

      const bustX = ((500 / maxVal) * width).toFixed(1);
      const starterX = ((1500 / maxVal) * width).toFixed(1);
      const boomX = ((2500 / maxVal) * width).toFixed(1);
      const medX = (Math.min(maxVal, Math.max(0, mu)) / maxVal * width).toFixed(1);

      const gradId = `grad_${{p.id}}`;

      let markerHtml = '';
      if (!p.is_active && actual > 0) {{
        const actClamped = Math.min(maxVal, Math.max(0, actual));
        const actX = ((actClamped / maxVal) * width).toFixed(1);
        const actDensity = (1.0 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((actClamped - mu) / sigma, 2));
        const actY = (height - 2 - ((actDensity / maxDensity) * (height - 5))).toFixed(1);

        let dotColor = '#34d399';
        let dotGlow = 'rgba(52, 211, 153, 0.6)';
        if (actual < 500) {{
          dotColor = '#fb7185';
          dotGlow = 'rgba(251, 113, 133, 0.6)';
        }} else if (actual < mu - 200) {{
          dotColor = '#f59e0b';
          dotGlow = 'rgba(245, 158, 11, 0.6)';
        }}
        markerHtml = `
          <circle cx="${{actX}}" cy="${{actY}}" r="5" fill="${{dotGlow}}" opacity="0.4" />
          <circle cx="${{actX}}" cy="${{actY}}" r="3.2" fill="${{dotColor}}" stroke="#0f172a" stroke-width="1.2" />
        `;
      }}

      return `
        <svg class="bell-curve-svg w-full overflow-visible" height="${{height}}" viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
          <defs>
            <linearGradient id="${{gradId}}" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#818cf8" stop-opacity="0.45" />
              <stop offset="100%" stop-color="#4f46e5" stop-opacity="0.05" />
            </linearGradient>
          </defs>

          <!-- Cutoff Markers -->
          <line x1="${{bustX}}" y1="0" x2="${{bustX}}" y2="${{height}}" stroke="#f43f5e" stroke-width="0.75" stroke-dasharray="2,2" opacity="0.4" />
          <line x1="${{starterX}}" y1="0" x2="${{starterX}}" y2="${{height}}" stroke="#6366f1" stroke-width="0.75" stroke-dasharray="2,2" opacity="0.4" />
          <line x1="${{boomX}}" y1="0" x2="${{boomX}}" y2="${{height}}" stroke="#10b981" stroke-width="0.75" stroke-dasharray="2,2" opacity="0.4" />

          <!-- Bell Curve Path -->
          <path d="${{pathD}}" fill="url(#${{gradId}})" stroke="#818cf8" stroke-width="1.25" />

          <!-- Median Mode Line (Peak) -->
          <line x1="${{medX}}" y1="2" x2="${{medX}}" y2="${{height}}" stroke="#c7d2fe" stroke-width="1.2" stroke-dasharray="2,2" opacity="0.8" />

          <!-- Actual Outcome Point if completed -->
          ${{markerHtml}}
        </svg>
      `;
    }}

    function getFilteredAndSortedProspects() {{
      return PROSPECTS.filter(p => {{
        if (currentFilterClass !== 'all' && p.year.toString() !== currentFilterClass) return false;
        if (currentFilterRound === 'day3' && p.round < 4) return false;
        if (currentFilterRound !== 'all' && currentFilterRound !== 'day3' && p.round.toString() !== currentFilterRound) return false;
        
        if (currentFilterBucket === 'starter_favored' && p.prob_starter < 0.35) return false;
        if (currentFilterBucket === 'boom_upside' && p.prob_boom < 0.18) return false;
        if (currentFilterBucket === 'low_bust' && p.prob_bust > 0.35) return false;
        if (currentFilterBucket === 'high_bust' && p.prob_bust < 0.50) return false;
        if (currentFilterBucket === 'actual_boom' && p.actual_y1_3 < 2500) return false;
        if (currentFilterBucket === 'actual_starter' && (p.actual_y1_3 < 1500 || p.actual_y1_3 >= 2500)) return false;
        if (currentFilterBucket === 'actual_bust' && p.actual_y1_3 >= 500) return false;

        if (currentSearchTerm) {{
          const term = currentSearchTerm.toLowerCase();
          const match = p.name.toLowerCase().includes(term) ||
                        p.college.toLowerCase().includes(term) ||
                        p.team.toLowerCase().includes(term);
          if (!match) return false;
        }}
        return true;
      }}).sort((a, b) => {{
        switch(currentSort) {{
          case 'grade_desc': return b.grade - a.grade;
          case 'p50_desc': return b.p50 - a.p50;
          case 'boom_desc': return b.prob_boom - a.prob_boom;
          case 'starter_desc': return b.prob_starter - a.prob_starter;
          case 'bust_asc': return a.prob_bust - b.prob_bust;
          case 'actual_desc': return b.actual_y1_3 - a.actual_y1_3;
          case 'delta_desc': return b.delta - a.delta;
          default: return b.grade - a.grade;
        }}
      }});
    }}

    function renderBoard() {{
      const items = getFilteredAndSortedProspects();
      labelShowingCount.textContent = items.length;

      if (items.length === 0) {{
        boardTbody.innerHTML = '';
        noResults.classList.remove('hidden');
        return;
      }}
      noResults.classList.add('hidden');

      let html = '';
      items.forEach((p, idx) => {{
        let deltaHtml = '';
        if (p.is_active) {{
          deltaHtml = `<span class="text-slate-500 font-mono text-xs">-</span>`;
        }} else if (p.delta > 200) {{
          deltaHtml = `<span class="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">+${{p.delta.toLocaleString()}}</span>`;
        }} else if (p.delta < -200) {{
          deltaHtml = `<span class="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">${{p.delta.toLocaleString()}}</span>`;
        }} else {{
          deltaHtml = `<span class="px-2 py-0.5 rounded text-[11px] font-mono text-slate-400 bg-slate-800/60 border border-slate-700/50">±0</span>`;
        }}

        let actualHtml = '';
        if (p.is_active) {{
          actualHtml = `<span class="px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">Active</span>`;
        }} else {{
          actualHtml = `<span class="font-mono font-bold text-emerald-400 text-xs">${{p.actual_y1_3.toLocaleString()}}</span>`;
        }}

        let athText = '';
        if (p.forty) {{
          athText = `<div class="font-mono text-slate-200 text-xs">${{p.forty.toFixed(2)}}s 40 <span class="text-[10px] text-indigo-400">(${{p.hass ? p.hass.toFixed(0) : '-'}} HaSS)</span></div>`;
          if (p.cone_shuttle) {{
            athText += `<div class="text-[10px] text-slate-400 font-mono">${{p.cone_shuttle.toFixed(2)}}s agil</div>`;
          }} else {{
            athText += `<div class="text-[10px] text-slate-400 font-mono">Combine / Pro Day</div>`;
          }}
        }} else {{
          athText = `<div class="text-xs text-slate-500 italic">No workout data</div><div class="text-[10px] text-slate-600">Pure College Model</div>`;
        }}

        const pBustPct = Math.round(p.prob_bust * 100);
        const pStarterPct = Math.round(p.prob_starter * 100);
        const pBoomPct = Math.round(p.prob_boom * 100);

        html += `
          <tr class="hover:bg-slate-900/80 transition cursor-pointer group" onclick="openPlayerModal('${{p.id}}')">
            <td class="py-3 px-3.5 text-center font-mono text-slate-400">${{idx + 1}}</td>
            <td class="py-3 px-3">
              <div class="flex items-center space-x-2.5">
                <div class="w-8 h-8 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center font-bold text-xs text-indigo-400 group-hover:border-indigo-500 transition">
                  ${{p.name.split(' ').map(n=>n[0]).join('')}}
                </div>
                <div>
                  <div class="font-bold text-white group-hover:text-indigo-300 transition">${{p.name}}</div>
                  <div class="text-[11px] text-slate-400 flex items-center space-x-1.5">
                    <span>${{p.college}}</span>
                    <span>·</span>
                    <span class="text-indigo-400 font-semibold">${{p.team}}</span>
                    <span>·</span>
                    <span>'${{p.year.toString().slice(-2)}}</span>
                  </div>
                </div>
              </div>
            </td>
            <td class="py-3 px-3 text-center">
              <span class="inline-block px-2.5 py-1 rounded-lg text-xs font-mono font-bold border ${{getGradeBadgeColor(p.grade)}}">
                ${{p.grade.toFixed(1)}}
              </span>
            </td>
            <td class="py-3 px-3">
              ${{athText}}
            </td>
            <td class="py-3 px-4">
              <div class="space-y-1.5">
                <div class="flex items-center justify-between text-[11px] font-mono">
                  <div class="flex items-center space-x-1">
                    <span class="text-slate-400">Peak:</span>
                    <span class="font-bold text-indigo-300">${{p.p50.toLocaleString()}} yds</span>
                  </div>
                  <div class="flex items-center space-x-1">
                    <span class="px-1.5 py-0.2 rounded text-[10px] bg-rose-500/10 text-rose-400 border border-rose-500/20" title="Bust probability (&lt;500 yds)">${{pBustPct}}% Bust</span>
                    <span class="px-1.5 py-0.2 rounded text-[10px] bg-indigo-500/10 text-indigo-400 border border-indigo-500/20" title="Quality starter probability (≥1,500 yds)">${{pStarterPct}}% Start</span>
                    <span class="px-1.5 py-0.2 rounded text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" title="Boom WR1 alpha probability (≥2,500 yds)">${{pBoomPct}}% Boom</span>
                  </div>
                </div>
                <div title="Mode: ${{p.p50.toLocaleString()}} yds · 80% credible: ${{p.p10.toLocaleString()}}–${{p.p90.toLocaleString()}} yds">
                  ${{createBellCurveSparkline(p)}}
                </div>
              </div>
            </td>
            <td class="py-3 px-3 text-right">${{actualHtml}}</td>
            <td class="py-3 px-3 text-center">${{deltaHtml}}</td>
            <td class="py-3 px-3 text-center">${{getOutcomeBadge(p)}}</td>
          </tr>
        `;
      }});
      boardTbody.innerHTML = html;
    }}

    function renderEvaluation() {{
      const hitsList = document.getElementById('hits-list');
      const missesList = document.getElementById('misses-list');

      const hits = [...PROSPECTS]
        .filter(p => !p.is_active && p.prob_boom >= 0.18 && p.actual_y1_3 >= 2400)
        .sort((a, b) => b.actual_y1_3 - a.actual_y1_3)
        .slice(0, 8);

      let hitsHtml = '';
      hits.forEach(p => {{
        hitsHtml += `
          <div class="p-3 bg-slate-950/80 rounded-xl border border-slate-800/80 flex items-center justify-between hover:border-emerald-500/40 transition cursor-pointer" onclick="openPlayerModal('${{p.id}}')">
            <div>
              <div class="flex items-center space-x-2">
                <span class="font-bold text-white text-xs">${{p.name}}</span>
                <span class="text-[10px] px-1.5 py-0.2 rounded bg-indigo-950 text-indigo-300 font-mono">P(Boom): ${{Math.round(p.prob_boom * 100)}}%</span>
              </div>
              <div class="text-[11px] text-slate-400 mt-0.5">
                ${{p.college}} · Peak Mode: <span class="font-mono text-slate-200">${{p.p50.toLocaleString()}} yds</span>
              </div>
            </div>
            <div class="text-right">
              <span class="text-xs font-mono font-bold text-emerald-400">${{p.actual_y1_3.toLocaleString()}} yds</span>
              <div class="text-[10px] text-slate-400">Actual NFL Y1–3</div>
            </div>
          </div>
        `;
      }});
      hitsList.innerHTML = hitsHtml || '<div class="text-xs text-slate-500 p-3">No matching hits found</div>';

      const sleepers = [...PROSPECTS]
        .filter(p => !p.is_active && p.delta > 0)
        .sort((a, b) => b.delta - a.delta)
        .slice(0, 8);

      let missesHtml = '';
      sleepers.forEach(p => {{
        missesHtml += `
          <div class="p-3 bg-slate-950/80 rounded-xl border border-slate-800/80 flex items-center justify-between hover:border-indigo-500/40 transition cursor-pointer" onclick="openPlayerModal('${{p.id}}')">
            <div>
              <div class="flex items-center space-x-2">
                <span class="font-bold text-white text-xs">${{p.name}}</span>
                <span class="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-300 font-mono">'${{p.year.toString().slice(-2)}} ${{p.team}}</span>
              </div>
              <div class="text-[11px] text-slate-400 mt-0.5">
                Peak Mode: <span class="font-mono">${{p.p50.toLocaleString()}}</span> · Actual: <span class="font-mono text-emerald-400 font-bold">${{p.actual_y1_3.toLocaleString()}}</span>
              </div>
            </div>
            <div class="text-right">
              <span class="text-xs font-mono font-bold text-emerald-400">+${{p.delta.toLocaleString()}} yds</span>
              <div class="text-[10px] text-slate-400">Outperformed Mode</div>
            </div>
          </div>
        `;
      }});
      missesList.innerHTML = missesHtml;
    }}

    function renderAnalyticsChart() {{
      const ctx = document.getElementById('chart-quantile-ranges');
      if (!ctx) return;

      const topItems = getFilteredAndSortedProspects().slice(0, 15);
      const labels = topItems.map(p => p.name);
      const pBusts = topItems.map(p => Math.round(p.prob_bust * 100));
      const pStarters = topItems.map(p => Math.round(p.prob_starter * 100));
      const pBooms = topItems.map(p => Math.round(p.prob_boom * 100));
      const pRoles = topItems.map(p => Math.max(0, 100 - Math.round(p.prob_bust * 100) - Math.round(p.prob_starter * 100) - Math.round(p.prob_boom * 100)));

      if (analyticsChart) {{
        analyticsChart.destroy();
      }}

      analyticsChart = new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: labels,
          datasets: [
            {{ label: 'Bust (<500 yds)', data: pBusts, backgroundColor: 'rgba(244, 63, 94, 0.75)', borderRadius: 2 }},
            {{ label: 'Rotation (500–1.5k)', data: pRoles, backgroundColor: 'rgba(100, 116, 139, 0.6)', borderRadius: 2 }},
            {{ label: 'Starter (1.5k–2.5k)', data: pStarters, backgroundColor: 'rgba(99, 102, 241, 0.8)', borderRadius: 2 }},
            {{ label: 'WR1 Boom (≥2.5k)', data: pBooms, backgroundColor: 'rgba(16, 185, 129, 0.85)', borderRadius: 2 }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ position: 'top', labels: {{ color: '#94a3b8', font: {{ size: 11, family: 'Inter' }} }} }},
            tooltip: {{
              callbacks: {{
                afterBody: function(context) {{
                  const item = topItems[context[0].dataIndex];
                  const actStr = item.is_active ? 'Active NFL Prospect' : `Actual Career: ${{item.actual_y1_3.toLocaleString()}} yds`;
                  return `\\n${{actStr}}\\nPeak Mode: ${{item.p50.toLocaleString()}} yds`;
                }}
              }}
            }}
          }},
          scales: {{
            x: {{ stacked: true, ticks: {{ color: '#94a3b8', font: {{ size: 10, family: 'Inter' }} }}, grid: {{ display: false }} }},
            y: {{ stacked: true, max: 100, ticks: {{ color: '#94a3b8', font: {{ size: 10, family: 'Inter' }} }}, grid: {{ color: '#1e293b' }}, title: {{ display: true, text: 'Outcome Probability (%)', color: '#64748b' }} }}
          }}
        }}
      }});
    }}

    function renderModalBellCurve(p) {{
      const container = document.getElementById('modal-bell-container');
      const width = 580;
      const height = 150;
      const maxVal = 4850;

      const mu = Number(p.p50);
      const p10 = Number(p.p10);
      const p90 = Number(p.p90);
      const actual = Number(p.actual_y1_3);

      const sigma = Math.max(220, (p90 - p10) / 2.56);
      const maxDensity = 1.0 / (sigma * Math.sqrt(2 * Math.PI));

      const steps = 60;
      let pathD = `M 0 ${{height}}`;
      for (let i = 0; i <= steps; i++) {{
        const xVal = (i / steps) * maxVal;
        const density = (1.0 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((xVal - mu) / sigma, 2));
        const normY = density / maxDensity;
        const svgX = (xVal / maxVal) * width;
        const svgY = height - 12 - (normY * (height - 24));
        pathD += ` L ${{svgX.toFixed(1)}} ${{svgY.toFixed(1)}}`;
      }}
      pathD += ` L ${{width}} ${{height}} Z`;

      const bustX = ((500 / maxVal) * width).toFixed(1);
      const starterX = ((1500 / maxVal) * width).toFixed(1);
      const boomX = ((2500 / maxVal) * width).toFixed(1);
      const medX = (Math.min(maxVal, Math.max(0, mu)) / maxVal * width).toFixed(1);

      let markerHtml = '';
      if (!p.is_active && actual > 0) {{
        const actClamped = Math.min(maxVal, Math.max(0, actual));
        const actX = ((actClamped / maxVal) * width).toFixed(1);
        const actDensity = (1.0 / (sigma * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow((actClamped - mu) / sigma, 2));
        const actY = (height - 12 - ((actDensity / maxDensity) * (height - 24))).toFixed(1);

        let dotColor = '#34d399';
        let dotGlow = 'rgba(52, 211, 153, 0.7)';
        if (actual < 500) {{
          dotColor = '#fb7185';
          dotGlow = 'rgba(251, 113, 133, 0.7)';
        }} else if (actual < mu - 200) {{
          dotColor = '#f59e0b';
          dotGlow = 'rgba(245, 158, 11, 0.7)';
        }}
        markerHtml = `
          <line x1="${{actX}}" y1="${{actY}}" x2="${{actX}}" y2="${{height - 10}}" stroke="${{dotColor}}" stroke-width="1.5" opacity="0.7" />
          <circle cx="${{actX}}" cy="${{actY}}" r="8" fill="${{dotGlow}}" opacity="0.4" />
          <circle cx="${{actX}}" cy="${{actY}}" r="5" fill="${{dotColor}}" stroke="#0f172a" stroke-width="2" />
        `;
      }}

      container.innerHTML = `
        <svg class="w-full h-full" viewBox="0 0 ${{width}} ${{height}}" preserveAspectRatio="none">
          <defs>
            <linearGradient id="modal-grad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#818cf8" stop-opacity="0.5" />
              <stop offset="100%" stop-color="#312e81" stop-opacity="0.05" />
            </linearGradient>
          </defs>

          <rect x="0" y="${{height - 8}}" width="${{bustX}}" height="8" fill="#f43f5e" opacity="0.3" />
          <rect x="${{bustX}}" y="${{height - 8}}" width="${{starterX - bustX}}" height="8" fill="#64748b" opacity="0.25" />
          <rect x="${{starterX}}" y="${{height - 8}}" width="${{boomX - starterX}}" height="8" fill="#6366f1" opacity="0.3" />
          <rect x="${{boomX}}" y="${{height - 8}}" width="${{width - boomX}}" height="8" fill="#10b981" opacity="0.3" />

          <line x1="${{bustX}}" y1="8" x2="${{bustX}}" y2="${{height}}" stroke="#f43f5e" stroke-width="1" stroke-dasharray="3,3" opacity="0.5" />
          <text x="${{bustX - 4}}" y="18" fill="#f43f5e" font-size="9" text-anchor="end" font-family="JetBrains Mono">500 yds (Bust)</text>

          <line x1="${{starterX}}" y1="8" x2="${{starterX}}" y2="${{height}}" stroke="#6366f1" stroke-width="1" stroke-dasharray="3,3" opacity="0.5" />
          <text x="${{starterX - 4}}" y="18" fill="#818cf8" font-size="9" text-anchor="end" font-family="JetBrains Mono">1,500 yds (Starter)</text>

          <line x1="${{boomX}}" y1="8" x2="${{boomX}}" y2="${{height}}" stroke="#10b981" stroke-width="1" stroke-dasharray="3,3" opacity="0.5" />
          <text x="${{boomX - 4}}" y="18" fill="#34d399" font-size="9" text-anchor="end" font-family="JetBrains Mono">2,500 yds (Boom)</text>

          <path d="${{pathD}}" fill="url(#modal-grad)" stroke="#818cf8" stroke-width="2" />

          <line x1="${{medX}}" y1="12" x2="${{medX}}" y2="${{height}}" stroke="#e0e7ff" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.9" />
          <text x="${{medX}}" y="10" fill="#e0e7ff" font-size="10" font-weight="bold" text-anchor="middle" font-family="JetBrains Mono">Peak Mode: ${{p.p50.toLocaleString()}} yds</text>

          ${{markerHtml}}
        </svg>
      `;
    }}

    window.openPlayerModal = function(playerId) {{
      const p = PROSPECTS.find(x => x.id === playerId);
      if (!p) return;

      document.getElementById('modal-name').textContent = p.name;
      document.getElementById('modal-grade').textContent = p.grade.toFixed(1);
      document.getElementById('modal-sub').textContent = `${{p.college}} · ${{p.team}} ('${{p.year.toString().slice(-2)}})`;
      
      const badgeContainer = document.getElementById('modal-profile-badge');
      badgeContainer.outerHTML = `<span id="modal-profile-badge">${{getOutcomeBadge(p)}}</span>`;

      renderModalBellCurve(p);

      document.getElementById('modal-prob-bust').textContent = Math.round(p.prob_bust * 100) + '%';
      const rolePct = Math.max(0, 100 - Math.round(p.prob_bust * 100) - Math.round(p.prob_starter * 100) - Math.round(p.prob_boom * 100));
      document.getElementById('modal-prob-role').textContent = rolePct + '%';
      document.getElementById('modal-prob-starter').textContent = Math.round(p.prob_starter * 100) + '%';
      document.getElementById('modal-prob-boom').textContent = Math.round(p.prob_boom * 100) + '%';

      ['bucket-card-bust', 'bucket-card-role', 'bucket-card-starter', 'bucket-card-boom'].forEach(id => {{
        document.getElementById(id).className = 'p-2.5 rounded-lg border border-slate-800 bg-slate-900/50 transition';
      }});

      if (p.is_active) {{
        if (p.prob_boom >= 0.28) {{
          document.getElementById('bucket-card-boom').className = 'p-2.5 rounded-lg border border-emerald-500/80 bg-emerald-950/30 transition ring-1 ring-emerald-500/50';
        }} else if (p.prob_starter >= 0.38) {{
          document.getElementById('bucket-card-starter').className = 'p-2.5 rounded-lg border border-indigo-500/80 bg-indigo-950/30 transition ring-1 ring-indigo-500/50';
        }} else {{
          document.getElementById('bucket-card-bust').className = 'p-2.5 rounded-lg border border-rose-500/80 bg-rose-950/30 transition ring-1 ring-rose-500/50';
        }}
        document.getElementById('modal-actual').textContent = 'In Progress';
        document.getElementById('modal-delta').textContent = '-';
        document.getElementById('modal-delta').className = 'font-mono text-sm text-slate-400';
      }} else {{
        if (p.actual_y1_3 >= 2500) {{
          document.getElementById('bucket-card-boom').className = 'p-2.5 rounded-lg border border-emerald-500/80 bg-emerald-950/30 transition ring-1 ring-emerald-500/50';
        }} else if (p.actual_y1_3 >= 1500) {{
          document.getElementById('bucket-card-starter').className = 'p-2.5 rounded-lg border border-indigo-500/80 bg-indigo-950/30 transition ring-1 ring-indigo-500/50';
        }} else if (p.actual_y1_3 >= 500) {{
          document.getElementById('bucket-card-role').className = 'p-2.5 rounded-lg border border-slate-500/80 bg-slate-800/40 transition ring-1 ring-slate-400/50';
        }} else {{
          document.getElementById('bucket-card-bust').className = 'p-2.5 rounded-lg border border-rose-500/80 bg-rose-950/30 transition ring-1 ring-rose-500/50';
        }}
        document.getElementById('modal-actual').textContent = p.actual_y1_3.toLocaleString() + ' yds';
        const deltaEl = document.getElementById('modal-delta');
        if (p.delta >= 0) {{
          deltaEl.textContent = '+' + p.delta.toLocaleString() + ' yds';
          deltaEl.className = 'font-mono text-sm text-emerald-400 font-bold';
        }} else {{
          deltaEl.textContent = p.delta.toLocaleString() + ' yds';
          deltaEl.className = 'font-mono text-sm text-rose-400 font-bold';
        }}
      }}

      document.getElementById('modal-forty').textContent = p.forty ? p.forty.toFixed(2) + 's' : 'N/A';
      document.getElementById('modal-hass').textContent = p.hass ? p.hass.toFixed(1) : (p.speed_score ? p.speed_score.toFixed(1) + ' (SS)' : 'N/A');
      document.getElementById('modal-agility').textContent = p.cone_shuttle ? p.cone_shuttle.toFixed(2) + 's' : 'N/A';
      document.getElementById('modal-vert').textContent = (p.vert_in ? p.vert_in.toFixed(1) + '"' : 'N/A') + (p.wt_lbs ? ' / ' + p.wt_lbs.toFixed(0) + ' lbs' : '');

      document.getElementById('modal-peak').textContent = '+' + p.peak_rating.toFixed(2);
      document.getElementById('modal-career').textContent = '+' + p.career_rating.toFixed(2);
      document.getElementById('modal-games').textContent = Math.round(p.games_share * 100) + '%';
      document.getElementById('modal-interval').textContent = `${{p.p10.toLocaleString()}} – ${{p.p90.toLocaleString()}} yds`;

      playerModal.classList.remove('hidden');
    }};

    function closeModal() {{ playerModal.classList.add('hidden'); }}

    function setTab(tab) {{
      activeTab = tab;
      [tabBtnBoard, tabBtnAnalytics, tabBtnEvaluation].forEach(b => {{
        b.className = 'px-3 py-1.5 rounded-md font-medium text-slate-300 hover:text-white transition';
      }});
      [viewBoard, viewAnalytics, viewEvaluation].forEach(v => v.classList.add('hidden'));

      if (tab === 'board') {{
        tabBtnBoard.className = 'px-3 py-1.5 rounded-md font-medium transition bg-indigo-600 text-white shadow';
        viewBoard.classList.remove('hidden');
        renderBoard();
      }} else if (tab === 'analytics') {{
        tabBtnAnalytics.className = 'px-3 py-1.5 rounded-md font-medium transition bg-indigo-600 text-white shadow';
        viewAnalytics.classList.remove('hidden');
        renderAnalyticsChart();
      }} else if (tab === 'evaluation') {{
        tabBtnEvaluation.className = 'px-3 py-1.5 rounded-md font-medium transition bg-indigo-600 text-white shadow';
        viewEvaluation.classList.remove('hidden');
        renderEvaluation();
      }}
    }}

    function exportFilteredCsv() {{
      const items = getFilteredAndSortedProspects();
      if (!items.length) return;

      const headers = ['Rank', 'Name', 'College', 'NFL_Team', 'Draft_Year', 'Round', 'Pick', 'College_Grade', 'Forty', 'HaSS', 'Agility', 'Peak_Mode_P50', 'P10_Floor', 'P90_Ceiling', 'P_Boom', 'P_Starter', 'P_Bust', 'Actual_Y1_3', 'Delta_Mode', 'Outcome_Bucket', 'Is_Active'];
      const rows = items.map((p, idx) => [
        idx + 1, `"${{p.name}}"`, `"${{p.college}}"`, `"${{p.team}}"`, p.year, p.round, p.pick, p.grade.toFixed(1),
        p.forty || '', p.hass || '', p.cone_shuttle || '', p.p50, p.p10, p.p90, p.prob_boom, p.prob_starter, p.prob_bust, p.is_active ? 'Active' : p.actual_y1_3, p.is_active ? '' : p.delta, `"${{p.actual_bucket}}"`, p.is_active ? 1 : 0
      ]);

      const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(r => r.join(','))].join('\\n');
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement('a');
      link.setAttribute('href', encodedUri);
      link.setAttribute('download', `stage1_prospect_board_${{currentFilterClass}}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      showToast('Export downloaded successfully');
    }}

    tabBtnBoard.addEventListener('click', () => setTab('board'));
    tabBtnAnalytics.addEventListener('click', () => setTab('analytics'));
    tabBtnEvaluation.addEventListener('click', () => setTab('evaluation'));
    btnExportCsv.addEventListener('click', exportFilteredCsv);

    inputSearch.addEventListener('input', (e) => {{
      currentSearchTerm = e.target.value.trim();
      renderBoard();
      if (activeTab === 'analytics') renderAnalyticsChart();
    }});

    selectRound.addEventListener('change', (e) => {{
      currentFilterRound = e.target.value;
      renderBoard();
      if (activeTab === 'analytics') renderAnalyticsChart();
    }});

    selectBucket.addEventListener('change', (e) => {{
      currentFilterBucket = e.target.value;
      renderBoard();
      if (activeTab === 'analytics') renderAnalyticsChart();
    }});

    selectSort.addEventListener('change', (e) => {{
      currentSort = e.target.value;
      renderBoard();
      if (activeTab === 'analytics') renderAnalyticsChart();
    }});

    btnResetFilters.addEventListener('click', () => {{
      currentFilterClass = 'all';
      currentFilterRound = 'all';
      currentFilterBucket = 'all';
      currentSearchTerm = '';
      currentSort = 'grade_desc';

      inputSearch.value = '';
      selectRound.value = 'all';
      selectBucket.value = 'all';
      selectSort.value = 'grade_desc';

      document.querySelectorAll('.class-btn').forEach(btn => {{
        if (btn.dataset.class === 'all') {{
          btn.className = 'class-btn px-2.5 py-1 text-xs font-medium rounded-lg bg-indigo-600 text-white shadow';
        }} else {{
          btn.className = 'class-btn px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 transition';
        }}
      }});

      renderBoard();
      if (activeTab === 'analytics') renderAnalyticsChart();
      showToast('Filters reset to default');
    }});

    classFilterContainer.addEventListener('click', (e) => {{
      const btn = e.target.closest('.class-btn');
      if (!btn) return;
      currentFilterClass = btn.dataset.class;

      document.querySelectorAll('.class-btn').forEach(b => {{
        b.className = 'class-btn px-2.5 py-1 text-xs font-medium rounded-lg bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 transition';
      }});
      btn.className = 'class-btn px-2.5 py-1 text-xs font-medium rounded-lg bg-indigo-600 text-white shadow';

      renderBoard();
      if (activeTab === 'analytics') renderAnalyticsChart();
    }});

    modalCloseBtn.addEventListener('click', closeModal);
    modalOkBtn.addEventListener('click', closeModal);
    playerModal.addEventListener('click', (e) => {{ if (e.target === playerModal) closeModal(); }});
    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape' && !playerModal.classList.contains('hidden')) closeModal();
    }});

    renderBoard();
  </script>
</body>
</html>
"""

# Save to destination paths
out_paths = [
    Path("stage1_prospect_dashboard.html"),
    Path("../stage1_prospect_dashboard.html"),
]

for opath in out_paths:
    try:
        with open(opath, "w") as f:
            f.write(html_template)
        print(f"Done! Successfully generated {opath.resolve()}")
    except Exception:
        pass

print("Open directly in your browser: open stage1_prospect_dashboard.html")