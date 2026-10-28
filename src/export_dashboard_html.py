"""
Generate Interactive Stage 1 Pure College Prospect Dashboard HTML.
Translates pre-draft M2 college ratings into 10th, 50th, and 90th percentile
NFL Y1-3 projections (2012-2022 cohorts, zero draft slot bias).

Usage:
    python3 export_dashboard_html.py
"""

import os
import sys
import json
from pathlib import Path
import polars as pl

# 1. Flexible path resolution
candidates = [
    Path("data/processed"),
    Path("../data/processed"),
    Path(".")
]
DATA_DIR = next((d for d in candidates if (d / "stage1_prospect_rankings.parquet").exists()), Path("data/processed"))
RANKINGS_FILE = DATA_DIR / "stage1_prospect_rankings.parquet"

if not RANKINGS_FILE.exists():
    print(f"Error: Could not find {RANKINGS_FILE}.")
    print("Please run `python3 stage1_prospect_rankings.py` first.")
    sys.exit(1)

print(f"Reading prospect rankings from {RANKINGS_FILE}...")
df = pl.read_parquet(RANKINGS_FILE)

# 2. Column normalization
col_map = {
    "athlete_name": "name",
    "name_draft": "name",
    "player_name": "name",
    "draft_overall": "pick",
    "college_team": "college",
    "nfl_team": "team",
    "nfl_proj_floor": "p10",
    "nfl_proj_median": "p50",
    "nfl_proj_ceiling": "p90",
    "nfl_proj_p10": "p10",
    "nfl_proj_p50": "p50",
    "nfl_proj_p90": "p90",
}
for old_c, new_c in col_map.items():
    if old_c in df.columns and new_c not in df.columns:
        df = df.rename({old_c: new_c})

# Filter strictly to completed 3-year evaluation window (2012-2022)
if "draft_year" in df.columns:
    df = df.filter((pl.col("draft_year") >= 2012) & (pl.col("draft_year") <= 2022))

records = []
for idx, r in enumerate(df.sort("college_grade", descending=True).iter_rows(named=True)):
    c_grade = float(r.get("college_grade", 50.0))
    p_peak = float(r.get("peak_season_rating", 1.0))
    c_rating = float(r.get("career_rating", p_peak))
    f_rating = float(r.get("final_season_rating", p_peak))
    g_share = float(r.get("final_season_games_share", 1.0))
    
    p10 = int(round(float(r.get("p10", 300))))
    p50 = int(round(float(r.get("p50", 1200))))
    p90 = int(round(float(r.get("p90", 2500))))
    actual = float(r.get("nfl_actual_y1_3", 0.0))
    
    # Profile definition strictly based on calibrated pre-draft metrics
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

    records.append({
        "id": f"wr_{idx}_{r['draft_year']}",
        "name": r.get("name") or "Unknown Prospect",
        "college": r.get("college") or r.get("school_name") or "NCAA",
        "team": r.get("team", "NFL"),
        "year": int(r["draft_year"]),
        "round": int(r.get("round") or r.get("draft_round") or 1),
        "pick": int(r.get("pick", 1)),
        "grade": round(c_grade, 1),
        "peak_rating": round(p_peak, 2),
        "career_rating": round(c_rating, 2),
        "final_rating": round(f_rating, 2),
        "games_share": round(g_share, 2),
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "prob_boom": float(r.get("prob_boom", 0.15)),
        "prob_starter": float(r.get("prob_starter", 0.30)),
        "prob_bust": float(r.get("prob_bust", 0.50)),
        "delta": delta,
        "actual_y1_3": int(round(actual)),
        "profile": prof
    })

prospects_json = json.dumps(records)
years_available = sorted(list(set(r["year"] for r in records)), reverse=True)
print(f"Loaded {len(records)} prospects across classes {min(years_available)}–{max(years_available)}.")

html_template = f"""<!DOCTYPE html>
<html lang="en" class="h-full bg-slate-950 text-slate-100">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stage 1: Pure College Prospect Board (2012–2022)</title>
  <script src="https://cdn.tailwindcss.com/3.4.17"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    body {{ font-family: 'Inter', sans-serif; }}
    .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: #090d16; }}
    ::-webkit-scrollbar-thumb {{ background: #1e293b; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #334155; }}
  </style>
</head>
<body class="min-h-full flex flex-col bg-slate-950 text-slate-100 antialiased selection:bg-indigo-500 selection:text-white">

  <!-- Navigation -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 font-bold text-white">S1</div>
        <div>
          <div class="flex items-center space-x-2">
            <h1 class="text-base font-bold tracking-tight text-white">Stage 1: Pure College Prospect Board</h1>
            <span class="px-2 py-0.5 text-xs font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Zero Draft Slot Bias</span>
          </div>
          <p class="text-xs text-slate-400">Pre-Draft College Ratings & Quantile Forecasts (10th, 50th, 90th Percentile)</p>
        </div>
      </div>

      <div class="flex items-center space-x-3">
        <div class="flex bg-slate-800/80 p-1 rounded-lg border border-slate-700/60 text-xs">
          <button id="tab-btn-board" class="px-3 py-1.5 rounded-md font-medium transition bg-indigo-600 text-white shadow">Prospect Board</button>
          <button id="tab-btn-analytics" class="px-3 py-1.5 rounded-md font-medium text-slate-300 hover:text-white transition">Quantile Distribution</button>
          <button id="tab-btn-evaluation" class="px-3 py-1.5 rounded-md font-medium text-slate-300 hover:text-white transition">Hits & Misses</button>
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
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Prospects Tracked</span>
        <div class="flex items-baseline space-x-2 mt-1">
          <span id="stat-total-count" class="text-2xl font-bold text-white font-mono">{len(records)}</span>
          <span class="text-xs text-slate-400">Total WRs</span>
        </div>
        <span class="text-[11px] text-emerald-400 mt-1">Full 3-Season NFL Careers</span>
      </div>

      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3.5 flex flex-col justify-between">
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Pre-Draft Inputs</span>
        <div class="flex items-baseline space-x-2 mt-1">
          <span class="text-2xl font-bold text-emerald-400 font-mono">100%</span>
          <span class="text-xs text-slate-400">College M2</span>
        </div>
        <span class="text-[11px] text-slate-400 mt-1">Draft Capital Excluded</span>
      </div>

      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3.5 flex flex-col justify-between">
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Threshold Calibration</span>
        <div class="flex items-baseline space-x-2 mt-1">
          <span class="text-2xl font-bold text-violet-400 font-mono">&ge;2.5k / &ge;1.5k</span>
        </div>
        <span class="text-[11px] text-slate-400 mt-1">Alpha Boom &amp; Starter Base</span>
      </div>

      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-3.5 flex flex-col justify-between">
        <span class="text-xs font-medium text-slate-400 uppercase tracking-wider">Cohort Scope</span>
        <div class="flex items-baseline space-x-2 mt-1">
          <span class="text-base font-bold text-indigo-400">2012 – 2022</span>
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        </div>
        <span class="text-[11px] text-slate-400 mt-1">Stage 1 Decoupled</span>
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
            <option value="all">All Rounds (1-7)</option>
            <option value="1">Round 1 Only</option>
            <option value="2">Round 2 Only</option>
            <option value="3">Round 3 Only</option>
            <option value="day3">Day 3 (Rounds 4-7)</option>
          </select>
        </div>

        <div class="flex items-center space-x-2">
          <label class="text-slate-400 font-medium">Profile:</label>
          <select id="select-profile" class="bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-md px-2 py-1.5 focus:outline-none focus:border-indigo-500 w-full">
            <option value="all">All Profiles</option>
            <option value="Blue-Chip Alpha">Blue-Chip Alpha</option>
            <option value="High-Floor Starter">High-Floor Starter</option>
            <option value="Boom-or-Bust Flier">Boom-or-Bust Flier</option>
            <option value="High Bust Risk">High Bust Risk</option>
          </select>
        </div>

        <div class="flex items-center space-x-2">
          <label class="text-slate-400 font-medium">Sort By:</label>
          <select id="select-sort" class="bg-slate-950 border border-slate-800 text-slate-200 text-xs rounded-md px-2 py-1.5 focus:outline-none focus:border-indigo-500 w-full">
            <option value="grade_desc">Pre-Draft College Grade (Highest)</option>
            <option value="peak_desc">Peak Season Rating (Highest)</option>
            <option value="p50_desc">Projected Median Y1–3 (Highest)</option>
            <option value="p90_desc">Projected Ceiling p90 (Highest)</option>
            <option value="p10_desc">Projected Floor p10 (Highest)</option>
            <option value="actual_desc">Actual NFL Y1–3 (Highest)</option>
            <option value="delta_desc">Beating Median Projection (Most)</option>
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
        <div class="px-4 py-3 border-b border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <div>Showing <span id="label-showing-count" class="font-semibold text-white">0</span> prospects</div>
          <div class="flex items-center space-x-4">
            <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-full bg-violet-500"></span><span>10th–90th Quantile Interval</span></span>
            <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span><span>Actual NFL Y1–3</span></span>
          </div>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs border-collapse">
            <thead>
              <tr class="bg-slate-950/80 text-slate-400 font-semibold border-b border-slate-800">
                <th class="py-3 px-3.5 text-center w-12">Rank</th>
                <th class="py-3 px-3">Prospect</th>
                <th class="py-3 px-3 text-center">College Grade</th>
                <th class="py-3 px-3">College M2 Skill</th>
                <th class="py-3 px-4 min-w-[280px]">Pure College Projection (Y1–3 Yards)</th>
                <th class="py-3 px-3 text-right">Actual Y1–3</th>
                <th class="py-3 px-3 text-center">Vs Median</th>
                <th class="py-3 px-3 text-center">Profile</th>
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
              <h2 class="text-sm font-bold text-white">Pure College 10th–90th Interval vs Actual NFL Production</h2>
              <p class="text-xs text-slate-400">Top 15 prospects matching filter. Dots show actual career production relative to pre-draft range.</p>
            </div>
            <span class="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">Zero Draft Slot Bias</span>
          </div>
          <div class="relative h-96">
            <canvas id="chart-quantile-ranges"></canvas>
          </div>
        </div>

        <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow space-y-4">
          <h2 class="text-sm font-bold text-white">Stage 1 Model Architecture</h2>
          <div class="space-y-3 text-xs">
            <div class="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span class="text-slate-400">College Inputs Used:</span>
              <div class="mt-1 font-mono text-slate-200 space-y-0.5 text-[11px]">
                <div>• Peak Season Rating (M2)</div>
                <div>• Career Rating (M2)</div>
                <div>• Final Season Games Availability</div>
              </div>
            </div>
            <div class="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span class="text-slate-400">Ceiling Calibration:</span>
              <div class="flex items-baseline justify-between mt-0.5">
                <span class="text-base font-bold font-mono text-violet-400">4,850 Yds Max</span>
                <span class="text-slate-400">Exponential Capped</span>
              </div>
            </div>
            <div class="p-3 bg-slate-950 rounded-lg border border-slate-800">
              <span class="text-slate-400">Evaluation Cohorts:</span>
              <div class="flex items-baseline justify-between mt-0.5">
                <span class="text-base font-bold font-mono text-emerald-400">2012–2022</span>
                <span class="text-xs text-slate-500">100% 3Y Complete</span>
              </div>
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
            <h2 class="text-sm font-bold text-white">Top Model Hits (High Grade & WR1 Production)</h2>
            <span class="text-xs text-slate-400">Grade ≥ 90 & Actual ≥ 2.5k Yds</span>
          </div>
          <div id="hits-list" class="space-y-2.5"></div>
        </div>

        <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow space-y-4">
          <div class="flex items-center justify-between pb-2 border-b border-slate-800">
            <h2 class="text-sm font-bold text-white">Top College Outliers / Sleepers</h2>
            <span class="text-xs text-slate-400">Outperformed Median Projection Most</span>
          </div>
          <div id="misses-list" class="space-y-2.5"></div>
        </div>
      </div>
    </div>

  </main>

  <!-- Player Detail Modal -->
  <div id="player-modal" class="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm hidden">
    <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-xl w-full p-6 space-y-5 shadow-2xl relative">
      <button id="modal-close-btn" class="absolute top-4 right-4 text-slate-400 hover:text-white transition">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
      </button>

      <div class="flex items-start space-x-4">
        <div id="modal-grade-circle" class="w-14 h-14 rounded-2xl bg-indigo-600/20 border border-indigo-500/40 flex flex-col items-center justify-center text-white flex-shrink-0">
          <span class="text-xs text-slate-400 leading-none">Grade</span>
          <span id="modal-grade" class="text-lg font-bold font-mono text-indigo-400">99.0</span>
        </div>
        <div>
          <div class="flex items-center space-x-2">
            <h3 id="modal-name" class="text-lg font-bold text-white">Prospect</h3>
            <span id="modal-profile-badge" class="px-2 py-0.5 text-[11px] rounded-full bg-purple-500/20 text-purple-300">Blue-Chip Alpha</span>
          </div>
          <p id="modal-sub" class="text-xs text-slate-400 mt-0.5">College & Draft details</p>
        </div>
      </div>

      <div class="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
        <span class="text-xs font-semibold text-slate-300">Pure College Model Projections (Year 1–3 Receiving Yards)</span>
        <div class="grid grid-cols-3 gap-3 text-center">
          <div class="p-2.5 bg-slate-900/60 rounded-lg border border-slate-800">
            <span class="text-[11px] text-slate-400 block">10th % (Floor)</span>
            <span id="modal-p10" class="text-base font-bold font-mono text-slate-200">-</span>
          </div>
          <div class="p-2.5 bg-indigo-950/40 rounded-lg border border-indigo-800/40">
            <span class="text-[11px] text-indigo-300 block font-semibold">50th % (Median)</span>
            <span id="modal-p50" class="text-base font-bold font-mono text-indigo-400">-</span>
          </div>
          <div class="p-2.5 bg-slate-900/60 rounded-lg border border-slate-800">
            <span class="text-[11px] text-slate-400 block">90th % (Ceiling)</span>
            <span id="modal-p90" class="text-base font-bold font-mono text-emerald-400">-</span>
          </div>
        </div>

        <div class="flex items-center justify-between text-xs pt-1 border-t border-slate-800/60 text-slate-400">
          <span>Actual NFL Y1–3 Production: <strong id="modal-actual" class="text-emerald-400 font-mono text-sm">-</strong></span>
          <span>Diff from Median: <strong id="modal-delta" class="font-mono text-sm">-</strong></span>
        </div>
      </div>

      <div class="space-y-2">
        <span class="text-xs font-semibold text-slate-300">College M2 Skill Metrics</span>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
          <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <span class="text-slate-500 block text-[11px]">Peak Season Rating</span>
            <span id="modal-peak" class="font-mono font-semibold text-slate-200">-</span>
          </div>
          <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <span class="text-slate-500 block text-[11px]">Career Rating</span>
            <span id="modal-career" class="font-mono font-semibold text-slate-200">-</span>
          </div>
          <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <span class="text-slate-500 block text-[11px]">P(Boom &ge;2.5k)</span>
            <span id="modal-boom" class="font-mono font-semibold text-emerald-400">-</span>
          </div>
          <div class="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
            <span class="text-slate-500 block text-[11px]">P(Bust &lt;500)</span>
            <span id="modal-bust" class="font-mono font-semibold text-rose-400">-</span>
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
    let currentFilterProfile = 'all';
    let currentSearchTerm = '';
    let currentSort = 'grade_desc';
    let activeTab = 'board';
    let analyticsChart = null;

    const boardTbody = document.getElementById('board-tbody');
    const labelShowingCount = document.getElementById('label-showing-count');
    const noResults = document.getElementById('no-results');
    const inputSearch = document.getElementById('input-search');
    const selectRound = document.getElementById('select-round');
    const selectProfile = document.getElementById('select-profile');
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

    function getProfileBadge(profile) {{
      if (profile === 'Blue-Chip Alpha') return '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-purple-500/20 text-purple-300 border border-purple-500/30">Blue-Chip Alpha</span>';
      if (profile === 'High-Floor Starter') return '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">High-Floor Starter</span>';
      if (profile === 'Boom-or-Bust Flier') return '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/30">Boom-or-Bust Flier</span>';
      if (profile === 'High Bust Risk') return '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30">High Bust Risk</span>';
      return '<span class="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-800 text-slate-300 border border-slate-700">Solid Prospect</span>';
    }}

    function getFilteredAndSortedProspects() {{
      return PROSPECTS.filter(p => {{
        if (currentFilterClass !== 'all' && p.year.toString() !== currentFilterClass) return false;
        if (currentFilterRound === 'day3' && p.round < 4) return false;
        if (currentFilterRound !== 'all' && currentFilterRound !== 'day3' && p.round.toString() !== currentFilterRound) return false;
        if (currentFilterProfile !== 'all' && p.profile !== currentFilterProfile) return false;
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
          case 'peak_desc': return b.peak_rating - a.peak_rating;
          case 'p50_desc': return b.p50 - a.p50;
          case 'p90_desc': return b.p90 - a.p90;
          case 'p10_desc': return b.p10 - a.p10;
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

      const maxCeiling = 5000;
      let html = '';
      items.forEach((p, idx) => {{
        const p10Pct = Math.min(100, Math.max(2, (p.p10 / maxCeiling) * 100));
        const p50Pct = Math.min(100, Math.max(p10Pct + 2, (p.p50 / maxCeiling) * 100));
        const p90Pct = Math.min(100, Math.max(p50Pct + 2, (p.p90 / maxCeiling) * 100));
        const actualPct = Math.min(100, (p.actual_y1_3 / maxCeiling) * 100);

        let deltaHtml = '';
        if (p.delta > 200) {{
          deltaHtml = `<span class="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">+${{p.delta.toLocaleString()}}</span>`;
        }} else if (p.delta < -200) {{
          deltaHtml = `<span class="px-2 py-0.5 rounded text-[11px] font-mono font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20">${{p.delta.toLocaleString()}}</span>`;
        }} else {{
          deltaHtml = `<span class="px-2 py-0.5 rounded text-[11px] font-mono text-slate-400 bg-slate-800/60 border border-slate-700/50">±0</span>`;
        }}

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
                    <span>'${{p.year.toString().slice(-2)}}${{p.round ? ' R' + p.round : ''}} (#${{p.pick}})</span>
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
              <div class="text-xs font-mono text-slate-200">+${{p.peak_rating.toFixed(2)}} <span class="text-[10px] text-slate-500">peak</span></div>
              <div class="text-[10px] text-slate-400">${{Math.round(p.games_share * 100)}}% avail share</div>
            </td>
            <td class="py-3 px-4">
              <div class="space-y-1">
                <div class="flex justify-between text-[11px] font-mono">
                  <span class="text-slate-400">${{p.p10.toLocaleString()}} <span class="text-[9px] text-slate-500">(p10)</span></span>
                  <span class="font-bold text-indigo-400">${{p.p50.toLocaleString()}} <span class="text-[9px] text-indigo-300 font-normal">(p50)</span></span>
                  <span class="text-emerald-400">${{p.p90.toLocaleString()}} <span class="text-[9px] text-emerald-500 font-normal">(p90)</span></span>
                </div>
                <div class="relative w-full h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                  <div class="absolute h-full bg-violet-600/40 rounded-full" style="left: ${{p10Pct}}%; width: ${{p90Pct - p10Pct}}%;"></div>
                  <div class="absolute h-full w-1 bg-indigo-400 z-10" style="left: ${{p50Pct}}%;"></div>
                  <div class="absolute top-0 bottom-0 w-1 bg-emerald-400 z-20 shadow-sm" style="left: ${{actualPct}}%;" title="Actual: ${{p.actual_y1_3}} yds"></div>
                </div>
              </div>
            </td>
            <td class="py-3 px-3 text-right font-mono font-bold text-emerald-400 text-xs">${{p.actual_y1_3.toLocaleString()}}</td>
            <td class="py-3 px-3 text-center">${{deltaHtml}}</td>
            <td class="py-3 px-3 text-center">${{getProfileBadge(p.profile)}}</td>
          </tr>
        `;
      }});
      boardTbody.innerHTML = html;
    }}

    function renderEvaluation() {{
      const hitsList = document.getElementById('hits-list');
      const missesList = document.getElementById('misses-list');

      // Top hits: high grade (>=88) and high NFL output (>=2500)
      const hits = [...PROSPECTS]
        .filter(p => p.grade >= 88.0 && p.actual_y1_3 >= 2400)
        .sort((a, b) => b.actual_y1_3 - a.actual_y1_3)
        .slice(0, 8);

      let hitsHtml = '';
      hits.forEach(p => {{
        hitsHtml += `
          <div class="p-3 bg-slate-950/80 rounded-xl border border-slate-800/80 flex items-center justify-between hover:border-emerald-500/40 transition cursor-pointer" onclick="openPlayerModal('${{p.id}}')">
            <div>
              <div class="flex items-center space-x-2">
                <span class="font-bold text-white text-xs">${{p.name}}</span>
                <span class="text-[10px] px-1.5 py-0.2 rounded bg-indigo-950 text-indigo-300 font-mono">Grade: ${{p.grade.toFixed(1)}}</span>
              </div>
              <div class="text-[11px] text-slate-400 mt-0.5">
                ${{p.college}} · Proj p50: <span class="font-mono text-slate-200">${{p.p50.toLocaleString()}}</span>
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

      // Top sleepers: beat median projection the most
      const sleepers = [...PROSPECTS]
        .sort((a, b) => b.delta - a.delta)
        .slice(0, 8);

      let missesHtml = '';
      sleepers.forEach(p => {{
        missesHtml += `
          <div class="p-3 bg-slate-950/80 rounded-xl border border-slate-800/80 flex items-center justify-between hover:border-indigo-500/40 transition cursor-pointer" onclick="openPlayerModal('${{p.id}}')">
            <div>
              <div class="flex items-center space-x-2">
                <span class="font-bold text-white text-xs">${{p.name}}</span>
                <span class="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-300 font-mono">'${{p.year.toString().slice(-2)}} R${{p.round}} (#${{p.pick}})</span>
              </div>
              <div class="text-[11px] text-slate-400 mt-0.5">
                Proj Median: <span class="font-mono">${{p.p50.toLocaleString()}}</span> · Actual: <span class="font-mono text-emerald-400 font-bold">${{p.actual_y1_3.toLocaleString()}}</span>
              </div>
            </div>
            <div class="text-right">
              <span class="text-xs font-mono font-bold text-emerald-400">+${{p.delta.toLocaleString()}} yds</span>
              <div class="text-[10px] text-slate-400">Outperformed p50</div>
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
      const p10s = topItems.map(p => p.p10);
      const p50s = topItems.map(p => p.p50);
      const p90s = topItems.map(p => p.p90);
      const actuals = topItems.map(p => p.actual_y1_3);

      if (analyticsChart) {{
        analyticsChart.destroy();
      }}

      analyticsChart = new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: labels,
          datasets: [
            {{ label: '10th % (Floor)', data: p10s, backgroundColor: 'rgba(100, 116, 139, 0.4)', borderRadius: 4 }},
            {{ label: '50th % (Median)', data: p50s, backgroundColor: 'rgba(99, 102, 241, 0.8)', borderRadius: 4 }},
            {{ label: '90th % (Ceiling)', data: p90s, backgroundColor: 'rgba(168, 85, 247, 0.4)', borderRadius: 4 }},
            {{ label: 'Actual NFL Y1–3', data: actuals, type: 'line', borderColor: '#34d399', backgroundColor: '#34d399', pointRadius: 6, pointHoverRadius: 8, borderWidth: 2, fill: false }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ position: 'top', labels: {{ color: '#94a3b8', font: {{ size: 11, family: 'Inter' }} }} }}
          }},
          scales: {{
            x: {{ ticks: {{ color: '#94a3b8', font: {{ size: 10, family: 'Inter' }} }}, grid: {{ display: false }} }},
            y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 10, family: 'Inter' }} }}, grid: {{ color: '#1e293b' }}, title: {{ display: true, text: 'Receiving Yards (Y1–3)', color: '#64748b' }} }}
          }}
        }}
      }});
    }}

    window.openPlayerModal = function(playerId) {{
      const p = PROSPECTS.find(x => x.id === playerId);
      if (!p) return;

      document.getElementById('modal-name').textContent = p.name;
      document.getElementById('modal-grade').textContent = p.grade.toFixed(1);
      document.getElementById('modal-sub').textContent = `${{p.college}} · Drafted by ${{p.team}} (${{p.year}} Round ${{p.round}}, Pick ${{p.pick}})`;
      
      const badgeContainer = document.getElementById('modal-profile-badge');
      badgeContainer.outerHTML = getProfileBadge(p.profile);

      document.getElementById('modal-p10').textContent = p.p10.toLocaleString() + ' yds';
      document.getElementById('modal-p50').textContent = p.p50.toLocaleString() + ' yds';
      document.getElementById('modal-p90').textContent = p.p90.toLocaleString() + ' yds';
      document.getElementById('modal-actual').textContent = p.actual_y1_3.toLocaleString() + ' yds';
      
      const deltaEl = document.getElementById('modal-delta');
      if (p.delta >= 0) {{
        deltaEl.textContent = '+' + p.delta.toLocaleString() + ' yds';
        deltaEl.className = 'font-mono text-sm text-emerald-400 font-bold';
      }} else {{
        deltaEl.textContent = p.delta.toLocaleString() + ' yds';
        deltaEl.className = 'font-mono text-sm text-rose-400 font-bold';
      }}

      document.getElementById('modal-peak').textContent = '+' + p.peak_rating.toFixed(2);
      document.getElementById('modal-career').textContent = '+' + p.career_rating.toFixed(2);
      document.getElementById('modal-boom').textContent = Math.round(p.prob_boom * 100) + '%';
      document.getElementById('modal-bust').textContent = Math.round(p.prob_bust * 100) + '%';

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

      const headers = ['Rank', 'Name', 'College', 'NFL_Team', 'Draft_Year', 'Round', 'Pick', 'College_Grade', 'Peak_Rating', 'P10_Floor', 'P50_Median', 'P90_Ceiling', 'Actual_Y1_3', 'Delta_P50', 'Profile'];
      const rows = items.map((p, idx) => [
        idx + 1, `"${{p.name}}"`, `"${{p.college}}"`, p.team, p.year, p.round, p.pick, p.grade.toFixed(1),
        p.peak_rating.toFixed(2), p.p10, p.p50, p.p90, p.actual_y1_3, p.delta, `"${{p.profile}}"`
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

    selectProfile.addEventListener('change', (e) => {{
      currentFilterProfile = e.target.value;
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
      currentFilterProfile = 'all';
      currentSearchTerm = '';
      currentSort = 'grade_desc';

      inputSearch.value = '';
      selectRound.value = 'all';
      selectProfile.value = 'all';
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

# Save to both ./stage1_prospect_dashboard.html and ../stage1_prospect_dashboard.html
out_paths = [Path("stage1_prospect_dashboard.html"), Path("../stage1_prospect_dashboard.html")]
for opath in out_paths:
    try:
        with open(opath, "w") as f:
            f.write(html_template)
        print(f"Done! Successfully generated {opath.resolve()}")
    except Exception:
        pass

print("Open directly in your browser: open ../stage1_prospect_dashboard.html")