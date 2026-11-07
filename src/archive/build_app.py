"""
Complete Standalone Single-Tab Modern Prospect Rankings Web App Builder (Column-Fix)
Robustly resolves schema variations across stage1, stage2, and recent parquet files.
"""

import json
import math
import os
import re
import sys
from pathlib import Path
import pyarrow.parquet as pq

# Search paths
candidates = [
    Path("data/processed"),
    Path("../data/processed"),
    Path("src/data/processed"),
    Path("."),
]
DATA_DIR = next(
    (d for d in candidates if (d / "stage2_team_fit_rankings.parquet").exists() or (d / "stage1_prospect_rankings.parquet").exists()),
    Path("data/processed"),
)

s2_path = DATA_DIR / "stage2_team_fit_rankings.parquet"
s1_path = DATA_DIR / "stage1_prospect_rankings.parquet"
recent_path = DATA_DIR / "stage1_recent_prospects.parquet"
json_path = Path("artifacts/file_generation/ttl=63d/intermediate/prospects_data.json")

prospects = []

def extract_val(row, keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            try:
                v = row[k]
                if not (isinstance(v, float) and math.isnan(v)):
                    return v
            except Exception:
                pass
    return default

# 1. Load recent prospects (2024, 2025, 2023)
if recent_path.exists():
    t_rec = pq.read_table(recent_path)
    for i in range(len(t_rec)):
        row = {k: t_rec[k][i].as_py() for k in t_rec.column_names}
        p_name = row.get("name") or "Prospect"
        c_grade = float(extract_val(row, ["college_grade", "grade"], 80.0))
        yr = int(extract_val(row, ["draft_year", "year"], 2024))
        
        p_x = float(extract_val(row, ["p_x"], 0.45))
        p_z = float(extract_val(row, ["p_z"], 0.35))
        p_slot = float(extract_val(row, ["p_slot"], 0.20))
        arch = row.get("archetype_primary") or ("Boundary X" if p_x >= max(p_z, p_slot) else ("Flanker Z" if p_z >= p_slot else "Interior Slot"))

        p10 = int(round(float(extract_val(row, ["stage2_p10", "nfl_proj_floor", "p10"], 300))))
        p50 = int(round(float(extract_val(row, ["stage2_p50", "nfl_proj_median", "stage1_p50", "p50"], 1400))))
        p90 = int(round(float(extract_val(row, ["stage2_p90", "nfl_proj_ceiling", "p90"], 3200))))

        prob_starter = float(extract_val(row, ["stage2_prob_starter", "prob_starter"], 0.35))
        prob_boom = float(extract_val(row, ["stage2_prob_boom", "prob_boom"], 0.18))
        prob_bust = float(extract_val(row, ["stage2_prob_bust", "prob_bust"], 0.38))

        prospects.append({
            "id": f"p_rec_{i}_{yr}",
            "name": p_name,
            "college": extract_val(row, ["college", "college_team", "school_name"], "NCAA"),
            "year": yr,
            "grade": round(c_grade, 1),
            "archetype": arch,
            "p_x": round(p_x, 2),
            "p_z": round(p_z, 2),
            "p_slot": round(p_slot, 2),
            "base_p10": p10,
            "base_p50": p50,
            "base_p90": p90,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "prob_starter": round(prob_starter, 3),
            "prob_boom": round(prob_boom, 3),
            "prob_bust": round(prob_bust, 3),
            "nfl_team": extract_val(row, ["draft_slot_info", "nfl_team", "team"], "NFL"),
            "mock_team": None,
        })

# 2. Load historical prospects (2012–2022)
target_hist_file = s2_path if s2_path.exists() else (s1_path if s1_path.exists() else None)

if target_hist_file:
    print(f"Loading historical prospects from {target_hist_file}...")
    t_hist = pq.read_table(target_hist_file)
    for i in range(len(t_hist)):
        row = {k: t_hist[k][i].as_py() for k in t_hist.column_names}
        p_name = row.get("name") or row.get("athlete_name") or row.get("player_name") or "Prospect"
        if any(p["name"].lower() == p_name.lower() for p in prospects):
            continue
        c_grade = float(extract_val(row, ["college_grade", "grade"], 75.0))
        yr = int(extract_val(row, ["draft_year", "year"], 2021))

        p_x = float(extract_val(row, ["p_x"], 0.40))
        p_z = float(extract_val(row, ["p_z"], 0.40))
        p_slot = float(extract_val(row, ["p_slot"], 0.20))
        arch = row.get("archetype_primary") or ("Boundary X" if p_x >= max(p_z, p_slot) else ("Flanker Z" if p_z >= p_slot else "Interior Slot"))

        # FIXED COLUMN RESOLUTION:
        # Check stage2_p90 -> nfl_proj_ceiling -> p90
        p10 = int(round(float(extract_val(row, ["stage2_p10", "nfl_proj_floor", "p10"], 300))))
        p50 = int(round(float(extract_val(row, ["stage2_p50", "nfl_proj_median", "stage1_p50", "p50"], 1300))))
        p90 = int(round(float(extract_val(row, ["stage2_p90", "nfl_proj_ceiling", "p90"], 2900))))

        prob_starter = float(extract_val(row, ["stage2_prob_starter", "prob_starter"], 0.30))
        prob_boom = float(extract_val(row, ["stage2_prob_boom", "prob_boom"], 0.15))
        prob_bust = float(extract_val(row, ["stage2_prob_bust", "prob_bust"], 0.45))

        prospects.append({
            "id": f"p_hist_{i}_{yr}",
            "name": p_name,
            "college": extract_val(row, ["college", "college_team", "school_name"], "NCAA"),
            "year": yr,
            "grade": round(c_grade, 1),
            "archetype": arch,
            "p_x": round(p_x, 2),
            "p_z": round(p_z, 2),
            "p_slot": round(p_slot, 2),
            "base_p10": p10,
            "base_p50": p50,
            "base_p90": p90,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "prob_starter": round(prob_starter, 3),
            "prob_boom": round(prob_boom, 3),
            "prob_bust": round(prob_bust, 3),
            "nfl_team": extract_val(row, ["team", "nfl_team", "draft_slot_info"], "NFL"),
            "mock_team": None,
        })
elif json_path.exists():
    print(f"Loading from fallback JSON {json_path}...")
    with open(json_path) as f:
        raw_json = json.load(f)
    for i, row in enumerate(raw_json):
        p_name = row.get("name") or "Prospect"
        if any(p["name"].lower() == p_name.lower() for p in prospects):
            continue
        c_grade = float(row.get("grade") or 75.0)
        yr = int(row.get("year") or 2021)
        p10 = int(round(float(row.get("p10") or 300)))
        p50 = int(round(float(row.get("p50") or 1300)))
        p90 = int(round(float(row.get("p90") or 2900)))
        prospects.append({
            "id": f"p_json_{i}_{yr}",
            "name": p_name,
            "college": row.get("college") or "NCAA",
            "year": yr,
            "grade": round(c_grade, 1),
            "archetype": "Boundary X",
            "p_x": 0.5, "p_z": 0.3, "p_slot": 0.2,
            "base_p10": p10, "base_p50": p50, "base_p90": p90,
            "p10": p10, "p50": p50, "p90": p90,
            "prob_starter": 0.30, "prob_boom": 0.15, "prob_bust": 0.45,
            "nfl_team": row.get("team") or "NFL",
            "mock_team": None,
        })

prospects.sort(key=lambda x: x["grade"], reverse=True)
print(f"Total compiled prospects for web app: {len(prospects)}")

prospects_json = json.dumps(prospects)

# 32 NFL Teams with verified hex color codes and offensive signatures
NFL_TEAMS = {
    "KC": {"name": "Chiefs", "city": "Kansas City", "color": "#E31837", "alt": "#FFB81C", "qb": "Patrick Mahomes", "proe": 0.95, "qb_epa": 1.35, "vacated": 0.32, "style": "Air Raid"},
    "DAL": {"name": "Cowboys", "city": "Dallas", "color": "#003594", "alt": "#869397", "qb": "Dak Prescott", "proe": 0.85, "qb_epa": 1.15, "vacated": 0.22, "style": "Pass Heavy"},
    "BUF": {"name": "Bills", "city": "Buffalo", "color": "#00338D", "alt": "#C60C30", "qb": "Josh Allen", "proe": 0.65, "qb_epa": 1.25, "vacated": 0.44, "style": "Vertical Pass"},
    "CIN": {"name": "Bengals", "city": "Cincinnati", "color": "#FB4F14", "alt": "#000000", "qb": "Joe Burrow", "proe": 0.65, "qb_epa": 0.85, "vacated": 0.24, "style": "Precision Spread"},
    "DET": {"name": "Lions", "city": "Detroit", "color": "#0076B6", "alt": "#B0B7BC", "qb": "Jared Goff", "proe": 0.20, "qb_epa": 0.80, "vacated": 0.22, "style": "Dome Play-Action"},
    "MIN": {"name": "Vikings", "city": "Minnesota", "color": "#4F2683", "alt": "#FFC62F", "qb": "Sam Darnold", "proe": 0.75, "qb_epa": 0.45, "vacated": 0.26, "style": "High PROE"},
    "HOU": {"name": "Texans", "city": "Houston", "color": "#03202F", "alt": "#A71930", "qb": "C.J. Stroud", "proe": 0.45, "qb_epa": 0.90, "vacated": 0.24, "style": "Zone Motion"},
    "MIA": {"name": "Dolphins", "city": "Miami", "color": "#008E97", "alt": "#FC4C02", "qb": "Tua Tagovailoa", "proe": 0.35, "qb_epa": 0.60, "vacated": 0.20, "style": "Motion in Space"},
    "PHI": {"name": "Eagles", "city": "Philadelphia", "color": "#004C54", "alt": "#A5ACAF", "qb": "Jalen Hurts", "proe": -0.35, "qb_epa": 0.50, "vacated": 0.18, "style": "RPO Option"},
    "SF": {"name": "49ers", "city": "San Francisco", "color": "#AA0000", "alt": "#B3995D", "qb": "Brock Purdy", "proe": -1.10, "qb_epa": 0.95, "vacated": 0.20, "style": "Run-First YAC"},
    "CHI": {"name": "Bears", "city": "Chicago", "color": "#0B162A", "alt": "#C83803", "qb": "Caleb Williams", "proe": 0.10, "qb_epa": 0.10, "vacated": 0.18, "style": "Spread Option"},
    "LAC": {"name": "Chargers", "city": "Los Angeles", "color": "#0080C6", "alt": "#FFC20E", "qb": "Justin Herbert", "proe": -0.40, "qb_epa": 0.65, "vacated": 0.48, "style": "Power Run / Shot"},
    "NYG": {"name": "Giants", "city": "New York", "color": "#0B2265", "alt": "#A71930", "qb": "Daniel Jones", "proe": -0.25, "qb_epa": -0.50, "vacated": 0.44, "style": "Open Target Room"},
    "CAR": {"name": "Panthers", "city": "Carolina", "color": "#0085CA", "alt": "#101820", "qb": "Bryce Young", "proe": -0.10, "qb_epa": -0.65, "vacated": 0.40, "style": "Short Option"},
    "NE": {"name": "Patriots", "city": "New England", "color": "#002244", "alt": "#C60C30", "qb": "Drake Maye", "proe": -0.25, "qb_epa": -0.45, "vacated": 0.38, "style": "Rebuild Pass"},
    "LAR": {"name": "Rams", "city": "Los Angeles", "color": "#003594", "alt": "#FFD100", "qb": "Matthew Stafford", "proe": 0.20, "qb_epa": 0.55, "vacated": 0.24, "style": "McVay Play-Action"},
    "BAL": {"name": "Ravens", "city": "Baltimore", "color": "#241773", "alt": "#9E7C0C", "qb": "Lamar Jackson", "proe": -0.75, "qb_epa": 0.85, "vacated": 0.22, "style": "Heavy Pistol"},
    "GB": {"name": "Packers", "city": "Green Bay", "color": "#203731", "alt": "#FFB612", "qb": "Jordan Love", "proe": 0.30, "qb_epa": 0.70, "vacated": 0.20, "style": "LaFleur System"},
    "WAS": {"name": "Commanders", "city": "Washington", "color": "#5A1414", "alt": "#FFB612", "qb": "Jayden Daniels", "proe": 0.15, "qb_epa": 0.70, "vacated": 0.32, "style": "Air Raid Spread"},
    "TB": {"name": "Buccaneers", "city": "Tampa Bay", "color": "#D50A0A", "alt": "#34302B", "qb": "Baker Mayfield", "proe": 0.40, "qb_epa": 0.55, "vacated": 0.20, "style": "Downfield Spread"},
    "ATL": {"name": "Falcons", "city": "Atlanta", "color": "#A71930", "alt": "#000000", "qb": "Kirk Cousins", "proe": 0.20, "qb_epa": 0.40, "vacated": 0.25, "style": "Quick Passing"},
    "SEA": {"name": "Seahawks", "city": "Seattle", "color": "#002244", "alt": "#69BE28", "qb": "Geno Smith", "proe": 0.45, "qb_epa": 0.45, "vacated": 0.18, "style": "Grubb Air"},
    "PIT": {"name": "Steelers", "city": "Pittsburgh", "color": "#FFB612", "alt": "#101820", "qb": "Russell Wilson", "proe": -0.65, "qb_epa": -0.10, "vacated": 0.34, "style": "Run-First Shot"},
    "IND": {"name": "Colts", "city": "Indianapolis", "color": "#002C5F", "alt": "#A2AAAD", "qb": "Anthony Richardson", "proe": -0.35, "qb_epa": -0.10, "vacated": 0.22, "style": "Vertical RPO"},
    "JAX": {"name": "Jaguars", "city": "Jacksonville", "color": "#006778", "alt": "#D7A22A", "qb": "Trevor Lawrence", "proe": 0.30, "qb_epa": 0.35, "vacated": 0.34, "style": "Pederson Spread"},
    "NYJ": {"name": "Jets", "city": "New York", "color": "#125740", "alt": "#000000", "qb": "Aaron Rodgers", "proe": 0.40, "qb_epa": 0.45, "vacated": 0.24, "style": "West Coast"},
    "ARI": {"name": "Cardinals", "city": "Arizona", "color": "#97233F", "alt": "#000000", "qb": "Kyler Murray", "proe": -0.10, "qb_epa": 0.35, "vacated": 0.36, "style": "Shotgun Run/Pass"},
    "NO": {"name": "Saints", "city": "New Orleans", "color": "#D3BC8D", "alt": "#101820", "qb": "Derek Carr", "proe": -0.15, "qb_epa": 0.30, "vacated": 0.24, "style": "Boot / Play-Action"},
    "DEN": {"name": "Broncos", "city": "Denver", "color": "#FB4F14", "alt": "#002244", "qb": "Bo Nix", "proe": 0.20, "qb_epa": -0.10, "vacated": 0.35, "style": "Payton Quick Pass"},
    "LV": {"name": "Raiders", "city": "Las Vegas", "color": "#A5ACAF", "alt": "#000000", "qb": "Gardner Minshew", "proe": 0.10, "qb_epa": -0.45, "vacated": 0.40, "style": "Spread Pass"},
    "CLE": {"name": "Browns", "city": "Cleveland", "color": "#311D00", "alt": "#FF3C00", "qb": "Deshaun Watson", "proe": 0.30, "qb_epa": -0.40, "vacated": 0.28, "style": "Spread Option"},
    "TEN": {"name": "Titans", "city": "Tennessee", "color": "#0C2340", "alt": "#4B92DB", "qb": "Will Levis", "proe": 0.10, "qb_epa": -0.35, "vacated": 0.30, "style": "Callahan Spread"},
}

teams_json = json.dumps(NFL_TEAMS)

html_code = f"""<!DOCTYPE html>
<html lang="en" class="h-full bg-zinc-950 text-zinc-100">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Board · NFL Wide Receiver Prospect Rankings</title>
  
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
    .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: #09090b; }}
    ::-webkit-scrollbar-thumb {{ background: #27272a; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #3f3f46; }}

    .prospect-row {{ transition: background-color 0.15s ease, border-color 0.15s ease; }}
    .prospect-row:hover {{ background-color: rgba(39, 39, 42, 0.4); }}
    
    .team-badge {{ transition: transform 0.15s ease, border-color 0.15s ease; }}
    .team-badge:hover {{ transform: scale(1.02); }}
  </style>
</head>
<body class="min-h-full flex flex-col bg-zinc-950 text-zinc-100 antialiased selection:bg-zinc-800 selection:text-white pb-16">

  <!-- Top App Navigation Bar -->
  <header class="border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur sticky top-0 z-30">
    <div class="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
      
      <!-- App Brand -->
      <div class="flex items-center space-x-3">
        <div class="w-8 h-8 rounded-lg bg-zinc-100 text-zinc-950 flex items-center justify-center font-black text-sm tracking-tighter shadow-sm">
          WR
        </div>
        <div>
          <h1 class="text-base font-bold tracking-tight text-white leading-none">Board</h1>
          <span class="text-[11px] font-medium text-zinc-400">Pure College Rankings · Zero Capital Bias</span>
        </div>
      </div>

      <!-- Action Status / Mock Active Indicator -->
      <div class="flex items-center space-x-3">
        <div id="mock-counter-container" class="hidden items-center space-x-2 bg-zinc-900 border border-zinc-800 rounded-full px-3 py-1 text-xs">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span id="mock-counter-text" class="text-zinc-300 font-medium">1 Mock Landing</span>
          <button id="btn-reset-all-mocks" class="text-zinc-400 hover:text-white transition ml-1 font-semibold">Reset</button>
        </div>
        <button id="btn-clear-search" class="text-xs text-zinc-400 hover:text-white transition py-1 px-2.5 rounded-lg border border-zinc-800 hover:border-zinc-700 bg-zinc-900">
          Reset Filters
        </button>
      </div>

    </div>
  </header>

  <!-- Filter & Rank Sorting Strip -->
  <section class="max-w-5xl mx-auto px-4 sm:px-6 pt-6 pb-2 w-full">
    
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-zinc-800/60">
      
      <!-- Sort By Selector -->
      <div class="flex items-center space-x-2">
        <span class="text-xs font-semibold text-zinc-400 uppercase tracking-wider flex-shrink-0">Sort By</span>
        <div class="relative">
          <select id="select-sort" class="bg-zinc-900 border border-zinc-800 hover:border-zinc-700 text-white text-xs font-semibold rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-zinc-600 appearance-none pr-8 cursor-pointer transition">
            <option value="grade">Overall Rank (College Grade)</option>
            <option value="floor">Highest Floor (p10 Rank)</option>
            <option value="median">Highest Median (p50 Rank)</option>
            <option value="ceiling">Highest Ceiling (p90 Rank)</option>
            <option value="boom">WR1 Boom Upside</option>
            <option value="safe">Lowest Bust Risk</option>
          </select>
          <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-zinc-400">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
          </div>
        </div>
      </div>

      <!-- Class Filter Chips & Search -->
      <div class="flex items-center space-x-2.5">
        <div class="flex items-center space-x-1 bg-zinc-900 p-1 rounded-lg border border-zinc-800/80 text-xs">
          <button data-class="all" class="class-pill px-2.5 py-1 rounded-md font-semibold bg-zinc-100 text-zinc-950 transition">All</button>
          <button data-class="2025" class="class-pill px-2.5 py-1 rounded-md font-medium text-zinc-400 hover:text-white transition">2025</button>
          <button data-class="2024" class="class-pill px-2.5 py-1 rounded-md font-medium text-zinc-400 hover:text-white transition">2024</button>
          <button data-class="2023" class="class-pill px-2.5 py-1 rounded-md font-medium text-zinc-400 hover:text-white transition">2023</button>
          <button data-class="2022" class="class-pill px-2.5 py-1 rounded-md font-medium text-zinc-400 hover:text-white transition">2022</button>
        </div>

        <div class="relative w-40 sm:w-48">
          <input id="input-search" type="text" placeholder="Search..."
                 class="w-full bg-zinc-900 border border-zinc-800 text-white text-xs rounded-lg pl-7 pr-2.5 py-1.5 placeholder-zinc-500 focus:outline-none focus:border-zinc-700 transition">
          <svg class="w-3.5 h-3.5 text-zinc-500 absolute left-2 top-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
        </div>
      </div>

    </div>

    <!-- Active List Metadata -->
    <div class="flex items-center justify-between pt-3 pb-1 text-xs text-zinc-500">
      <div>Rankings (<span id="count-visible" class="text-zinc-300 font-medium">0</span> prospects)</div>
      <div id="sort-description" class="text-zinc-400 font-medium">Ranked by pre-draft college translation grade</div>
    </div>

  </section>

  <!-- Pure Prospect Rankings Feed -->
  <main class="max-w-5xl mx-auto px-4 sm:px-6 w-full flex-1">
    <div id="rankings-container" class="divide-y divide-zinc-800/60 border-t border-b border-zinc-800/80">
      <!-- Injected by JavaScript -->
    </div>

    <div id="empty-state" class="hidden py-20 text-center text-zinc-500">
      <p class="text-sm">No prospects found matching your search</p>
    </div>
  </main>

  <!-- ========================================================================= -->
  <!-- HYPOTHETICAL MOCK TEAM SLIDE-OVER DRAWER -->
  <!-- ========================================================================= -->
  <div id="drawer-backdrop" class="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 hidden transition-opacity opacity-0"></div>

  <div id="drawer-sheet" class="fixed inset-y-0 right-0 max-w-lg w-full bg-zinc-950 border-l border-zinc-800 z-50 transform translate-x-full transition-transform duration-200 ease-out flex flex-col shadow-2xl">
    
    <!-- Drawer Header -->
    <div class="p-6 border-b border-zinc-800 flex items-center justify-between">
      <div>
        <span class="text-[11px] font-bold uppercase tracking-wider text-zinc-400">Mock Team Fit</span>
        <h2 id="drawer-prospect-name" class="text-2xl font-black text-white tracking-tight mt-0.5">Marvin Harrison Jr.</h2>
        <div class="flex items-center space-x-2 mt-1">
          <span id="drawer-prospect-meta" class="text-xs text-zinc-400">Ohio State · Class of 2024</span>
          <span id="drawer-prospect-arch" class="text-[10px] font-semibold px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">Boundary X</span>
        </div>
      </div>
      <button id="btn-close-drawer" class="w-8 h-8 rounded-lg bg-zinc-900 border border-zinc-800 hover:border-zinc-700 flex items-center justify-center text-zinc-400 hover:text-white transition">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
      </button>
    </div>

    <!-- Active Mock Team Impact Card -->
    <div class="p-6 bg-zinc-900/60 border-b border-zinc-800">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-3">
          <div id="drawer-team-logo" class="w-10 h-10 rounded-xl flex items-center justify-center font-black text-sm text-white shadow" style="background:#E31837;">
            KC
          </div>
          <div>
            <div class="text-[11px] text-zinc-400 font-medium">Selected Landing Spot</div>
            <h3 id="drawer-team-title" class="text-lg font-bold text-white leading-tight">Kansas City Chiefs</h3>
          </div>
        </div>

        <div class="text-right">
          <div class="text-[11px] text-zinc-400 font-medium">Projected Shift</div>
          <div id="drawer-team-shift" class="text-lg font-black text-emerald-400 font-mono">+480 yds</div>
        </div>
      </div>

      <!-- 3 Pillars Breakdown -->
      <div class="grid grid-cols-3 gap-2 mt-4 pt-4 border-t border-zinc-800/60 text-xs">
        <div class="bg-zinc-950 p-2.5 rounded-lg border border-zinc-800">
          <span class="text-[10px] text-zinc-400 font-semibold block uppercase">QB &amp; Volume</span>
          <span id="drawer-pillar-qb" class="font-bold text-white block mt-0.5">+24.5%</span>
          <span id="drawer-qb-name" class="text-[10px] text-zinc-400 truncate block">P. Mahomes</span>
        </div>
        <div class="bg-zinc-950 p-2.5 rounded-lg border border-zinc-800">
          <span class="text-[10px] text-zinc-400 font-semibold block uppercase">Vacancy Runway</span>
          <span id="drawer-pillar-vac" class="font-bold text-white block mt-0.5">+14.0%</span>
          <span id="drawer-vac-val" class="text-[10px] text-zinc-400 block">32% Vacated</span>
        </div>
        <div class="bg-zinc-950 p-2.5 rounded-lg border border-zinc-800">
          <span class="text-[10px] text-zinc-400 font-semibold block uppercase">Role Fit</span>
          <span id="drawer-pillar-role" class="font-bold text-emerald-400 block mt-0.5">Synergy</span>
          <span id="drawer-role-desc" class="text-[10px] text-zinc-400 truncate block">Kelce Pairing</span>
        </div>
      </div>

      <div class="flex items-center space-x-2 mt-4">
        <button id="btn-apply-mock" class="flex-1 py-2 rounded-lg bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-bold transition shadow">
          Apply Mock Team to Board
        </button>
        <button id="btn-remove-mock" class="py-2 px-3 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs font-semibold transition">
          Clear
        </button>
      </div>
    </div>

    <!-- NFL Teams Grid Picker -->
    <div class="p-6 flex-1 overflow-y-auto">
      <div class="flex items-center justify-between pb-2 mb-3">
        <span class="text-xs font-bold text-zinc-400 uppercase tracking-wider">Choose NFL Team</span>
        <span class="text-[11px] text-zinc-400">Click to preview fit</span>
      </div>

      <div id="teams-grid" class="grid grid-cols-2 gap-2.5">
        <!-- Injected by JavaScript -->
      </div>
    </div>

  </div>

  <!-- Embedded Client Logic -->
  <script>
    const PROSPECTS = {prospects_json};
    const TEAMS = {teams_json};

    let activeSort = 'grade';
    let activeClass = 'all';
    let searchQuery = '';
    let selectedProspectId = null;
    let previewTeamCode = 'KC';

    const rankingsContainer = document.getElementById('rankings-container');
    const emptyState = document.getElementById('empty-state');
    const countVisible = document.getElementById('count-visible');
    const sortDescription = document.getElementById('sort-description');
    const selectSort = document.getElementById('select-sort');
    const inputSearch = document.getElementById('input-search');
    const btnClearSearch = document.getElementById('btn-clear-search');
    const classPills = document.querySelectorAll('.class-pill');

    const drawerBackdrop = document.getElementById('drawer-backdrop');
    const drawerSheet = document.getElementById('drawer-sheet');
    const btnCloseDrawer = document.getElementById('btn-close-drawer');
    const teamsGrid = document.getElementById('teams-grid');
    const btnApplyMock = document.getElementById('btn-apply-mock');
    const btnRemoveMock = document.getElementById('btn-remove-mock');

    const mockCounterContainer = document.getElementById('mock-counter-container');
    const mockCounterText = document.getElementById('mock-counter-text');
    const btnResetAllMocks = document.getElementById('btn-reset-all-mocks');

    // Sort descriptions
    const SORT_LABELS = {{
      'grade': 'Ranked by pre-draft college translation grade',
      'floor': 'Ranked by highest projected floor (p10 percentile safety)',
      'median': 'Ranked by highest expected median peak outcome (p50)',
      'ceiling': 'Ranked by highest potential ceiling (p90 WR1 upside)',
      'boom': 'Ranked by probability of WR1 Boom (≥2,500 yards)',
      'safe': 'Ranked by lowest bust/flop risk (<500 yards)'
    }};

    // Team Fit Math for Mock Simulation
    function computeMockFit(prospect, teamCode) {{
      const team = TEAMS[teamCode] || TEAMS['KC'];
      const c_grade = prospect.grade || 75.0;

      const raw_qb_vol = (team.proe * 0.075) + (team.qb_epa * 0.090);
      const raw_vacancy = (team.vacated - 0.22) * 0.50;

      let role_bonus = 0.0;
      let role_label = 'Neutral Room';
      if (team.vacated >= 0.35) {{
        role_bonus += 0.06;
        role_label = 'Open Runway';
      }} else if (teamCode === 'DAL' || teamCode === 'KC' || teamCode === 'MIN') {{
        role_bonus += 0.04;
        role_label = 'Complementary';
      }} else if (teamCode === 'SF' || teamCode === 'CHI') {{
        role_bonus -= 0.03;
        role_label = 'Crowded Room';
      }}

      const talent_factor = Math.max(0.65, Math.min(1.45, Math.pow(c_grade / 78.0, 1.35)));
      const delta_qb = raw_qb_vol * talent_factor;
      const delta_vac = raw_vacancy * talent_factor;

      const fit_mult = Math.max(0.72, Math.min(1.35, 1.0 + delta_qb + delta_vac + role_bonus));
      const new_p50 = Math.round(prospect.base_p50 * fit_mult);
      const new_p90 = Math.round(prospect.base_p90 * Math.pow(fit_mult, 1.25));
      const new_p10 = Math.round(prospect.base_p10 * Math.max(0.60, Math.min(1.40, 1.0 + role_bonus)));
      const shift_yds = new_p50 - prospect.base_p50;

      return {{
        team: team,
        teamCode: teamCode,
        fit_mult: fit_mult,
        delta_qb: delta_qb,
        delta_vac: delta_vac,
        role_label: role_label,
        shift_yds: shift_yds,
        new_p50: new_p50,
        new_p90: new_p90,
        new_p10: new_p10
      }};
    }}

    function getSortedProspects() {{
      let list = PROSPECTS.filter(p => {{
        const matchesClass = activeClass === 'all' || String(p.year) === activeClass;
        const matchesSearch = !searchQuery || 
          p.name.toLowerCase().includes(searchQuery) || 
          p.college.toLowerCase().includes(searchQuery) ||
          p.archetype.toLowerCase().includes(searchQuery);
        return matchesClass && matchesSearch;
      }});

      list = [...list];

      list.sort((a, b) => {{
        const a_p10 = a.mock_fit ? a.mock_fit.new_p10 : a.p10;
        const b_p10 = b.mock_fit ? b.mock_fit.new_p10 : b.p10;

        const a_p50 = a.mock_fit ? a.mock_fit.new_p50 : a.p50;
        const b_p50 = b.mock_fit ? b.mock_fit.new_p50 : b.p50;

        const a_p90 = a.mock_fit ? a.mock_fit.new_p90 : a.p90;
        const b_p90 = b.mock_fit ? b.mock_fit.new_p90 : b.p90;

        if (activeSort === 'grade') return b.grade - a.grade;
        if (activeSort === 'floor') return b_p10 - a_p10;
        if (activeSort === 'median') return b_p50 - a_p50;
        if (activeSort === 'ceiling') return b_p90 - a_p90;
        if (activeSort === 'boom') return b.prob_boom - a.prob_boom;
        if (activeSort === 'safe') return a.prob_bust - b.prob_bust;
        return 0;
      }});

      return list;
    }}

    function renderRankings() {{
      const list = getSortedProspects();
      countVisible.innerText = list.length;
      sortDescription.innerText = SORT_LABELS[activeSort] || SORT_LABELS['grade'];

      // Update mock counter
      const mockedCount = PROSPECTS.filter(p => p.mock_team).length;
      if (mockedCount > 0) {{
        mockCounterContainer.classList.remove('hidden');
        mockCounterContainer.classList.add('flex');
        mockCounterText.innerText = `${{mockedCount}} Mock Landing${{mockedCount > 1 ? 's' : ''}} Active`;
      }} else {{
        mockCounterContainer.classList.add('hidden');
        mockCounterContainer.classList.remove('flex');
      }}

      if (list.length === 0) {{
        rankingsContainer.innerHTML = '';
        emptyState.classList.remove('hidden');
        return;
      }}
      emptyState.classList.add('hidden');

      rankingsContainer.innerHTML = list.map((p, idx) => {{
        const rankNum = String(idx + 1).padStart(2, '0');
        const hasMock = p.mock_team !== null && p.mock_team !== undefined;
        const mock = p.mock_fit;

        let primaryBadge = `<span class="text-xs font-mono font-bold text-zinc-300">Grade ${{p.grade}}</span>`;
        if (activeSort === 'floor') {{
          primaryBadge = `<span class="text-xs font-mono font-bold text-zinc-300">Floor: #${{idx+1}} (${{p.p10}} yds)</span>`;
        }} else if (activeSort === 'ceiling') {{
          primaryBadge = `<span class="text-xs font-mono font-bold text-zinc-300">Ceiling: #${{idx+1}} (${{p.p90}} yds)</span>`;
        }} else if (activeSort === 'median') {{
          primaryBadge = `<span class="text-xs font-mono font-bold text-zinc-300">Median: #${{idx+1}} (${{p.p50}} yds)</span>`;
        }} else if (activeSort === 'boom') {{
          const bPct = Math.round(p.prob_boom * 100);
          primaryBadge = `<span class="text-xs font-mono font-bold text-emerald-400">Boom ${{bPct}}%</span>`;
        }} else if (activeSort === 'safe') {{
          const sPct = Math.round(p.prob_bust * 100);
          primaryBadge = `<span class="text-xs font-mono font-bold text-zinc-300">Bust ${{sPct}}%</span>`;
        }}

        const archStyle = p.archetype === 'Boundary X' 
          ? 'bg-zinc-800/80 text-zinc-200 border-zinc-700' 
          : (p.archetype === 'Flanker Z' ? 'bg-zinc-900 text-zinc-300 border-zinc-800' : 'bg-zinc-900 text-zinc-400 border-zinc-800');

        return `
          <div class="prospect-row py-4 px-3 sm:px-4 flex items-center justify-between cursor-pointer group" onclick="openDrawer('${{p.id}}')">
            
            <!-- Left: Rank Number + Name & College -->
            <div class="flex items-center space-x-4 sm:space-x-6 min-w-0">
              <span class="text-2xl sm:text-3xl font-black text-zinc-400 font-mono tracking-tighter w-8 sm:w-10 text-left flex-shrink-0 group-hover:text-white transition">
                ${{rankNum}}
              </span>

              <div class="min-w-0">
                <div class="flex items-center space-x-2.5">
                  <h3 class="text-base sm:text-lg font-bold text-white tracking-tight truncate group-hover:text-zinc-100 transition">
                    ${{p.name}}
                  </h3>
                  <span class="text-[10px] font-semibold px-2 py-0.5 rounded border ${{archStyle}} flex-shrink-0">
                    ${{p.archetype}}
                  </span>
                </div>
                
                <div class="flex items-center space-x-2 text-xs text-zinc-400 mt-0.5">
                  <span class="font-medium">${{p.college}}</span>
                  <span>·</span>
                  <span>Class of ${{p.year}}</span>
                </div>
              </div>
            </div>

            <!-- Right: Sort Metric + Mock Team Badge -->
            <div class="flex items-center space-x-4 flex-shrink-0">
              <div class="text-right hidden sm:block">
                ${{primaryBadge}}
              </div>

              ${{hasMock ? `
                <div class="flex items-center space-x-2 bg-zinc-900 border border-zinc-700 rounded-lg px-2.5 py-1.5 shadow-sm">
                  <span class="w-5 h-5 rounded flex items-center justify-center font-black text-[10px] text-white" style="background:${{mock.team.color}}">${{p.mock_team}}</span>
                  <span class="text-xs font-bold font-mono text-emerald-400">${{mock.shift_yds >= 0 ? '+' : ''}}${{mock.shift_yds}}</span>
                </div>
              ` : `
                <button class="text-xs font-semibold px-2.5 py-1.5 rounded-lg border border-zinc-800/80 bg-zinc-900/60 hover:bg-zinc-800 text-zinc-400 hover:text-white transition flex items-center space-x-1"
                        onclick="event.stopPropagation(); openDrawer('${{p.id}}')">
                  <span>+ Mock</span>
                </button>
              `}}
            </div>

          </div>
        `;
      }}).join('');
    }}

    // =========================================================================
    // DRAWER / HYPOTHETICAL TEAM INTERACTION
    // =========================================================================

    function openDrawer(prospectId) {{
      selectedProspectId = prospectId;
      const prospect = PROSPECTS.find(p => p.id === prospectId);
      if (!prospect) return;

      previewTeamCode = prospect.mock_team || 'KC';

      document.getElementById('drawer-prospect-name').innerText = prospect.name;
      document.getElementById('drawer-prospect-meta').innerText = `${{prospect.college}} · Class of ${{prospect.year}} · Grade ${{prospect.grade}}`;
      document.getElementById('drawer-prospect-arch').innerText = prospect.archetype;

      updateDrawerTeamPreview();
      renderTeamsGrid();

      drawerBackdrop.classList.remove('hidden');
      setTimeout(() => {{
        drawerBackdrop.classList.remove('opacity-0');
        drawerSheet.classList.remove('translate-x-full');
      }}, 10);
    }}

    function closeDrawer() {{
      drawerBackdrop.classList.add('opacity-0');
      drawerSheet.classList.add('translate-x-full');
      setTimeout(() => {{
        drawerBackdrop.classList.add('hidden');
      }}, 200);
    }}

    function updateDrawerTeamPreview() {{
      const prospect = PROSPECTS.find(p => p.id === selectedProspectId);
      if (!prospect) return;

      const fit = computeMockFit(prospect, previewTeamCode);
      const team = fit.team;

      document.getElementById('drawer-team-logo').innerText = previewTeamCode;
      document.getElementById('drawer-team-logo').style.backgroundColor = team.color;
      document.getElementById('drawer-team-title').innerText = `${{team.city}} ${{team.name}}`;

      const shiftEl = document.getElementById('drawer-team-shift');
      shiftEl.innerText = `${{fit.shift_yds >= 0 ? '+' : ''}}${{fit.shift_yds}} yds`;
      shiftEl.className = fit.shift_yds >= 0 ? 'text-lg font-black text-emerald-400 font-mono' : 'text-lg font-black text-rose-400 font-mono';

      document.getElementById('drawer-pillar-qb').innerText = `${{fit.delta_qb >= 0 ? '+' : ''}}${{(fit.delta_qb * 100).toFixed(1)}}%`;
      document.getElementById('drawer-qb-name').innerText = team.qb;

      document.getElementById('drawer-pillar-vac').innerText = `${{fit.delta_vac >= 0 ? '+' : ''}}${{(fit.delta_vac * 100).toFixed(1)}}%`;
      document.getElementById('drawer-vac-val').innerText = `${{Math.round(team.vacated * 100)}}% Vacated`;

      document.getElementById('drawer-pillar-role').innerText = fit.role_label;
      document.getElementById('drawer-role-desc').innerText = team.style;
    }}

    function renderTeamsGrid() {{
      teamsGrid.innerHTML = Object.entries(TEAMS).map(([code, team]) => {{
        const isSelected = code === previewTeamCode;
        return `
          <div class="team-badge p-2.5 rounded-xl border cursor-pointer flex items-center justify-between ${{isSelected ? 'bg-zinc-900 border-zinc-600 shadow' : 'bg-zinc-950 border-zinc-800 hover:border-zinc-700'}}"
               onclick="selectPreviewTeam('${{code}}')">
            <div class="flex items-center space-x-2.5 min-w-0">
              <span class="w-6 h-6 rounded-md flex items-center justify-center font-black text-[11px] text-white flex-shrink-0" style="background:${{team.color}}">
                ${{code}}
              </span>
              <div class="min-w-0">
                <div class="text-xs font-bold text-white truncate">${{team.name}}</div>
                <div class="text-[10px] text-zinc-400 truncate">${{team.style}}</div>
              </div>
            </div>
            <span class="text-[10px] font-mono text-zinc-400 font-semibold">${{team.proe >= 0 ? '+' : ''}}${{team.proe}}</span>
          </div>
        `;
      }}).join('');
    }}

    function selectPreviewTeam(code) {{
      previewTeamCode = code;
      updateDrawerTeamPreview();
      renderTeamsGrid();
    }}

    // Apply Mock Team
    btnApplyMock.addEventListener('click', () => {{
      const prospect = PROSPECTS.find(p => p.id === selectedProspectId);
      if (prospect) {{
        prospect.mock_team = previewTeamCode;
        prospect.mock_fit = computeMockFit(prospect, previewTeamCode);
        closeDrawer();
        renderRankings();
      }}
    }});

    // Remove Mock Team
    btnRemoveMock.addEventListener('click', () => {{
      const prospect = PROSPECTS.find(p => p.id === selectedProspectId);
      if (prospect) {{
        prospect.mock_team = null;
        prospect.mock_fit = null;
        closeDrawer();
        renderRankings();
      }}
    }});

    // Reset All Mocks
    btnResetAllMocks.addEventListener('click', () => {{
      PROSPECTS.forEach(p => {{
        p.mock_team = null;
        p.mock_fit = null;
      }});
      renderRankings();
    }});

    // Drawer close buttons
    btnCloseDrawer.addEventListener('click', closeDrawer);
    drawerBackdrop.addEventListener('click', closeDrawer);

    // Event Listeners for Filters
    selectSort.addEventListener('change', (e) => {{
      activeSort = e.target.value;
      renderRankings();
    }});

    inputSearch.addEventListener('input', (e) => {{
      searchQuery = e.target.value.toLowerCase().trim();
      renderRankings();
    }});

    btnClearSearch.addEventListener('click', () => {{
      inputSearch.value = '';
      searchQuery = '';
      activeClass = 'all';
      selectSort.value = 'grade';
      activeSort = 'grade';
      classPills.forEach(p => {{
        p.className = 'class-pill px-2.5 py-1 rounded-md font-medium text-zinc-400 hover:text-white transition';
      }});
      document.querySelector('[data-class="all"]').className = 'class-pill px-2.5 py-1 rounded-md font-semibold bg-zinc-100 text-zinc-950 transition';
      renderRankings();
    }});

    classPills.forEach(pill => {{
      pill.addEventListener('click', () => {{
        classPills.forEach(p => {{
          p.className = 'class-pill px-2.5 py-1 rounded-md font-medium text-zinc-400 hover:text-white transition';
        }});
        pill.className = 'class-pill px-2.5 py-1 rounded-md font-semibold bg-zinc-100 text-zinc-950 transition';
        activeClass = pill.getAttribute('data-class');
        renderRankings();
      }});
    }});

    // Initial render
    renderRankings();
  </script>
</body>
</html>
"""

out_dir = Path("artifacts/file_generation/ttl=63d/output")
inter_dir = Path("artifacts/file_generation/ttl=63d/intermediate")
out_dir.mkdir(parents=True, exist_ok=True)
inter_dir.mkdir(parents=True, exist_ok=True)

out_file = out_dir / "prospect_rankings_app.html"
inter_file = inter_dir / "prospect_rankings_app.html"

with open(out_file, "w", encoding="utf-8") as f:
    f.write(html_code)

with open(inter_file, "w", encoding="utf-8") as f:
    f.write(html_code)

with open("prospect_rankings_app.html", "w", encoding="utf-8") as f:
    f.write(html_code)

print(f"Generated standalone web app at {out_file.resolve()}")