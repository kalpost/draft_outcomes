#!/usr/bin/env python3
"""
Dynamic Year-Specific NFL WR Prospect Board Builder (`build_rankings_app.py`)
=============================================================================
Reads output parquets written by `run_unified_pipeline.py` and emits a self-contained
HTML application (`prospect_rankings_app.html` / `index.html`) with:
  - Every scored draft class in the dataset (`DRAFT_START_YEAR`–`DRAFT_END_YEAR`, plus any
    `SCORE_UNDRAFTED_CLASS_YEAR` pre-draft college prospects).
  - Intuitive Conditional Outcome Framing (replacing the misleading raw `p50` focal point):
      * Full 80% Outcome Range (`p10 – p90`)
      * "If He Hits" (`hit_yds`): Expected 3-year NFL yardage conditional on panning out as an
        NFL starter (`0.42 * p50 + 0.58 * p90`), so a high-upside prospect like Tetairoa McMillan
        shows his true ~2,420-yard starter expectation (3,160 ceiling) rather than an unconditional
        1,333-yard median that blends bust and starter regimes together.
      * "If He Misses" (`miss_yds`): Downside/bust expectation (`p10–p20` floor regime) if injuries
        or role issues derail his rookie contract.
      * Hit Odds (`P(Starter >= 1,500)` & `P(Boom >= 2,500)`) vs. Miss Odds (`P(Bust < 500)`).
  - Toggle: Team-Neutral (Stage 1.5) vs. Drafted-Team Fit (Stage 2).
  - Interactive Year-Specific Mock Landing-Spot Simulator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

from config import (
    COMBINE_RAW_PATH,
    COMPLETED_NFL_SEASON,
    NFL_STATS_PLAYER_PATH,
    PROCESSED_DIR,
    ROOT_DIR,
    STAGE1_RANKINGS_PATH,
    STAGE2_RANKINGS_PATH,
    RECENT_PROSPECTS_PATH,
    norm_nfl_team,
    parse_height_inches,
)

# Any draft class up through `COMPLETED_NFL_SEASON - 2` has logged 3 NFL seasons (Y1, Y2, Y3)
DEFAULT_LAST_COMPLETE_CLASS: int = COMPLETED_NFL_SEASON - 2

TEAM_COLORS: dict[str, str] = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#241773", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#0B162A", "CIN": "#FB4F14", "CLE": "#311D00",
    "DAL": "#003594", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#203731",
    "HOU": "#03202F", "IND": "#002C5F", "JAX": "#006778", "KC": "#E31837",
    "LAC": "#0080C6", "LAR": "#003594", "LV": "#A5ACAF", "MIA": "#008E97",
    "MIN": "#4F2683", "NE": "#002244", "NO": "#D3BC8D", "NYG": "#0B2265",
    "NYJ": "#125740", "PHI": "#004C54", "PIT": "#FFB612", "SEA": "#002244",
    "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#0C2340", "WAS": "#5A1414",
}

TEAM_NAMES: dict[str, tuple[str, str]] = {
    "ARI": ("Cardinals", "Arizona"), "ATL": ("Falcons", "Atlanta"),
    "BAL": ("Ravens", "Baltimore"), "BUF": ("Bills", "Buffalo"),
    "CAR": ("Panthers", "Carolina"), "CHI": ("Bears", "Chicago"),
    "CIN": ("Bengals", "Cincinnati"), "CLE": ("Browns", "Cleveland"),
    "DAL": ("Cowboys", "Dallas"), "DEN": ("Broncos", "Denver"),
    "DET": ("Lions", "Detroit"), "GB": ("Packers", "Green Bay"),
    "HOU": ("Texans", "Houston"), "IND": ("Colts", "Indianapolis"),
    "JAX": ("Jaguars", "Jacksonville"), "KC": ("Chiefs", "Kansas City"),
    "LAC": ("Chargers", "Los Angeles"), "LAR": ("Rams", "Los Angeles"),
    "LV": ("Raiders", "Las Vegas"), "MIA": ("Dolphins", "Miami"),
    "MIN": ("Vikings", "Minnesota"), "NE": ("Patriots", "New England"),
    "NO": ("Saints", "New Orleans"), "NYG": ("Giants", "New York"),
    "NYJ": ("Jets", "New York"), "PHI": ("Eagles", "Philadelphia"),
    "PIT": ("Steelers", "Pittsburgh"), "SEA": ("Seahawks", "Seattle"),
    "SF": ("49ers", "San Francisco"), "TB": ("Buccaneers", "Tampa Bay"),
    "TEN": ("Titans", "Tennessee"), "WAS": ("Commanders", "Washington"),
}


# ==============================================================================
# 1. DATA LOADING & CONDITIONAL OUTCOME CALCULATION
# ==============================================================================
def _num(v, default=None):
    if v is None:
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def _inum(v, default=None):
    f = _num(v)
    return default if f is None else int(round(f))


def compute_conditional_outcomes(p10: int, p50: int, p90: int) -> tuple[int, int]:
    """
    Computes:
      - `hit_yds` ("If He Hits"): Expected 3-year yards conditional on panning out
        in the upper/starter regime (`0.42 * p50 + 0.58 * p90`).
      - `miss_yds` ("If He Misses"): Expected 3-year yards in the downside/bust
        regime (`0.72 * p10 + 0.28 * p50`).
    """
    hit_yds = int(round(0.42 * p50 + 0.58 * p90))
    miss_yds = int(round(0.72 * p10 + 0.28 * p50))
    return hit_yds, miss_yds


def load_ranking_rows(data_dir: Path) -> list[dict]:
    s2 = data_dir / STAGE2_RANKINGS_PATH.name
    if s2.exists():
        print(f"Loading {s2}")
        return pq.read_table(s2).to_pylist()
    rows: list[dict] = []
    for p in (data_dir / STAGE1_RANKINGS_PATH.name, data_dir / RECENT_PROSPECTS_PATH.name):
        if p.exists():
            print(f"Loading {p}")
            rows.extend(pq.read_table(p).to_pylist())
    return rows


def build_prospects(rows: list[dict], last_complete: int) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        yr = _inum(r.get("draft_year"))
        name = r.get("name")
        if yr is None or not name:
            continue

        n_p10 = _inum(r.get("nfl_proj_floor"), _inum(r.get("stage1_p10"), 300))
        n_p50 = _inum(r.get("nfl_proj_median"), _inum(r.get("stage1_p50"), 1200))
        n_p90 = _inum(r.get("nfl_proj_ceiling"), _inum(r.get("stage1_p90"), 2900))
        n_hit, n_miss = compute_conditional_outcomes(n_p10, n_p50, n_p90)

        s_p10 = _inum(r.get("stage2_p10"), n_p10)
        s_p50 = _inum(r.get("stage2_p50"), n_p50)
        s_p90 = _inum(r.get("stage2_p90"), n_p90)
        s_hit, s_miss = compute_conditional_outcomes(s_p10, s_p50, s_p90)

        raw_team = str(r.get("team") or "")
        team_clean = "" if raw_team in ("UNK", "None") else raw_team

        out.append({
            "id": f"p{i}",
            "name": str(name),
            "college": r.get("college") or "NCAA",
            "year": yr,
            "round": _inum(r.get("round")),
            "pick": _inum(r.get("pick")),
            "team": team_clean,
            "age": _num(r.get("draft_age")),
            "grade": round(_num(r.get("college_grade"), 75.0), 1),
            "arch": r.get("archetype_primary") or "Boundary X",
            "px": round(_num(r.get("p_x"), 0.34), 3),
            "pz": round(_num(r.get("p_z"), 0.33), 3),
            "ps": round(_num(r.get("p_slot"), 0.33), 3),
            # Stage 1.5: Team-Neutral
            "n_p10": n_p10, "n_p50": n_p50, "n_p90": n_p90,
            "n_hit": n_hit, "n_miss": n_miss,
            "n_st": _num(r.get("prob_starter"), 0.3),
            "n_bm": _num(r.get("prob_boom"), 0.15),
            "n_bu": _num(r.get("prob_bust"), 0.45),
            # Stage 2: Drafted-Team Fit
            "s_p10": s_p10, "s_p50": s_p50, "s_p90": s_p90,
            "s_hit": s_hit, "s_miss": s_miss,
            "s_st": _num(r.get("stage2_prob_starter"), _num(r.get("prob_starter"), 0.3)),
            "s_bm": _num(r.get("stage2_prob_boom"), _num(r.get("prob_boom"), 0.15)),
            "s_bu": _num(r.get("stage2_prob_bust"), _num(r.get("prob_bust"), 0.45)),
            "fit": _num(r.get("team_fit_mult"), 1.0),
            "d_qb": _num(r.get("delta_qb_vol"), 0.0),
            "d_vac": _num(r.get("delta_vacancy"), 0.0),
            "d_role": _num(r.get("delta_role"), 0.0),
            "notes": r.get("team_fit_notes") or "Balanced Room",
            # Ground-truth 3-year NFL outcome
            "actual": _inum(r.get("nfl_actual_y1_3"), 0),
            "done": yr <= last_complete,
            # Combine / Pro Day measurables
            "ht": _num(r.get("ht_in")),
            "wt": _num(r.get("wt_lbs")),
            "forty": _num(r.get("forty_sec")),
            "burst": _num(r.get("burst_score")),
            "agil": _num(r.get("agility_sum")),
            "no40": (
                bool(r.get("missing_forty"))
                if r.get("missing_forty") is not None
                else _num(r.get("forty_sec")) is None
            ),
        })
    out.sort(key=lambda p: (p["grade"], p["s_hit"]), reverse=True)
    return out


# ==============================================================================
# 2. YEAR-SPECIFIC NFL TEAM ENVIRONMENTS FOR CLIENT-SIDE MOCK SIMULATOR
# ==============================================================================
def build_year_specific_team_environments(data_dir: Path, years: list[int]) -> dict[str, dict]:
    stats_file = data_dir / NFL_STATS_PLAYER_PATH.name
    combine_file = COMBINE_RAW_PATH if COMBINE_RAW_PATH.exists() else (data_dir / "combine.parquet")

    fallback_envs = {}
    for tm, (n, city) in TEAM_NAMES.items():
        col = TEAM_COLORS.get(tm, "#3f3f46")
        for yr in years:
            fallback_envs[f"{tm}_{yr}"] = {
                "name": n, "city": city, "color": col, "year": yr,
                "qb": "Starting QB", "proe": 0.0, "qb_epa": 0.0, "vacated": 0.24,
                "style": "Pro Style", "incumbents": [],
            }

    if not stats_file.exists():
        return fallback_envs

    print(f"Extracting year-specific NFL team environments from {stats_file}...")
    stats_rows = pq.read_table(stats_file).to_pylist()

    vet_size = {}
    if combine_file.exists():
        for r in pq.read_table(combine_file).to_pylist():
            pnm = str(r.get("player_name") or "").lower().strip()
            ht = parse_height_inches(r.get("ht"))
            wt = _num(r.get("wt"))
            if pnm and ht and wt:
                vet_size[pnm] = (ht, wt)

    by_st_player: dict[tuple[int, str], dict[str, dict]] = {}
    team_att_epa: dict[tuple[int, str], dict] = {}
    all_seasons = set()

    for r in stats_rows:
        if r.get("season_type") != "REG":
            continue
        s_raw = r.get("season")
        tm_raw = r.get("recent_team") or r.get("team")
        if not s_raw or not tm_raw:
            continue
        s = int(s_raw)
        tm = norm_nfl_team(str(tm_raw))
        all_seasons.add(s)

        pid = str(r.get("player_id") or "")
        pname = str(r.get("player_display_name") or r.get("player_name") or "Player")
        pos = str(r.get("position") or "")

        att = float(r.get("attempts") or 0.0)
        p_epa = float(r.get("passing_epa") or 0.0)
        tgts = float(r.get("targets") or 0.0)
        rec = float(r.get("receptions") or 0.0)
        yd = float(r.get("receiving_yards") or 0.0)
        air = float(r.get("receiving_air_yards") or 0.0)
        yac = float(r.get("receiving_yards_after_catch") or 0.0)

        tot = team_att_epa.setdefault(
            (s, tm), {"att": 0.0, "pass_epa": 0.0, "tgts": 0.0, "air": 0.0, "qbs": {}}
        )
        tot["att"] += att
        tot["pass_epa"] += p_epa
        tot["tgts"] += tgts
        tot["air"] += max(0.0, air)
        if att >= 15:
            tot["qbs"][pname] = tot["qbs"].get(pname, 0.0) + att

        st_dict = by_st_player.setdefault((s, tm), {})
        p_acc = st_dict.setdefault(
            pid,
            {
                "pid": pid, "name": pname, "pos": pos, "att": 0.0, "targets": 0.0,
                "receptions": 0.0, "rec_yds": 0.0, "air_yds": 0.0, "yac_yds": 0.0,
            },
        )
        p_acc["att"] += att
        p_acc["targets"] += tgts
        p_acc["receptions"] += rec
        p_acc["rec_yds"] += yd
        p_acc["air_yds"] += max(0.0, air)
        p_acc["yac_yds"] += max(0.0, yac)

    season_norms = {}
    for s in all_seasons:
        s_atts = [t["att"] for (sy, _), t in team_att_epa.items() if sy == s and t["att"] >= 150]
        s_epas = [
            t["pass_epa"] / max(t["att"], 1.0)
            for (sy, _), t in team_att_epa.items()
            if sy == s and t["att"] >= 150
        ]
        if s_atts:
            m_a = sum(s_atts) / len(s_atts)
            m_e = sum(s_epas) / len(s_epas)
            season_norms[s] = {
                "att_mean": m_a,
                "att_std": max(25.0, (sum((x - m_a) ** 2 for x in s_atts) / len(s_atts)) ** 0.5),
                "epa_mean": m_e,
                "epa_std": max(0.04, (sum((x - m_e) ** 2 for x in s_epas) / len(s_epas)) ** 0.5),
            }

    max_stat_season = max(all_seasons) if all_seasons else COMPLETED_NFL_SEASON
    envs = {}

    for draft_yr in years:
        prev_s = min(draft_yr - 1, max_stat_season)
        curr_s = draft_yr if draft_yr <= max_stat_season else None

        for tm, (n, city) in TEAM_NAMES.items():
            col = TEAM_COLORS.get(tm, "#3f3f46")
            ts = team_att_epa.get((prev_s, tm))
            if not ts:
                envs[f"{tm}_{draft_yr}"] = fallback_envs[f"{tm}_{draft_yr}"]
                continue

            sn = season_norms.get(
                prev_s, {"att_mean": 550.0, "att_std": 60.0, "epa_mean": 0.05, "epa_std": 0.12}
            )
            epa_per_att = ts["pass_epa"] / max(ts["att"], 100.0)
            pass_vol_z = round(max(-2.2, min(2.2, (ts["att"] - sn["att_mean"]) / sn["att_std"])), 2)
            qb_epa_z = round(max(-2.2, min(2.2, (epa_per_att - sn["epa_mean"]) / sn["epa_std"])), 2)

            top_qb = max(ts["qbs"].items(), key=lambda x: x[1])[0] if ts["qbs"] else "Starting QB"

            if pass_vol_z >= 0.70 and qb_epa_z >= 0.70:
                style = "Air Raid / Elite Pass"
            elif pass_vol_z >= 0.50:
                style = "Pass-Heavy Spread"
            elif pass_vol_z <= -0.70:
                style = "Run-First Ground Attack"
            elif qb_epa_z >= 0.50:
                style = "High-Efficiency Play-Action"
            else:
                style = "Balanced Pro-Style"

            tot_tgts = max(ts["tgts"], 200.0)
            curr_roster_pids = (
                {
                    pid
                    for pid, pdata in by_st_player.get((curr_s, tm), {}).items()
                    if pdata["targets"] >= 8 or pdata["att"] >= 15
                }
                if curr_s is not None
                else None
            )
            p_list = list(by_st_player.get((prev_s, tm), {}).values())

            vacated_tgts = 0.0
            incumbents = []
            wr_te = [p for p in p_list if p["pos"] in ("WR", "TE") and p["targets"] >= 15]
            wr_te.sort(key=lambda x: x["targets"], reverse=True)

            for p in wr_te:
                t_share = p["targets"] / tot_tgts
                is_gone = (curr_roster_pids is not None) and (p["pid"] not in curr_roster_pids)
                if is_gone:
                    vacated_tgts += p["targets"]
                else:
                    if (p["pos"] == "WR" and t_share >= 0.11) or (p["pos"] == "TE" and t_share >= 0.18):
                        adot = p["air_yds"] / max(p["targets"], 1.0)
                        yac_r = p["yac_yds"] / max(p["rec_yds"], 1.0)
                        ypc = p["rec_yds"] / max(p["receptions"], 1.0)
                        p_low = p["name"].lower().strip()
                        ht_v, wt_v = vet_size.get(p_low, (72.5, 200.0))
                        bmi_v = (wt_v * 703.0) / (ht_v ** 2)

                        if p["pos"] == "TE":
                            vec = [0.15, 0.15, 0.70]
                        else:
                            vx = (adot - 10.5) * 0.35 + (0.35 - yac_r) * 3.5 + (ht_v - 72.0) * 0.35 + (bmi_v - 26.0) * 0.45
                            vz = (ypc - 13.0) * 0.35 + (adot - 10.0) * 0.25 - (bmi_v - 26.0) * 0.35
                            vslot = (10.2 - adot) * 0.45 + (yac_r - 0.32) * 4.5 - (ht_v - 71.5) * 0.25
                            sc = [vx, vz, vslot]
                            m_sc = max(sc)
                            ex = [math.exp((v - m_sc) / 1.15) for v in sc]
                            s_ex = sum(ex)
                            vec = [round(v / s_ex, 3) for v in ex]

                        incumbents.append({
                            "name": p["name"],
                            "pos": p["pos"],
                            "target_share": round(t_share, 3),
                            "adot": round(adot, 1),
                            "vector": vec,
                        })

            vac_pct = round(
                max(0.04, min(0.68, (vacated_tgts / tot_tgts) if curr_roster_pids is not None else 0.24)),
                2,
            )

            envs[f"{tm}_{draft_yr}"] = {
                "name": n,
                "city": city,
                "color": col,
                "year": draft_yr,
                "qb": top_qb,
                "proe": pass_vol_z,
                "qb_epa": qb_epa_z,
                "vacated": vac_pct,
                "style": style,
                "incumbents": incumbents[:3],
            }

    return envs


# ==============================================================================
# 3. SELF-CONTAINED HTML APPLICATION TEMPLATE
# ==============================================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boom or Bust — NFL WR Prospect Translation Board</title>
<style>
:root{
  --bg:#09090b;--panel:#141418;--panel2:#1c1c22;--line:#27272a;
  --mut:#a1a1aa;--dim:#71717a;--txt:#fafafa;
  --g:#34d399;--g-bg:rgba(52,211,153,.12);
  --b:#60a5fa;--b-bg:rgba(96,165,250,.12);
  --r:#fb7185;--r-bg:rgba(251,113,133,.12);
  --a:#fbbf24
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.wrap{max-width:1080px;margin:0 auto;padding:0 18px}
header{position:sticky;top:0;z-index:20;background:rgba(9,9,11,.90);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
header .wrap{height:60px;display:flex;align-items:center;justify-content:space-between}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#fafafa,#a1a1aa);color:#09090b;display:grid;place-items:center;font-weight:900;font-size:12px;letter-spacing:-.03em}
.brand b{font-size:15px;letter-spacing:-.01em}.brand small{display:block;color:var(--mut);font-size:11px;margin-top:1px}
button,select,input{font:inherit;color:var(--txt)}
.btn{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 11px;font-size:12px;cursor:pointer;color:var(--mut);transition:all .12s}
.btn:hover{border-color:#52525b;color:#fff}
.controls{padding:14px 0 12px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;border-bottom:1px solid var(--line)}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);font-weight:600}
select,input[type=text]{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:12px}
input[type=text]{width:185px}
.seg{display:flex;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:2px;gap:2px;flex-wrap:wrap}
.seg button{background:none;border:0;border-radius:6px;padding:4px 9px;font-size:12px;color:var(--mut);cursor:pointer;font-weight:500}
.seg button.on{background:#f4f4f5;color:#09090b;font-weight:700}
.chips{margin-left:auto}
.meta{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:10px 0;color:var(--dim);font-size:12px}
.meta b{color:#e4e4e7;font-weight:600}
.col-hdr{display:grid;grid-template-columns:42px minmax(0,1.25fr) minmax(240px,1.2fr) 155px 90px;gap:14px;padding:6px 8px;font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim);font-weight:700}
.row{display:grid;grid-template-columns:42px minmax(0,1.25fr) minmax(240px,1.2fr) 155px 90px;gap:14px;align-items:center;padding:12px 8px;border-top:1px solid var(--line);cursor:pointer;transition:background .1s}
.row:hover{background:rgba(39,39,42,.38)}
.rk{font-size:20px;font-weight:800;color:#52525b}.row:hover .rk{color:#fff}
.nm{font-weight:700;font-size:15px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sub{color:var(--mut);font-size:12px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tag{display:inline-block;font-size:10px;font-weight:600;padding:1px 6px;border-radius:5px;border:1px solid var(--line);background:var(--panel);color:#d4d4d8;margin-left:6px;vertical-align:1px}
.tag.warn{color:var(--a);border-color:#5b4a12}
.tag.mock{color:var(--g);border-color:#14532d}
.bar{position:relative;height:8px;background:var(--panel2);border-radius:4px}
.bar .fill{position:absolute;top:0;bottom:0;background:linear-gradient(90deg,#3f3f46 0%,#60a5fa 65%,#34d399 100%);border-radius:4px;opacity:.85}
.bar .hit-tick{position:absolute;top:-3px;width:3px;height:14px;background:#34d399;border-radius:1.5px;box-shadow:0 0 6px rgba(52,211,153,.6)}
.bar .dot{position:absolute;top:-3px;width:14px;height:14px;margin-left:-7px;border-radius:50%;border:2px solid var(--bg);z-index:2}
.dot.in{background:#fafafa}.dot.out{background:var(--r)}
.barlbl{display:flex;justify-content:space-between;color:var(--dim);font-size:10px;margin-top:6px}
.barlbl .hit-Inline{color:var(--g);font-weight:700}
.outcomes{display:flex;gap:6px;justify-content:flex-end}
.pill-box{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 7px;text-align:right;min-width:72px}
.pill-box.hit{border-color:rgba(52,211,153,.3);background:var(--g-bg)}
.pill-box.miss{border-color:rgba(251,113,133,.22);background:var(--r-bg)}
.pill-box small{display:block;font-size:9px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
.pill-box b{font-size:12px}
.met{text-align:right}.met small{display:block;color:var(--dim);font-size:10px;text-transform:uppercase;letter-spacing:.05em}.met b{font-size:15px}
#empty{display:none;padding:60px 0;text-align:center;color:var(--dim)}
footer{color:var(--dim);font-size:11px;padding:18px 0 60px;border-top:1px solid var(--line);line-height:1.6}
#backdrop{position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:40;display:none}
#drawer{position:fixed;top:0;right:0;bottom:0;width:min(500px,100%);background:var(--bg);border-left:1px solid var(--line);z-index:50;transform:translateX(100%);transition:transform .18s ease-out;overflow-y:auto}
#drawer.open{transform:none}
.dh{padding:18px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:10px}
.dh h2{margin:2px 0 4px;font-size:22px;font-weight:800}
.sec{padding:16px 20px;border-bottom:1px solid var(--line)}
.sec h4{margin:0 0 10px;font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.grid2{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:10px}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px 11px}
.cell.hit-card{border-color:rgba(52,211,153,.35);background:var(--g-bg)}
.cell.miss-card{border-color:rgba(251,113,133,.28);background:var(--r-bg)}
.cell small{display:block;color:var(--mut);font-size:10px;text-transform:uppercase;letter-spacing:.05em}
.cell b{display:block;font-size:16px;margin-top:3px}
.cell span.sub-note{display:block;font-size:11px;color:var(--dim);margin-top:2px}
.pos{color:var(--g)}.neg{color:var(--r)}
.tbl{width:100%;border-collapse:collapse;font-size:12px}
.tbl th{color:var(--dim);font-weight:600;text-align:right;padding:5px 6px;font-size:10px;text-transform:uppercase}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.tbl td{padding:6px;text-align:right;border-top:1px solid var(--line)}
.mix{display:flex;height:8px;border-radius:4px;overflow:hidden;background:var(--panel)}
.mixlbl{display:flex;justify-content:space-between;font-size:11px;color:var(--mut);margin-top:6px}
.teams{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:10px}
.tb{border:1px solid var(--line);background:var(--panel2);border-radius:7px;padding:6px 0;font-size:11px;font-weight:800;cursor:pointer;text-align:center;color:#fff}
.tb.on{outline:2px solid #f4f4f5}
.note{color:var(--dim);font-size:11px;margin-top:8px;line-height:1.45}
.acts{display:flex;gap:8px;margin-top:12px}
.acts .primary{flex:1;background:#f4f4f5;color:#09090b;border:0;border-radius:8px;padding:8px;font-weight:700;cursor:pointer}
@media(max-width:760px){
  .col-hdr{display:none}
  .row{grid-template-columns:34px minmax(0,1fr) 86px}.bar-col,.outcomes{display:none}
  .chips{margin-left:0}
}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="brand"><div class="logo">WR</div><div><b>Boom or Bust &mdash; WR Translation Board</b><small><span id="yr-span"></span> classes &middot; zero draft-capital bias</small></div></div>
  <div><span id="mock-pill" class="tag mock" style="display:none"></span> <button class="btn" id="btn-reset-mocks" style="display:none">Clear mocks</button> <button class="btn" id="btn-reset">Reset filters</button></div>
</div></header>

<div class="wrap">
  <div class="controls">
    <span class="lbl">Sort</span>
    <select id="sel-sort">
      <option value="grade">College Grade</option>
      <option value="hit">If He Hits (Starter Projection)</option>
      <option value="ceiling">Ceiling (p90)</option>
      <option value="floor">Floor (p10)</option>
      <option value="boom">Boom Odds (&ge;2,500 yds)</option>
      <option value="safe">Lowest Bust Risk (&lt;500 yds)</option>
      <option value="actual">Actual Y1&ndash;3 Yards</option>
    </select>
    <div class="seg" id="seg-view">
      <button data-view="neutral">Team-Neutral</button>
      <button data-view="fit" class="on">Drafted-Team Fit</button>
    </div>
    <select id="sel-arch">
      <option value="all">All Archetypes</option>
      <option>Boundary X</option><option>Flanker Z</option><option>Interior Slot</option>
    </select>
    <input id="inp-q" type="text" placeholder="Search player, school, team">
    <div class="seg chips" id="seg-class"></div>
  </div>
  <div class="meta"><div><b id="count">0</b> prospects &middot; <span id="sortdesc"></span></div><div id="calib"></div></div>
  <div class="col-hdr">
    <div>Rk</div>
    <div>Prospect</div>
    <div>80% Outcome Range (p10 &rarr; If Hits &rarr; p90)</div>
    <div style="text-align:right">If Hits / If Misses</div>
    <div style="text-align:right">Primary</div>
  </div>
  <main id="rows"></main>
  <div id="empty">No prospects match your current filter.</div>
  <footer>
    <b>Why "If He Hits" instead of raw Median (p50)?</b> NFL rookie-contract receiver outcomes are bimodal: prospects either earn a starting role (~1,800&ndash;3,800+ yds) or get derailed by injuries/depth-chart burial (&lt;600 yds). An unconditional p50 blends both regimes into a misleading ~1,200-yard middle ground that few players actually finish at. Each prospect is shown with their <b>80% Outcome Range (p10&ndash;p90)</b>, their <b>"If He Hits" Starter Projection</b> (green tick), and their <b>"If He Misses" Downside</b>.
    Actuals are displayed for completed classes through <span id="lastdone"></span> (white dot = landed inside p10&ndash;p90, red = outside).
  </footer>
</div>

<div id="backdrop"></div>
<aside id="drawer"><div id="drawer-body"></div></aside>

<script>
const P = __PROSPECTS__;
const ENVS = __ENVS__;
const META = __META__;
const SCALE = 4850;
const $ = id => document.getElementById(id);
const fmt = n => (n == null ? '–' : Math.round(n).toLocaleString());
const pct = (v, d) => (v * 100).toFixed(d || 0) + '%';
const sgn = (v, d) => (v >= 0 ? '+' : '') + (v * 100).toFixed(d == null ? 1 : d) + '%';
const clip = (v, a, b) => Math.min(b, Math.max(a, v));
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const st = { sort: 'grade', view: 'fit', cls: 'all', arch: 'all', q: '', sel: null, preview: 'KC' };

const SORT_DESC = {
  grade: 'ranked by pre-draft college translation grade',
  hit: 'ranked by "If He Hits" 3-year starter projection',
  ceiling: 'ranked by 90th-percentile ceiling (p90)',
  floor: 'ranked by 10th-percentile floor (p10)',
  boom: 'ranked by probability of a 2,500+ yard boom',
  safe: 'ranked by lowest probability of a sub-500 yard bust',
  actual: 'ranked by actual first-3-season NFL receiving yards'
};

function erf(x) {
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
function deriveProbs(p10, p50, p90) {
  const s10 = Math.sqrt(Math.max(0, p10)), s50 = Math.sqrt(Math.max(25, p50)), s90 = Math.sqrt(Math.max(100, p90));
  const sl = Math.max(3, (s50 - s10) / 1.38), sr = Math.max(3.8, (s90 - s50) / 1.38);
  const cdf = y => { const sy = Math.sqrt(Math.max(0, y)); const z = (sy - s50) / (sy < s50 ? sl : sr); return 0.5 * (1 + erf(z / Math.SQRT2)); };
  const bust = clip(cdf(500), 0.03, 0.94);
  const starter = clip(1 - cdf(1500), 0.03, 0.94);
  const boom = clip(1 - cdf(2500), 0.01, Math.min(0.88, starter - 0.04));
  const hit = Math.round(0.42 * p50 + 0.58 * p90);
  const miss = Math.round(0.72 * p10 + 0.28 * p50);
  return { starter, boom, bust, hit, miss };
}

function mockFit(p, code) {
  const key = code + '_' + p.year;
  const env = ENVS[key] || ENVS[code + '_' + META.last_complete] || {
    name: code, city: code, color: '#3f3f46', year: p.year, qb: 'Starting QB',
    proe: 0.0, qb_epa: 0.0, vacated: 0.24, style: 'Pro Style', incumbents: []
  };

  const rVec = [p.px, p.pz, p.ps];
  const rawQbVol = (env.proe * 0.055) + (env.qb_epa * 0.075);
  const rawVacancy = (env.vacated - 0.24) * 0.38;

  let congestionPen = 0.0;
  let synergyBon = 0.0;
  let notes = [];

  const incs = env.incumbents || [];
  const wrIncShare = incs.filter(i => i.pos === 'WR').reduce((s, i) => s + i.target_share, 0.0);

  incs.forEach(inc => {
    const vVec = inc.vector || [0.33, 0.33, 0.33];
    const dot = rVec[0]*vVec[0] + rVec[1]*vVec[1] + rVec[2]*vVec[2];
    const share = inc.target_share;
    if (share >= 0.19 && dot >= 0.46) {
      const pen = (share - 0.14) * Math.pow(dot, 1.3) * 0.48;
      congestionPen += pen;
      notes.push('Clash w/ ' + inc.name + ' (' + Math.round(dot*100) + '%, -' + (pen*100).toFixed(1) + '%)');
    } else if (share >= 0.19 && dot <= 0.28) {
      const syn = share * (1.0 - dot) * 0.22;
      synergyBon += syn;
      notes.push('Synergy w/ ' + inc.name + ' (' + Math.round(dot*100) + '%, +' + (syn*100).toFixed(1) + '%)');
    }
  });

  if (wrIncShare <= 0.28) {
    const rBon = p.grade >= 90.0 ? 0.065 : 0.035;
    synergyBon += rBon;
    notes.push('Open WR1 Runway');
  } else if (env.vacated >= 0.34 && wrIncShare <= 0.35) {
    synergyBon += 0.045;
    notes.push('High Vacancy (' + Math.round(env.vacated*100) + '%)');
  }

  const rawRole = synergyBon - congestionPen;
  const upsideFactor = Math.max(0.75, Math.min(1.35, Math.pow(p.grade / 80.0, 1.25)));
  const downsideInsulation = Math.max(0.45, Math.min(1.0, 1.0 - Math.max(0.0, (p.grade - 86.0) * 0.045)));

  const dQb = rawQbVol >= 0 ? rawQbVol * upsideFactor : rawQbVol * downsideInsulation;
  const dVac = rawVacancy >= 0 ? rawVacancy * upsideFactor : rawVacancy * downsideInsulation;
  const dRole = rawRole * (rawRole > 0 ? 1.10 : 0.85);

  const mult = clip(1.0 + dQb + dVac + dRole, 0.82, 1.28);
  const p10 = Math.round(clip(p.n_p10 * (1.0 + dRole * 1.2), 0, 1600));
  const p50 = Math.round(clip(p.n_p50 * mult, p10 + 180, 3600));
  const ceilExp = mult >= 1.0 ? 1.10 : 0.85;
  const p90 = Math.round(clip(p.n_p90 * Math.pow(mult, ceilExp), p50 + 700, 4850));

  const roleLabel = rawRole > 0.03 ? 'Synergy' : (rawRole < -0.03 ? 'Congested' : 'Balanced');
  return Object.assign({ p10, p50, p90, mult, dQb, dVac, dRole, roleLabel, notes: notes.join('; ') || 'Balanced Room', team: env, code }, deriveProbs(p10, p50, p90));
}

function eff(p) {
  if (p.mock_fit) return p.mock_fit;
  if (st.view === 'fit') return { p10: p.s_p10, p50: p.s_p50, p90: p.s_p90, hit: p.s_hit, miss: p.s_miss, starter: p.s_st, boom: p.s_bm, bust: p.s_bu };
  return { p10: p.n_p10, p50: p.n_p50, p90: p.n_p90, hit: p.n_hit, miss: p.n_miss, starter: p.n_st, boom: p.n_bm, bust: p.n_bu };
}

function sortKey(p, e) {
  switch (st.sort) {
    case 'hit': return e.hit;
    case 'floor': return e.p10;
    case 'ceiling': return e.p90;
    case 'boom': return e.boom;
    case 'safe': return -e.bust;
    case 'actual': return p.done ? p.actual : -1;
    default: return p.grade;
  }
}

function visible() {
  const q = st.q;
  const list = P.filter(p =>
    (st.cls === 'all' || String(p.year) === st.cls) &&
    (st.arch === 'all' || p.arch === st.arch) &&
    (!q || p.name.toLowerCase().includes(q) || String(p.college).toLowerCase().includes(q) || String(p.team).toLowerCase().includes(q)));
  return list.map(p => ({ p, e: eff(p) })).sort((a, b) => {
    const d = sortKey(b.p, b.e) - sortKey(a.p, a.e);
    return d !== 0 ? d : b.e.hit - a.e.hit;
  });
}

function metric(p, e) {
  switch (st.sort) {
    case 'hit': return ['If Hits', fmt(e.hit)];
    case 'floor': return ['Floor', fmt(e.p10)];
    case 'ceiling': return ['Ceiling', fmt(e.p90)];
    case 'boom': return ['Boom', pct(e.boom)];
    case 'safe': return ['Bust', pct(e.bust)];
    case 'actual': return ['Actual', p.done ? fmt(p.actual) : 'pending'];
    default: return ['Grade', p.grade.toFixed(1)];
  }
}

function barHtml(p, e) {
  const x = v => clip(v / SCALE * 100, 0, 100);
  let dot = '';
  if (p.done) {
    const inside = p.actual >= e.p10 && p.actual <= e.p90;
    dot = '<i class="dot ' + (inside ? 'in' : 'out') + '" style="left:' + x(p.actual) + '%" title="Actual 3-Yr NFL Yards: ' + fmt(p.actual) + '"></i>';
  }
  return '<div class="bar" title="80% Range: ' + fmt(e.p10) + ' – ' + fmt(e.p90) + ' yds | If He Hits: ' + fmt(e.hit) + ' yds">' +
    '<div class="fill" style="left:' + x(e.p10) + '%;width:' + Math.max(1, x(e.p90) - x(e.p10)) + '%"></div>' +
    '<div class="hit-tick" style="left:' + x(e.hit) + '%"></div>' + dot + '</div>' +
    '<div class="barlbl mono"><span>p10: ' + fmt(e.p10) + '</span><span class="hit-Inline">Hit: ' + fmt(e.hit) + '</span><span>p90: ' + fmt(e.p90) + '</span></div>';
}

function render() {
  const list = visible();
  $('count').textContent = list.length;
  $('sortdesc').textContent = SORT_DESC[st.sort];

  const done = list.filter(x => x.p.done);
  if (done.length >= 5) {
    const inside = done.filter(x => x.p.actual >= x.e.p10 && x.p.actual <= x.e.p90).length;
    $('calib').innerHTML = 'Completed classes shown (n=' + done.length + '): <b>' + Math.round(inside / done.length * 100) + '%</b> landed inside p10&ndash;p90 range';
  } else { $('calib').textContent = ''; }

  const mocked = P.filter(p => p.mock_fit).length;
  $('mock-pill').style.display = mocked ? '' : 'none';
  $('btn-reset-mocks').style.display = mocked ? '' : 'none';
  $('mock-pill').textContent = mocked + ' mock' + (mocked > 1 ? 's' : '') + ' active';

  $('empty').style.display = list.length ? 'none' : 'block';
  $('rows').innerHTML = list.map((x, i) => {
    const p = x.p, e = x.e, m = metric(p, e);
    const pickTxt = (p.pick && p.pick < 500) ? 'Rd ' + p.round + ' #' + p.pick : 'Pre-Draft';
    const tags = (p.no40 ? '<span class="tag warn" title="No 40-yard dash on record">No 40</span>' : '') +
      (p.mock_fit ? '<span class="tag mock">Mock ' + esc(p.mock_fit.code) + ' (' + p.year + ')</span>' : '');
    return '<div class="row" data-id="' + p.id + '">' +
      '<div class="rk mono">' + String(i + 1).padStart(2, '0') + '</div>' +
      '<div style="min-width:0"><div class="nm">' + esc(p.name) + '<span class="tag">' + esc(p.arch) + '</span>' + tags + '</div>' +
      '<div class="sub">' + esc(p.college) + ' &middot; ' + p.year + (p.team ? ' &middot; ' + esc(p.team) : '') + ' &middot; ' + pickTxt + '</div></div>' +
      '<div class="bar-col">' + barHtml(p, e) + '</div>' +
      '<div class="outcomes mono">' +
        '<div class="pill-box hit" title="Expected 3-Yr Yards If He Hits (' + pct(e.starter) + ' starter odds)"><small>If Hits</small><b class="pos">' + fmt(e.hit) + '</b></div>' +
        '<div class="pill-box miss" title="Downside 3-Yr Yards If He Misses (' + pct(e.bust) + ' bust risk)"><small>If Miss</small><b class="neg">' + fmt(e.miss) + '</b></div>' +
      '</div>' +
      '<div class="met"><small>' + m[0] + '</small><b class="mono">' + m[1] + '</b></div></div>';
  }).join('');
}

function ink(hex) {
  const n = parseInt(hex.slice(1), 16), r = n >> 16, g = (n >> 8) & 255, b = n & 255;
  return (0.299 * r + 0.587 * g + 0.114 * b) > 150 ? '#09090b' : '#fff';
}
function ftIn(h) { return h == null ? '–' : Math.floor(h / 12) + "'" + Math.round(h % 12) + '"'; }

function renderDrawer() {
  const p = st.sel; if (!p) return;
  const m = mockFit(p, st.preview);
  const neutral = { p10: p.n_p10, p50: p.n_p50, p90: p.n_p90, hit: p.n_hit, miss: p.n_miss, starter: p.n_st, boom: p.n_bm, bust: p.n_bu };
  const drafted = { p10: p.s_p10, p50: p.s_p50, p90: p.s_p90, hit: p.s_hit, miss: p.s_miss, starter: p.s_st, boom: p.s_bm, bust: p.s_bu };
  const active = p.mock_fit || drafted;

  const row = (lbl, e) => '<tr><td>' + lbl + '</td><td class="mono">' + fmt(e.p10) + ' &ndash; ' + fmt(e.p90) +
    '</td><td class="mono pos"><b>' + fmt(e.hit) + '</b></td><td class="mono neg">' + fmt(e.miss) +
    '</td><td class="mono">' + pct(e.starter) + '</td><td class="mono">' + pct(e.boom) + '</td></tr>';
  const shiftCls = v => (v >= 0 ? 'pos' : 'neg');
  const draftShift = drafted.hit - neutral.hit;
  const mockShift = m ? m.hit - neutral.hit : 0;

  let outcome = '';
  if (p.done) {
    const inside = p.actual >= drafted.p10 && p.actual <= drafted.p90;
    outcome = '<div class="sec"><h4>Actual NFL Outcome (Years 1&ndash;3)</h4><div class="grid3">' +
      '<div class="cell"><small>Actual Y1&ndash;3 Yards</small><b class="mono">' + fmt(p.actual) + '</b></div>' +
      '<div class="cell"><small>80% Range</small><b class="mono">' + fmt(drafted.p10) + '&ndash;' + fmt(drafted.p90) + '</b></div>' +
      '<div class="cell"><small>Range Check</small><b class="' + (inside ? 'pos' : 'neg') + '">' + (inside ? 'Inside Range' : 'Outlier') + '</b></div></div></div>';
  } else {
    outcome = '<div class="sec"><h4>Actual NFL Outcome</h4><div class="note" style="margin:0">Class of ' + p.year + ' has not completed three full NFL seasons yet.</div></div>';
  }

  const ath = '<div class="grid3">' +
    '<div class="cell"><small>Height / Weight</small><b class="mono">' + ftIn(p.ht) + ' / ' + (p.wt == null ? '–' : Math.round(p.wt) + ' lbs') + '</b></div>' +
    '<div class="cell"><small>40-Yard Dash</small><b class="mono">' + (p.forty == null ? 'No 40' : p.forty.toFixed(2) + 's') + '</b></div>' +
    '<div class="cell"><small>Burst Score</small><b class="mono">' + (p.burst == null ? '–' : p.burst.toFixed(0)) + '</b></div></div>' +
    '<div class="note">Short-Area Agility (Cone + Shuttle): ' + (p.agil == null ? 'n/a' : p.agil.toFixed(2) + 's') + ' &middot; Draft Age: ' + (p.age == null ? 'n/a' : p.age.toFixed(1)) + '</div>';

  const mix = '<div class="mix"><div style="width:' + p.px * 100 + '%;background:#f4f4f5"></div><div style="width:' + p.pz * 100 + '%;background:#60a5fa"></div><div style="width:' + p.ps * 100 + '%;background:#34d399"></div></div>' +
    '<div class="mixlbl"><span>Boundary X ' + pct(p.px) + '</span><span>Flanker Z ' + pct(p.pz) + '</span><span>Interior Slot ' + pct(p.ps) + '</span></div>';

  const pillar = (lbl, v) => '<div class="cell"><small>' + lbl + '</small><b class="mono ' + shiftCls(v) + '">' + sgn(v) + '</b></div>';

  let mockBox = '';
  if (m) {
    const incNames = (m.team.incumbents || []).map(i => i.name + ' (' + Math.round(i.target_share*100) + '%)').join(', ');
    mockBox = '<div class="grid3" style="margin-top:12px">' +
      '<div class="cell"><small>Mock "If Hits" (' + p.year + ')</small><b class="mono pos">' + fmt(m.hit) + '</b></div>' +
      '<div class="cell"><small>vs Neutral</small><b class="mono ' + shiftCls(mockShift) + '">' + (mockShift >= 0 ? '+' : '') + fmt(mockShift) + '</b></div>' +
      '<div class="cell"><small>Room Fit</small><b style="font-size:13px">' + esc(m.roleLabel) + '</b></div></div>' +
      '<div class="note">' + esc(m.team.city) + ' ' + esc(m.team.name) + ' (' + p.year + ') &middot; QB: ' + esc(m.team.qb) + ' &middot; ' + esc(m.team.style) + ' &middot; ' + Math.round(m.team.vacated * 100) + '% targets vacated</div>' +
      (incNames ? '<div class="note" style="margin-top:4px;color:var(--mut)">Returning Starters: ' + esc(incNames) + '</div>' : '') +
      '<div class="note" style="margin-top:4px;color:var(--dim)">Notes: ' + esc(m.notes) + '</div>';
  }

  const teamCodes = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
  ];

  $('drawer-body').innerHTML =
    '<div class="dh"><div><span class="lbl">Prospect Profile</span><h2>' + esc(p.name) + '</h2>' +
    '<div class="sub">' + esc(p.college) + ' &middot; Class of ' + p.year + ((p.pick && p.pick < 500) ? ' &middot; Rd ' + p.round + ' #' + p.pick : ' &middot; Pre-Draft') + (p.team ? ' &middot; ' + esc(p.team) : '') + '</div>' +
    '<div style="margin-top:6px"><span class="tag" style="margin-left:0">' + esc(p.arch) + '</span><span class="tag">Grade ' + p.grade.toFixed(1) + '</span></div></div>' +
    '<button class="btn" id="btn-close" style="align-self:flex-start">&times;</button></div>' +

    '<div class="sec"><h4>Conditional 3-Year NFL Projection</h4>' +
    '<div class="grid2">' +
      '<div class="cell hit-card"><small>If He Hits (Starter Regime)</small><b class="mono pos">' + fmt(active.hit) + ' yds</b><span class="sub-note">' + pct(active.starter) + ' Starter Odds &middot; Ceiling: ' + fmt(active.p90) + ' yds</span></div>' +
      '<div class="cell miss-card"><small>If He Misses (Bust/Injury Regime)</small><b class="mono neg">' + fmt(active.miss) + ' yds</b><span class="sub-note">' + pct(active.bust) + ' Bust Risk &middot; Floor: ' + fmt(active.p10) + ' yds</span></div>' +
    '</div>' +
    '<table class="tbl"><tr><th>Context</th><th>80% Range (p10&ndash;p90)</th><th>If Hits</th><th>If Miss</th><th>Start</th><th>Boom</th></tr>' +
    row('Team-Neutral', neutral) + row('Drafted Team' + (p.team ? ' (' + esc(p.team) + ')' : ''), drafted) + '</table>' +
    '<div class="note">Landing-spot shift on "If He Hits" projection: <span class="mono ' + shiftCls(draftShift) + '">' + (draftShift >= 0 ? '+' : '') + fmt(draftShift) + ' yds</span> (Unconditional mathematical median p50: ' + fmt(active.p50) + ' yds).</div></div>' +

    outcome +

    '<div class="sec"><h4>Drafted-Team Environment Fit &middot; ' + p.fit.toFixed(2) + 'x</h4><div class="grid3">' +
    pillar('QB &amp; Volume', p.d_qb) + pillar('Vacancy', p.d_vac) + pillar('Role Fit', p.d_role) + '</div>' +
    '<div class="note">' + esc(p.notes || 'Balanced Room') + '</div></div>' +

    '<div class="sec"><h4>Play-Style Archetype Distribution</h4>' + mix + '</div>' +
    '<div class="sec"><h4>Combine &amp; Pro Day Biomechanics</h4>' + ath + '</div>' +

    '<div class="sec"><h4>Mock Landing-Spot Simulator (Class of ' + p.year + ')</h4>' +
    '<div class="teams">' + teamCodes.map(c => {
      const col = (ENVS[c + '_' + p.year] || {}).color || '#3f3f46';
      return '<button class="tb' + (c === st.preview ? ' on' : '') + '" data-team="' + c + '" style="background:' + col + ';color:' + ink(col) + '">' + c + '</button>';
    }).join('') + '</div>' +
    mockBox +
    '<div class="acts"><button class="primary" id="btn-apply">Apply Mock to Board</button><button class="btn" id="btn-clear-mock">Clear</button></div>' +
    '<div class="note">Simulates this prospect in any team\'s exact ' + p.year + ' offensive environment (starting QB EPA, pass volume, vacated targets, and incumbent WR archetype overlap).</div></div>';
}

function openDrawer(id) {
  const p = P.find(x => x.id === id); if (!p) return;
  st.sel = p;
  st.preview = p.mock_fit ? p.mock_fit.code : (p.team || 'KC');
  renderDrawer();
  $('backdrop').style.display = 'block';
  requestAnimationFrame(() => $('drawer').classList.add('open'));
}
function closeDrawer() {
  $('drawer').classList.remove('open');
  $('backdrop').style.display = 'none';
}

function buildChips() {
  const years = ['all'].concat(META.years.map(String));
  $('seg-class').innerHTML = years.map(y => '<button data-cls="' + y + '"' + (y === st.cls ? ' class="on"' : '') + '>' + (y === 'all' ? 'All' : y) + '</button>').join('');
}
function syncControls() {
  document.querySelectorAll('#seg-class button').forEach(b => b.classList.toggle('on', b.dataset.cls === st.cls));
  document.querySelectorAll('#seg-view button').forEach(b => b.classList.toggle('on', b.dataset.view === st.view));
  $('sel-sort').value = st.sort; $('sel-arch').value = st.arch; $('inp-q').value = st.q;
}

$('seg-class').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; st.cls = b.dataset.cls; syncControls(); render(); });
$('seg-view').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; st.view = b.dataset.view; syncControls(); render(); });
$('sel-sort').addEventListener('change', e => { st.sort = e.target.value; render(); });
$('sel-arch').addEventListener('change', e => { st.arch = e.target.value; render(); });
$('inp-q').addEventListener('input', e => { st.q = e.target.value.toLowerCase().trim(); render(); });
$('btn-reset').addEventListener('click', () => { st.sort = 'grade'; st.view = 'fit'; st.cls = 'all'; st.arch = 'all'; st.q = ''; syncControls(); render(); });
$('btn-reset-mocks').addEventListener('click', () => { P.forEach(p => { p.mock_fit = null; }); render(); });
$('rows').addEventListener('click', e => { const r = e.target.closest('.row'); if (r) openDrawer(r.dataset.id); });
$('backdrop').addEventListener('click', closeDrawer);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });
$('drawer-body').addEventListener('click', e => {
  const t = e.target.closest('[data-team]');
  if (t) { st.preview = t.dataset.team; renderDrawer(); return; }
  if (e.target.closest('#btn-close')) { closeDrawer(); return; }
  if (e.target.closest('#btn-apply')) { st.sel.mock_fit = mockFit(st.sel, st.preview); closeDrawer(); render(); return; }
  if (e.target.closest('#btn-clear-mock')) { st.sel.mock_fit = null; closeDrawer(); render(); }
});

$('lastdone').textContent = META.last_complete;
$('yr-span').textContent = META.years[META.years.length - 1] + '–' + META.years[0];
buildChips();
syncControls();
render();
</script>
</body>
</html>
"""


def build_rankings_app(
    data_dir: Path | None = None,
    out_path: Path | None = None,
    last_complete: int = DEFAULT_LAST_COMPLETE_CLASS,
) -> Path:
    target_dir = data_dir or PROCESSED_DIR
    rows = load_ranking_rows(target_dir)
    prospects = build_prospects(rows, last_complete)
    if not prospects:
        raise SystemExit("No prospects found in the parquet output.")

    years = sorted({p["year"] for p in prospects}, reverse=True)
    meta = {"years": years, "last_complete": last_complete}
    envs = build_year_specific_team_environments(target_dir, years)

    def _dump(obj) -> str:
        return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")

    html = (
        HTML_TEMPLATE.replace("__PROSPECTS__", _dump(prospects))
        .replace("__ENVS__", _dump(envs))
        .replace("__META__", _dump(meta))
    )

    out = out_path or (ROOT_DIR / "prospect_rankings_app.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(
        f"Generated dynamic board with {len(prospects)} prospects across classes "
        f"{years[-1]}–{years[0]} -> {out.resolve()} ({out.stat().st_size / 1024:.0f} KB)"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the dynamic year-specific WR prospect board HTML app.")
    ap.add_argument("--data-dir", help="Directory containing the pipeline's output parquets")
    ap.add_argument("--out", default=str(ROOT_DIR / "prospect_rankings_app.html"), help="Output HTML path")
    ap.add_argument(
        "--last-complete",
        type=int,
        default=DEFAULT_LAST_COMPLETE_CLASS,
        help="Last draft class with 3 full NFL seasons of stats (default: %(default)s)",
    )
    args = ap.parse_args()
    build_rankings_app(
        data_dir=Path(args.data_dir) if args.data_dir else None,
        out_path=Path(args.out),
        last_complete=args.last_complete,
    )


if __name__ == "__main__":
    main()