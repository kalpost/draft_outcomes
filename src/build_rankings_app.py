#!/usr/bin/env python3
"""
Boom or Bust — NFL WR Prospect Translation Board & Interactive Lineup / Monte Carlo Lab (`build_rankings_app.py`)
=================================================================================================================
Reads output parquets written by `run_unified_pipeline.py` and emits a self-contained,
zero-dependency HTML application (`prospect_rankings_app.html`) branded as **Boom or Bust**
with a minimalist **Black / White / Gray / Gold** design system (team colors reserved
strictly for NFL team badges):

  Tab 1 — Prospect Rankings Board:
    - Minimalist high-contrast Black (`#09090b`), Zinc Gray (`#18181b`), Crisp White (`#fafafa`),
      and Championship Gold (`#d4af37`) palette.
    - Visual Range Track per row (`p10 Floor — If He Hits — p90 Ceiling`) with exact delta
      shown **relative to Team-Neutral Base (`n_hit`)** whenever Drafted-Team Fit or a Mock
      landing spot is active.
    - Quick Team-Fit Drawer: compares Team-Neutral Base vs. Drafted vs. What-If with all yardage
      deltas calculated strictly **vs. Team-Neutral Base**.

  Tab 2 — Front-Office Lineup Lab & Rookie Contract Sim Game:
    - **Clean Instant Search + Interactive Prospect Picker List** (replacing the cluttered dropdown):
      * Search by receiver name, college, draft year, or NFL team with instant filtering,
        keyboard navigation (`Up`/`Down`/`Enter`), and a sleek scrollable prospect list showing
        each receiver's Grade, Class, College, and Archetype.
    - Interactive **Madden-Style 11-Personnel Offensive Lineup**:
      * Places the selected prospect directly into the team's actual starting offense (`QB1`,
        `WR1 (Boundary X)`, `WR2 (Flanker Z)`, `WR3 (Slot)`, `TE1`, plus `Bench / WR4`) alongside
        the real NFL players who played for that team in that season.
      * **Interactive GM Roster Moves**: Click any returning WR/TE card in the lineup to
        **Trade / Bench** them (vacating their targets into the offense in real time!), or toggle
        the **QB Tier Override** (`Actual QB` / `Elite All-Pro QB` / `Bridge QB`) and watch the
        prospect's Depth Chart Slot, Target Share, Fit Multiplier, and **Delta vs. Base** update live.
    - **Rookie Contract Career Roller & 1,000-Trial Monte Carlo Distribution**:
      * Simulates 1,000 careers in real time (Gray = Bust `<500`, White = Starter `500–2,499`,
        Gold = Boom `>=2,500`) and lets you roll individual 3-year rookie contracts with a
        percentile marker, Year 1–3 stat lines, and a **Year-4 Front-Office Contract Verdict**.
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
    RECENT_PROSPECTS_PATH,
    ROOT_DIR,
    STAGE1_RANKINGS_PATH,
    STAGE2_RANKINGS_PATH,
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
    Matches `run_unified_pipeline.py` exactly:
      - `hit_yds` ("If He Hits"): `0.48 * p50 + 0.52 * p90`
      - `miss_yds` ("If He Misses"): `0.74 * p10 + 0.26 * p50`
    """
    hit_yds = int(round(0.48 * p50 + 0.52 * p90))
    miss_yds = int(round(0.74 * p10 + 0.26 * p50))
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
        fb_n_hit, fb_n_miss = compute_conditional_outcomes(n_p10, n_p50, n_p90)
        n_hit = _inum(r.get("nfl_proj_hit"), fb_n_hit)
        n_miss = _inum(r.get("nfl_proj_miss"), fb_n_miss)

        s_p10 = _inum(r.get("stage2_p10"), n_p10)
        s_p50 = _inum(r.get("stage2_p50"), n_p50)
        s_p90 = _inum(r.get("stage2_p90"), n_p90)
        fb_s_hit, fb_s_miss = compute_conditional_outcomes(s_p10, s_p50, s_p90)
        s_hit = _inum(r.get("stage2_hit"), fb_s_hit)
        s_miss = _inum(r.get("stage2_miss"), fb_s_miss)

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
            "ypc": round(_num(r.get("college_ypc"), 14.2), 1),
            # Stage 1.5: Team-Neutral Base
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
            "hass": _num(r.get("hass")),
            "burst": _num(r.get("burst_score")),
            "agil": _num(r.get("agility_sum")),
            "no40": (
                bool(r.get("missing_forty"))
                if r.get("missing_forty") is not None
                else _num(r.get("forty_sec")) is None
            ),
        })
    out.sort(key=lambda p: (p["grade"], p["n_hit"]), reverse=True)
    return out


# ==============================================================================
# 2. YEAR-SPECIFIC NFL TEAM ENVIRONMENTS & STARTING LINEUPS FOR CLIENT-SIDE LAB
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
                "qb": "Starting QB", "qb_yds": 3600, "qb_td": 24,
                "proe": 0.0, "qb_epa": 0.0, "vacated": 0.24,
                "tot_att": 540, "style": "Balanced Pro-Style", "incumbents": [], "departed": [],
            }

    if not stats_file.exists():
        return fallback_envs

    print(f"Extracting year-specific NFL team lineups & environments from {stats_file}...")
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
        p_yds = float(r.get("passing_yards") or 0.0)
        p_td = float(r.get("passing_tds") or 0.0)
        p_epa = float(r.get("passing_epa") or 0.0)
        tgts = float(r.get("targets") or 0.0)
        rec = float(r.get("receptions") or 0.0)
        yd = float(r.get("receiving_yards") or 0.0)
        r_td = float(r.get("receiving_tds") or 0.0)
        air = float(r.get("receiving_air_yards") or 0.0)
        yac = float(r.get("receiving_yards_after_catch") or 0.0)

        tot = team_att_epa.setdefault(
            (s, tm), {"att": 0.0, "pass_epa": 0.0, "tgts": 0.0, "air": 0.0, "qbs": {}}
        )
        tot["att"] += att
        tot["pass_epa"] += p_epa
        tot["tgts"] += tgts
        tot["air"] += max(0.0, air)
        if att >= 10:
            qb_entry = tot["qbs"].setdefault(pname, {"att": 0.0, "yds": 0.0, "td": 0.0})
            qb_entry["att"] += att
            qb_entry["yds"] += p_yds
            qb_entry["td"] += p_td

        st_dict = by_st_player.setdefault((s, tm), {})
        p_acc = st_dict.setdefault(
            pid,
            {
                "pid": pid, "name": pname, "pos": pos, "att": 0.0, "targets": 0.0,
                "receptions": 0.0, "rec_yds": 0.0, "rec_td": 0.0, "air_yds": 0.0, "yac_yds": 0.0,
            },
        )
        p_acc["att"] += att
        p_acc["targets"] += tgts
        p_acc["receptions"] += rec
        p_acc["rec_yds"] += yd
        p_acc["rec_td"] += r_td
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

            if ts["qbs"]:
                top_qb, qb_st = max(ts["qbs"].items(), key=lambda x: x[1]["att"])
                qb_yds = int(round(qb_st["yds"]))
                qb_td = int(round(qb_st["td"]))
            else:
                top_qb, qb_yds, qb_td = "Starting QB", 3500, 22

            if pass_vol_z >= 0.70 and qb_epa_z >= 0.70:
                style = "High-Octane Spread"
            elif pass_vol_z >= 0.50:
                style = "Pass-Heavy Volume"
            elif pass_vol_z <= -0.70:
                style = "Run-First Play-Action"
            elif qb_epa_z >= 0.50:
                style = "High-EPA Precision"
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
            departed = []
            wr_te = [p for p in p_list if p["pos"] in ("WR", "TE") and p["targets"] >= 12]
            wr_te.sort(key=lambda x: x["targets"], reverse=True)

            for p in wr_te:
                t_share = p["targets"] / tot_tgts
                is_gone = (curr_roster_pids is not None) and (p["pid"] not in curr_roster_pids)
                adot = p["air_yds"] / max(p["targets"], 1.0)
                yac_r = p["yac_yds"] / max(p["rec_yds"], 1.0)
                ypc = p["rec_yds"] / max(p["receptions"], 1.0)
                p_low = p["name"].lower().strip()
                ht_v, wt_v = vet_size.get(p_low, (72.5, 200.0))
                bmi_v = (wt_v * 703.0) / (ht_v ** 2)

                if p["pos"] == "TE":
                    vec = [0.15, 0.15, 0.70]
                    arch_lbl = "Inline / Slot TE"
                else:
                    vx = (adot - 10.5) * 0.35 + (0.35 - yac_r) * 3.5 + (ht_v - 72.0) * 0.35 + (bmi_v - 26.0) * 0.45
                    vz = (ypc - 13.0) * 0.35 + (adot - 10.0) * 0.25 - (bmi_v - 26.0) * 0.35
                    vslot = (10.2 - adot) * 0.45 + (yac_r - 0.32) * 4.5 - (ht_v - 71.5) * 0.25
                    sc = [vx, vz, vslot]
                    m_sc = max(sc)
                    ex = [math.exp((v - m_sc) / 1.15) for v in sc]
                    s_ex = sum(ex)
                    vec = [round(v / s_ex, 3) for v in ex]
                    arch_lbl = ["Boundary X", "Flanker Z", "Interior Slot"][int(sc.index(m_sc))]

                ovr = int(round(min(97, max(68, 66 + t_share * 95 + (p["rec_yds"] / 115.0)))))
                p_obj = {
                    "name": p["name"],
                    "pos": p["pos"],
                    "arch": arch_lbl,
                    "ovr": ovr,
                    "target_share": round(t_share, 3),
                    "targets": int(round(p["targets"])),
                    "rec": int(round(p["receptions"])),
                    "yds": int(round(p["rec_yds"])),
                    "td": int(round(p["rec_td"])),
                    "adot": round(adot, 1),
                    "vector": vec,
                }

                if is_gone:
                    vacated_tgts += p["targets"]
                    if t_share >= 0.08 and len(departed) < 3:
                        departed.append(p_obj)
                else:
                    if len(incumbents) < 5 and t_share >= 0.06:
                        incumbents.append(p_obj)

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
                "qb_yds": qb_yds,
                "qb_td": qb_td,
                "tot_att": int(round(ts["att"])),
                "proe": pass_vol_z,
                "qb_epa": qb_epa_z,
                "vacated": vac_pct,
                "style": style,
                "incumbents": incumbents,
                "departed": departed,
            }

    return envs


# ==============================================================================
# 3. SELF-CONTAINED TWO-TAB HTML APPLICATION (BLACK / WHITE / GRAY / GOLD)
# ==============================================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boom or Bust &mdash; NFL WR Prospect Translation &amp; Lineup Lab</title>
<style>
:root{
  /* Strict Minimalist Black, White, Gray, & Gold Palette */
  --bg:#09090b;
  --surface:#111114;
  --surface2:#18181c;
  --surface3:#222228;
  --line:#27272d;
  --line-hi:#3f3f46;
  --txt:#fafafa;
  --mut:#a1a1aa;
  --dim:#71717a;
  --gold:#d4af37;
  --gold-hi:#f3cf55;
  --gold-bg:rgba(212,175,55,.12);
  --gold-bd:rgba(212,175,55,.38)
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:13.5px/1.45 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:1180px;margin:0 auto;padding:0 22px}

/* ---- Top Navigation & Tab Switcher ---- */
header{position:sticky;top:0;z-index:30;background:rgba(9,9,11,.92);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
header .wrap{height:60px;display:flex;align-items:center;justify-content:space-between;gap:16px}
.brand{display:flex;align-items:center;gap:12px}
.logo{width:32px;height:32px;border-radius:6px;background:var(--gold);color:#09090b;display:grid;place-items:center;font-weight:900;font-size:11px;letter-spacing:-.03em}
.brand b{font-size:15px;letter-spacing:.03em;text-transform:uppercase;color:var(--txt)}.brand small{display:block;color:var(--dim);font-size:11px;margin-top:1px}
.nav-tabs{display:flex;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:3px;gap:3px}
.nav-tabs button{background:none;border:0;border-radius:5px;padding:6px 14px;font-size:12px;font-weight:600;color:var(--mut);cursor:pointer;transition:all .12s}
.nav-tabs button:hover{color:var(--txt)}
.nav-tabs button.on{background:var(--txt);color:var(--bg);font-weight:700}
button,select,input{font:inherit;color:var(--txt)}
.btn{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:6px 11px;font-size:12px;cursor:pointer;color:var(--mut);font-weight:600;transition:all .12s}
.btn:hover{border-color:var(--line-hi);color:var(--txt)}
.btn.gold{background:var(--gold);color:#09090b;border-color:var(--gold);font-weight:700}
.btn.gold:hover{background:var(--gold-hi)}
.btn.primary{background:var(--txt);color:var(--bg);border-color:var(--txt);font-weight:700}
.btn.primary:hover{background:#e4e4e7}

/* ---- Controls Bar ---- */
.controls{padding:14px 0 12px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;border-bottom:1px solid var(--line)}
.lbl{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);font-weight:700}
select,input[type=text]{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font-size:12px;outline:none}
select:focus,input[type=text]:focus{border-color:var(--gold)}
input[type=text]{width:190px}
.seg{display:flex;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:2px;gap:2px;flex-wrap:wrap}
.seg button{background:none;border:0;border-radius:4px;padding:4px 9px;font-size:11.5px;color:var(--mut);cursor:pointer;font-weight:500}
.seg button.on{background:var(--surface3);color:var(--gold);font-weight:700}
.chips{margin-left:auto}
.meta{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:10px 0;color:var(--dim);font-size:12px}
.meta b{color:var(--txt);font-weight:600}

/* ---- Tab 1: Prospect Rankings Table & Minimalist Range Track ---- */
.col-hdr{display:grid;grid-template-columns:42px minmax(0,1.15fr) minmax(300px,1.55fr) 136px 88px;gap:16px;padding:8px 10px;font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);font-weight:700}
.row{display:grid;grid-template-columns:42px minmax(0,1.15fr) minmax(300px,1.55fr) 136px 88px;gap:16px;align-items:center;padding:13px 10px;border-top:1px solid var(--line);cursor:pointer;transition:background .1s}
.row:hover{background:var(--surface)}
.rk{font-size:16px;font-weight:700;color:var(--dim)}.row:hover .rk{color:var(--gold)}
.nm{font-weight:700;font-size:14.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sub{color:var(--mut);font-size:11.5px;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tag{display:inline-block;font-size:10px;font-weight:600;padding:1px 6px;border-radius:4px;border:1px solid var(--line);background:var(--surface2);color:var(--mut);margin-left:6px;vertical-align:1px}
.tag.gold{color:var(--gold);border-color:var(--gold-bd);background:var(--gold-bg)}
.tm-badge{display:inline-block;font-size:10px;font-weight:800;padding:1px 5px;border-radius:3px;letter-spacing:.03em;vertical-align:1px}

/* Minimalist Gold/White/Gray Range Track */
.rng-wrap{display:flex;flex-direction:column;gap:5px}
.rng-top{display:flex;justify-content:space-between;align-items:baseline;font-size:11px}
.rng-endpoints{color:var(--mut);font-size:11px}
.rng-hit-lbl{color:var(--txt);font-weight:700;font-size:11px}
.rng-hit-lbl span{color:var(--gold)}
.rng-track{position:relative;height:6px;background:var(--surface2);border-radius:3px;border:1px solid var(--line)}
.rng-band{position:absolute;top:0;bottom:0;background:linear-gradient(90deg,rgba(161,161,170,.25) 0%,rgba(212,175,55,.55) 100%);border-radius:2px}
.rng-marker{position:absolute;top:-3px;width:2.5px;height:10px;background:var(--gold);border-radius:1px;margin-left:-1.25px}
.rng-actual{position:absolute;top:-3.5px;width:11px;height:11px;margin-left:-5.5px;border-radius:50%;border:2px solid var(--bg);z-index:2}
.rng-actual.in{background:var(--txt)}
.rng-actual.out{background:var(--dim)}
.rng-sub{display:flex;justify-content:space-between;font-size:10px;color:var(--dim)}

.hit-col{text-align:right}
.hit-col b{display:block;font-size:13.5px;color:var(--txt)}
.hit-col small{display:block;font-size:10.5px;color:var(--dim);margin-top:1px}
.met{text-align:right}.met small{display:block;color:var(--dim);font-size:10px;text-transform:uppercase;letter-spacing:.05em}.met b{font-size:15px;color:var(--gold)}
#empty{display:none;padding:60px 0;text-align:center;color:var(--dim)}
footer{color:var(--dim);font-size:11px;padding:18px 0 60px;border-top:1px solid var(--line);line-height:1.6}

/* ---- Semantic Text Utilities (Strictly Gold, White, Gray) ---- */
.pos{color:var(--gold)}
.neg{color:var(--dim)}
.acc{color:var(--txt)}

/* ---- Quick What-If Drawer on Tab 1 ---- */
#backdrop{position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:40;display:none}
#drawer{position:fixed;top:0;right:0;bottom:0;width:min(500px,100%);background:var(--bg);border-left:1px solid var(--line);z-index:50;transform:translateX(100%);transition:transform .18s ease-out;overflow-y:auto}
#drawer.open{transform:none}
.dh{padding:18px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:10px;background:var(--surface)}
.dh h2{margin:2px 0 4px;font-size:20px;font-weight:800}
.sec{padding:16px 20px;border-bottom:1px solid var(--line)}
.sec h4{margin:0 0 10px;font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);display:flex;justify-content:space-between;align-items:center}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.cell{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 12px}
.cell.gold-cell{border-color:var(--gold-bd);background:var(--gold-bg)}
.cell small{display:block;color:var(--dim);font-size:9.5px;text-transform:uppercase;letter-spacing:.06em;font-weight:700}
.cell b{display:block;font-size:15.5px;margin-top:3px}
.cell span.sub-note{display:block;font-size:10.5px;color:var(--mut);margin-top:2px}
.teams{display:grid;grid-template-columns:repeat(8,1fr);gap:4px;margin:10px 0}
.tb{border:1px solid var(--line);background:var(--surface2);border-radius:4px;padding:6px 0;font-size:10.5px;font-weight:800;cursor:pointer;text-align:center;color:#fff;transition:transform .1s,opacity .1s;opacity:.82}
.tb:hover{opacity:1;border-color:var(--txt)}
.tb.on{opacity:1;outline:2px solid var(--gold);transform:scale(1.06);z-index:2}
.tbl{width:100%;border-collapse:collapse;font-size:12px}
.tbl th{color:var(--dim);font-weight:700;text-align:right;padding:6px;font-size:9.5px;text-transform:uppercase;letter-spacing:.06em}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.tbl td{padding:7px 6px;text-align:right;border-top:1px solid var(--line)}
.note{color:var(--dim);font-size:11px;margin-top:8px;line-height:1.45}
.acts{display:flex;gap:8px;margin-top:12px}

/* ---- Tab 2: Front-Office Lineup Lab & Career Sim Game ---- */
#tab-sim{display:none;padding:20px 0 60px}
.sim-layout{display:grid;grid-template-columns:320px minmax(0,1fr);gap:20px;align-items:start}
.sim-card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:16px}
.sim-card h3{margin:0 0 12px;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);font-weight:800}
.sim-field{margin-bottom:13px}
.sim-field label{display:flex;justify-content:space-between;align-items:center;font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin-bottom:6px;font-weight:700}
.sim-field select{width:100%;padding:7px 9px;font-size:12.5px;background:var(--surface2)}
.quick-row{display:flex;gap:5px;margin-top:8px}
.quick-row button{flex:1;padding:5px 6px;font-size:11px}

/* Clean Instant Search + Scrollable Prospect Picker List in Tab 2 */
.search-box{position:relative;display:flex;gap:6px;margin-bottom:7px}
.search-box input[type=text]{flex:1;width:100%;background:var(--bg);border:1px solid var(--line-hi);border-radius:6px;padding:7px 26px 7px 10px;font-size:12.5px;color:var(--txt)}
.search-box input[type=text]:focus{border-color:var(--gold)}
.search-clear{position:absolute;right:78px;top:50%;transform:translateY(-50%);background:none;border:0;color:var(--dim);cursor:pointer;font-size:14px;padding:0 4px;display:none}
.search-clear:hover{color:var(--txt)}
.search-cls-sel{width:70px!important;padding:6px 6px!important;font-size:11.5px!important;background:var(--bg)!important;border-color:var(--line-hi)!important;flex-shrink:0}

.p-list{max-height:215px;overflow-y:auto;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:3px;display:flex;flex-direction:column;gap:2px}
.p-list::-webkit-scrollbar{width:5px}
.p-list::-webkit-scrollbar-thumb{background:var(--line-hi);border-radius:3px}
.p-item{display:flex;justify-content:space-between;align-items:center;padding:6px 8px;border-radius:4px;cursor:pointer;border:1px solid transparent;transition:all .1s}
.p-item:hover{background:var(--surface2);border-color:var(--line)}
.p-item.on{background:var(--gold-bg);border-color:var(--gold)}
.p-item-l{min-width:0;flex:1}
.p-item-nm{font-weight:700;font-size:12.5px;color:var(--txt);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.p-item.on .p-item-nm{color:var(--gold)}
.p-item-sub{font-size:10.5px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.p-item-r{display:flex;align-items:center;gap:5px;margin-left:8px;flex-shrink:0}
.p-item-grd{font-size:11.5px;font-weight:800;color:var(--txt);padding:1px 5px;border-radius:3px;background:var(--surface2);border:1px solid var(--line)}
.p-item.on .p-item-grd{background:var(--gold);color:#09090b;border-color:var(--gold)}

/* Madden-Style 11-Personnel Tactical Formation Grid */
.field-board{margin-top:14px;background:radial-gradient(circle at 50% 0%,rgba(212,175,55,.06),transparent 70%),repeating-linear-gradient(0deg,var(--bg) 0px,var(--bg) 38px,rgba(255,255,255,.025) 39px);border:1px solid var(--line-hi);border-radius:8px;padding:16px;position:relative}
.field-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid var(--line)}
.los-line{border-top:1px dashed rgba(212,175,55,.45);margin:6px 0 14px;position:relative}
.los-line span{position:absolute;top:-8px;right:10px;background:var(--bg);padding:0 6px;font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--gold)}

.lineup-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.p-card{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px 11px;position:relative;transition:all .14s;display:flex;flex-direction:column;justify-content:space-between;min-height:112px}
.p-card.vet{cursor:pointer}
.p-card.vet:hover{border-color:var(--line-hi);background:var(--surface2)}
.p-card.prospect{border:1.5px solid var(--gold);background:linear-gradient(180deg,rgba(212,175,55,.14) 0%,var(--surface) 100%);box-shadow:0 0 20px rgba(212,175,55,.08)}
.p-card.benched{opacity:.42;border-style:dashed;background:var(--bg)}
.p-card-top{display:flex;justify-content:space-between;align-items:center;gap:4px}
.slot-pill{font-size:9.5px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;padding:2px 6px;border-radius:3px;background:var(--surface3);color:var(--mut)}
.p-card.prospect .slot-pill{background:var(--gold);color:#09090b}
.ovr-badge{font-size:12px;font-weight:900;padding:1px 6px;border-radius:4px;border:1px solid var(--line-hi);background:var(--bg);color:var(--txt)}
.p-card.prospect .ovr-badge{border-color:var(--gold);color:var(--gold)}
.p-card-name{font-weight:800;font-size:13.5px;margin:7px 0 2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.p-card-arch{font-size:10.5px;color:var(--dim)}
.p-card-stats{margin-top:8px;padding-top:7px;border-top:1px solid var(--line);display:flex;justify-content:space-between;align-items:baseline;font-size:11px}
.sh-bar{height:3px;background:var(--surface3);border-radius:2px;margin-top:4px;overflow:hidden}
.sh-fill{height:100%;background:var(--mut)}
.p-card.prospect .sh-fill{background:var(--gold)}
.act-hint{font-size:9px;text-transform:uppercase;letter-spacing:.05em;color:var(--dim);margin-top:5px;text-align:right}
.p-card.benched .act-hint{color:var(--gold)}

.qb-bar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-top:12px;padding:10px 12px;background:var(--surface);border:1px solid var(--line);border-radius:6px;flex-wrap:wrap}

/* Histogram & Career Roller Game Box */
.hist-wrap{margin-top:14px;padding:14px;background:var(--bg);border:1px solid var(--line);border-radius:6px}
.hist-bars{display:grid;grid-template-columns:repeat(10,1fr);gap:6px;align-items:end;height:118px;padding-top:18px;border-bottom:1px solid var(--line)}
.hbar-col{display:flex;flex-direction:column;align-items:center;height:100%;justify-content:flex-end}
.hbar{width:100%;border-radius:2px 2px 0 0;background:#d4d4d8;transition:height .2s;position:relative}
.hbar.bust{background:#3f3f46}
.hbar.boom{background:var(--gold)}
.hbar.active-bin{outline:1.5px solid var(--txt);box-shadow:0 0 10px rgba(212,175,55,.35)}
.hbar span{position:absolute;top:-15px;left:0;right:0;text-align:center;font-size:9.5px;color:var(--mut)}
.hist-lbls{display:grid;grid-template-columns:repeat(10,1fr);gap:6px;margin-top:6px;text-align:center;font-size:9.5px;color:var(--dim)}

.career-box{margin-top:14px;padding:16px;background:var(--bg);border:1px solid var(--line-hi);border-radius:6px}
.career-hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;flex-wrap:wrap;gap:8px}
.verdict-pill{font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;padding:4px 10px;border-radius:4px;border:1px solid var(--line-hi);background:var(--surface2)}
.verdict-pill.gold{border-color:var(--gold);background:var(--gold-bg);color:var(--gold)}
.roll-strip{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;padding-top:10px;border-top:1px solid var(--line);align-items:center}
.roll-chip{font-size:11px;padding:2px 8px;border-radius:4px;border:1px solid var(--line);background:var(--surface);color:var(--mut);cursor:pointer}
.roll-chip.gold{border-color:var(--gold-bd);color:var(--gold);background:var(--gold-bg)}

@media(max-width:900px){
  .col-hdr{display:none}
  .row{grid-template-columns:34px minmax(0,1fr) 86px}.rng-wrap,.hit-col{display:none}
  .chips{margin-left:0}
  .sim-layout{grid-template-columns:1fr}
  .lineup-row{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="brand">
    <div class="logo">B/B</div>
    <div><b>Boom or Bust</b><small>NFL WR Prospect Translation &amp; Lineup Lab &middot; <span id="yr-span"></span> classes &middot; zero draft-capital bias</small></div>
  </div>
  <div class="nav-tabs" id="nav-tabs">
    <button data-tab="board" class="on">1. Prospect Rankings Board</button>
    <button data-tab="sim">2. Lineup &amp; Career Sim Lab</button>
  </div>
  <div>
    <span id="mock-pill" class="tag gold" style="display:none"></span>
    <button class="btn" id="btn-reset-mocks" style="display:none">Clear Pins</button>
    <button class="btn" id="btn-reset">Reset</button>
  </div>
</div></header>

<!-- ====================================================================== -->
<!-- TAB 1: PROSPECT RANKINGS BOARD                                         -->
<!-- ====================================================================== -->
<div class="wrap" id="tab-board">
  <div class="controls">
    <span class="lbl">Sort</span>
    <select id="sel-sort">
      <option value="grade">Prospect Grade</option>
      <option value="hit">If He Hits (Starter Target)</option>
      <option value="ceiling">Ceiling (p90)</option>
      <option value="floor">Floor (p10)</option>
      <option value="boom">Boom Odds (&ge;2,500 yds)</option>
      <option value="safe">Lowest Bust Risk (&lt;500 yds)</option>
      <option value="actual">Actual Y1&ndash;3 Yards</option>
    </select>
    <div class="seg" id="seg-view">
      <button data-view="neutral">Base (Team-Neutral)</button>
      <button data-view="fit" class="on">Landing-Spot Fit</button>
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
    <div>80% Outcome Range (p10 Floor &mdash; If He Hits &mdash; p90 Ceiling)</div>
    <div style="text-align:right">If Hits (vs Base)</div>
    <div style="text-align:right">Primary</div>
  </div>
  <main id="rows"></main>
  <div id="empty">No prospects match your current filter.</div>
  <footer>
    <b>Boom or Bust &mdash; Base vs. Landing-Spot Framing:</b> Every team-fit shift (`+ / - yds vs Base`) is measured strictly against the prospect's <b>Team-Neutral Base (Stage 1.5)</b> expectation, isolating the exact impact of QB efficiency, pass volume, vacated targets, and WR room role overlap. For completed classes through <span id="lastdone"></span>, the circular dot marks actual 3-year NFL receiving yards (white = inside p10&ndash;p90, gray = outside).
  </footer>
</div>

<!-- ====================================================================== -->
<!-- TAB 2: FRONT-OFFICE LINEUP LAB & CAREER SIM GAME                       -->
<!-- ====================================================================== -->
<div class="wrap" id="tab-sim">
  <div class="sim-layout">
    <!-- Left Column: GM Controls with Clean Search + Scrollable List -->
    <div class="sim-card">
      <h3>Boom or Bust &middot; GM Controls</h3>
      <div class="sim-field">
        <label>
          <span>1. Search &amp; Select WR Prospect</span>
          <span id="sim-match-cnt" class="mono" style="color:var(--dim);font-weight:500"></span>
        </label>
        <div class="search-box">
          <input id="sim-search" type="text" placeholder="Type name, college, or year..." autocomplete="off">
          <button id="sim-search-clear" class="search-clear" title="Clear search">&times;</button>
          <select id="sim-cls-filter" class="search-cls-sel" title="Filter by Draft Class"></select>
        </div>
        <div class="p-list" id="sim-prospect-list"></div>
      </div>

      <div class="sim-field">
        <label><span>2. NFL Roster Season</span></label>
        <select id="sim-env-year"></select>
      </div>
      <div class="sim-field">
        <label><span>3. Select NFL Team (32 Franchises)</span></label>
        <div class="teams" id="sim-teams-grid"></div>
        <div class="quick-row">
          <button class="btn" id="sim-btn-best" title="Find team with highest positive gain over Base">Best Fit</button>
          <button class="btn" id="sim-btn-worst" title="Find most congested depth chart">Worst Fit</button>
          <button class="btn" id="sim-btn-drafted" title="Reset to actual drafted team &amp; year">Drafted Spot</button>
        </div>
      </div>
      <div class="sim-field" style="border-top:1px solid var(--line);padding-top:12px;margin-top:12px">
        <label><span>4. Prospect Alignment Override</span></label>
        <div class="seg" id="sim-role-seg" style="width:100%">
          <button data-role="auto" class="on" style="flex:1">Auto</button>
          <button data-role="Boundary X" style="flex:1">X</button>
          <button data-role="Flanker Z" style="flex:1">Z</button>
          <button data-role="Interior Slot" style="flex:1">Slot</button>
        </div>
      </div>
      <div style="border-top:1px solid var(--line);padding-top:12px;margin-top:12px;display:flex;flex-direction:column;gap:6px">
        <button class="btn gold" id="sim-btn-pin" style="width:100%">Pin Scenario to Rankings Board</button>
        <button class="btn" id="sim-btn-reset-room" style="width:100%">Restore Full Returning Roster</button>
      </div>
    </div>

    <!-- Right Column: Interactive Starting Lineup + Base vs Fit + Career Sim -->
    <div id="sim-main-panel"></div>
  </div>
</div>

<!-- Quick Team-Swap Drawer for Tab 1 -->
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
const sgnYds = v => (v >= 0 ? '+' : '') + Math.round(v).toLocaleString();
const clip = (v, a, b) => Math.min(b, Math.max(a, v));
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const TEAM_CODES = [
  "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
  "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
  "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
  "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
];

const st = {
  tab: 'board',
  sort: 'grade',
  view: 'fit',
  cls: 'all',
  arch: 'all',
  q: '',
  sel: P[0] || null,
  preview: (P[0] && P[0].team) ? P[0].team : 'KC',
  simCls: 'all',
  simQ: '',
  simProspectId: P[0] ? P[0].id : 'p0',
  simTeam: (P[0] && P[0].team) ? P[0].team : 'KC',
  simYear: P[0] ? P[0].year : 2025,
  simRoleOverride: 'auto',
  simQbMode: 'actual', // 'actual' | 'elite' | 'bridge'
  benchedVets: {},     // key: vet name -> true when traded/benched in lineup game
  singleRoll: null,
  rollHistory: []
};

const SORT_DESC = {
  grade: 'ranked by unified translation grade',
  hit: 'ranked by "If He Hits" 3-year starter projection',
  ceiling: 'ranked by 90th-percentile ceiling (p90)',
  floor: 'ranked by 10th-percentile floor (p10)',
  boom: 'ranked by probability of a 2,500+ yard boom',
  safe: 'ranked by lowest probability of a sub-500 yard bust',
  actual: 'ranked by actual first-3-season NFL receiving yards'
};

function ink(hex) {
  if (!hex || hex[0] !== '#') return '#fff';
  const n = parseInt(hex.slice(1), 16), r = n >> 16, g = (n >> 8) & 255, b = n & 255;
  return (0.299 * r + 0.587 * g + 0.114 * b) > 155 ? '#09090b' : '#fff';
}
function ftIn(h) { return h == null ? '–' : Math.floor(h / 12) + "'" + Math.round(h % 12) + '"'; }

// ---- Split-Normal Sqrt-Scale Math & Continuous Probabilities ----------------
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
  const hit = Math.round(0.48 * p50 + 0.52 * p90);
  const miss = Math.round(0.74 * p10 + 0.26 * p50);
  return { starter, boom, bust, hit, miss };
}

// ---- Stage 2 Ex-Ante Team Fit & Lineup Builder (Relative to Base!) ----------
function mockFit(p, code, envYear, opts) {
  const options = opts || {};
  const yr = envYear || p.year;
  const key = code + '_' + yr;
  const env = ENVS[key] || ENVS[code + '_' + META.last_complete] || {
    name: code, city: code, color: '#3f3f46', year: yr, qb: 'Starting QB',
    qb_yds: 3600, qb_td: 24, tot_att: 540,
    proe: 0.0, qb_epa: 0.0, vacated: 0.24, style: 'Balanced Pro-Style', incumbents: [], departed: []
  };

  let rVec = [p.px, p.pz, p.ps];
  let activeArch = p.arch;
  if (options.roleOverride && options.roleOverride !== 'auto') {
    activeArch = options.roleOverride;
    if (activeArch === 'Boundary X') rVec = [0.78, 0.14, 0.08];
    else if (activeArch === 'Flanker Z') rVec = [0.14, 0.76, 0.10];
    else if (activeArch === 'Interior Slot') rVec = [0.06, 0.14, 0.80];
  }

  const benchedMap = options.benchedVets || {};
  let extraVacated = 0.0;
  const allIncs = env.incumbents || [];
  const activeIncs = [];
  allIncs.forEach(inc => {
    if (benchedMap[inc.name]) {
      extraVacated += inc.target_share;
    } else {
      activeIncs.push(inc);
    }
  });

  let effQbEpa = env.qb_epa;
  let effProe = env.proe;
  if (options.qbMode === 'elite') {
    effQbEpa = Math.max(env.qb_epa, 1.35);
    effProe = Math.max(env.proe, 0.65);
  } else if (options.qbMode === 'bridge') {
    effQbEpa = -1.05;
    effProe = Math.min(env.proe, -0.45);
  }

  const effVacated = clip(env.vacated + extraVacated, 0.04, 0.78);
  const rawQbVol = (effProe * 0.055) + (effQbEpa * 0.075);
  const rawVacancy = (effVacated - 0.24) * 0.38;

  let congestionPen = 0.0;
  let synergyBon = 0.0;
  let notes = [];

  const wrIncShare = activeIncs.filter(i => i.pos === 'WR').reduce((s, i) => s + i.target_share, 0.0);

  activeIncs.forEach(inc => {
    const vVec = inc.vector || [0.33, 0.33, 0.33];
    const dot = rVec[0]*vVec[0] + rVec[1]*vVec[1] + rVec[2]*vVec[2];
    const share = inc.target_share;
    if (share >= 0.18 && dot >= 0.45) {
      const pen = (share - 0.13) * Math.pow(dot, 1.3) * 0.48;
      congestionPen += pen;
      notes.push('Clash w/ ' + inc.name + ' (-' + (pen*100).toFixed(1) + '%)');
    } else if (share >= 0.18 && dot <= 0.29) {
      const syn = share * (1.0 - dot) * 0.22;
      synergyBon += syn;
      notes.push('Synergy w/ ' + inc.name + ' (+' + (syn*100).toFixed(1) + '%)');
    }
  });

  if (wrIncShare <= 0.28) {
    const rBon = p.grade >= 90.0 ? 0.065 : 0.035;
    synergyBon += rBon;
    notes.push('Open WR1 Runway');
  } else if (effVacated >= 0.34 && wrIncShare <= 0.38) {
    synergyBon += 0.045;
    notes.push('High Vacancy (' + Math.round(effVacated*100) + '%)');
  }

  const rawRole = synergyBon - congestionPen;
  const upsideFactor = Math.max(0.75, Math.min(1.35, Math.pow(p.grade / 80.0, 1.25)));
  const downsideInsulation = Math.max(0.45, Math.min(1.0, 1.0 - Math.max(0.0, (p.grade - 86.0) * 0.045)));

  const dQb = rawQbVol >= 0 ? rawQbVol * upsideFactor : rawQbVol * downsideInsulation;
  const dVac = rawVacancy >= 0 ? rawVacancy * upsideFactor : rawVacancy * downsideInsulation;
  const dRole = rawRole * (rawRole > 0 ? 1.10 : 0.85);

  const mult = clip(1.0 + dQb + dVac + dRole, 0.82, 1.28);
  const p10 = Math.round(clip(p.n_p10 * (1.0 + dRole * 1.15), 0, 1400));
  const p50 = Math.round(clip(p.n_p50 * mult, p10 + 180, 3480));
  const ceilExp = mult >= 1.0 ? 1.07 : 0.86;
  const p90 = Math.round(clip(p.n_p90 * Math.pow(mult, ceilExp), p50 + 750, 4950));

  const probs = deriveProbs(p10, p50, p90);
  // EXACT DELTAS RELATIVE TO BASE (Team-Neutral Stage 1.5), NEVER vs Drafted!
  const deltaBaseHit = probs.hit - p.n_hit;
  const deltaBaseP90 = p90 - p.n_p90;
  const deltaBaseP10 = p10 - p.n_p10;

  const roleLabel = rawRole > 0.03 ? 'Synergy' : (rawRole < -0.03 ? 'Congested' : 'Balanced');
  return Object.assign(
    {
      p10, p50, p90, mult, dQb, dVac, dRole, effVacated, effQbEpa, effProe,
      deltaBaseHit, deltaBaseP90, deltaBaseP10, activeArch,
      roleLabel, notes: notes.join(' · ') || 'Balanced Depth Chart',
      team: env, code, envYear: yr, activeIncs
    },
    probs
  );
}

// ---- Build 11-Personnel Starting Lineup with Prospect Inserted --------------
function buildStartingLineup(p, m) {
  const env = m.team;
  const allIncs = env.incumbents || [];
  const estProjShare = clip((m.hit / 3.0) / Math.max(env.qb_yds || 3600, 2900) * 0.92, 0.12, 0.31);
  const prospectNode = {
    isProspect: true,
    name: p.name,
    pos: 'WR',
    arch: m.activeArch,
    ovr: Math.round(p.grade),
    target_share: estProjShare,
    yds: Math.round(m.hit / 3.0),
    rec: Math.round((m.hit / 3.0) / clip(p.ypc * 0.88, 11.5, 16.5)),
    benched: false
  };

  const wrPool = [prospectNode];
  let te1Node = null;

  allIncs.forEach(inc => {
    const isBenched = !!(st.benchedVets && st.benchedVets[inc.name]);
    const copy = Object.assign({}, inc, { isProspect: false, benched: isBenched });
    if (inc.pos === 'TE' && !te1Node && !isBenched) {
      te1Node = copy;
    } else {
      wrPool.push(copy);
    }
  });

  if (!te1Node) {
    const anyTe = allIncs.find(i => i.pos === 'TE');
    te1Node = anyTe
      ? Object.assign({}, anyTe, { isProspect: false, benched: !!(st.benchedVets && st.benchedVets[anyTe.name]) })
      : { isProspect: false, name: 'Starting TE', pos: 'TE', arch: 'Inline TE', ovr: 74, target_share: 0.12, yds: 520, rec: 46, benched: false, synthetic: true };
  }

  const activeWrs = wrPool.filter(w => !w.benched).sort((a, b) => (b.target_share - a.target_share) || (b.ovr - a.ovr));
  const benchedWrs = wrPool.filter(w => w.benched);

  const rawSum = activeWrs.slice(0, 3).reduce((s, w) => s + w.target_share, 0) + (te1Node.benched ? 0 : te1Node.target_share);
  const scale = rawSum > 0.78 ? (0.76 / rawSum) : 1.0;

  const slots = [
    { slot: 'WR1 · Boundary X', player: activeWrs[0] || null },
    { slot: 'WR2 · Flanker Z',  player: activeWrs[1] || null },
    { slot: 'WR3 · Slot',       player: activeWrs[2] || null },
    { slot: 'TE1 · Seam / Y',   player: te1Node }
  ];

  const benchList = activeWrs.slice(3).concat(benchedWrs);
  return { slots, benchList, scale };
}

// ---- Monte Carlo Engine (1,000-Trial Distribution + Single Career Roller) ---
function drawSqrtSplitNormal(s10, s50, s90) {
  const sl = Math.max(3, (s50 - s10) / 1.38), sr = Math.max(3.8, (s90 - s50) / 1.38);
  const u1 = Math.max(1e-7, Math.random()), u2 = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  const sDraw = Math.max(0, s50 + z * (z < 0 ? sl : sr));
  return Math.round(clip(sDraw * sDraw, 0, 5200));
}

function runMonteCarlo1000(m) {
  const N = 1000;
  const s10 = Math.sqrt(Math.max(0, m.p10)), s50 = Math.sqrt(Math.max(25, m.p50)), s90 = Math.sqrt(Math.max(100, m.p90));
  const bins = new Array(10).fill(0);
  const samples = [];
  for (let i = 0; i < N; i++) {
    const y = drawSqrtSplitNormal(s10, s50, s90);
    samples.push(y);
    const bIdx = Math.min(9, Math.floor(y / 500));
    bins[bIdx]++;
  }
  samples.sort((a, b) => a - b);
  return { bins, samples };
}

function rollSingleCareer(p, m) {
  const s10 = Math.sqrt(Math.max(0, m.p10)), s50 = Math.sqrt(Math.max(25, m.p50)), s90 = Math.sqrt(Math.max(100, m.p90));
  const totYds = drawSqrtSplitNormal(s10, s50, s90);

  const sy = Math.sqrt(Math.max(0, totYds));
  const sl = Math.max(3, (s50 - s10) / 1.38), sr = Math.max(3.8, (s90 - s50) / 1.38);
  const z = (sy - s50) / (sy < s50 ? sl : sr);
  const pctile = Math.round(clip(0.5 * (1 + erf(z / Math.SQRT2)) * 100, 1, 99));

  const w1 = 0.24 + Math.random() * 0.08;
  const w2 = 0.34 + Math.random() * 0.06;
  const y1 = Math.round(totYds * w1);
  const y2 = Math.round(totYds * w2);
  const y3 = Math.max(0, totYds - y1 - y2);

  const baseYpc = clip((p.ypc || 14.0) * 0.82 + (p.px * 1.4 + p.pz * 0.9 - p.ps * 0.8), 10.4, 17.5);
  const mkSeason = (yrNum, yds) => {
    const ypc = +(baseYpc + (Math.random() - 0.5) * 1.3).toFixed(1);
    const rec = Math.max(0, Math.round(yds / ypc));
    const tdRate = clip(0.058 + p.px * 0.03 + (Math.random() - 0.45) * 0.03, 0.02, 0.14);
    const td = Math.max(0, Math.round(rec * tdRate));
    return { yr: p.year + yrNum - 1, rec, yds, td, ypc: rec > 0 ? (yds / rec).toFixed(1) : '0.0' };
  };

  const s1 = mkSeason(1, y1), s2 = mkSeason(2, y2), s3 = mkSeason(3, y3);
  const totRec = s1.rec + s2.rec + s3.rec;
  const totTd = s1.td + s2.td + s3.td;

  let outcome = 'Rotational WR3', verdict = 'Rookie Deal Expires · Test Free Agency', isGold = false;
  if (totYds >= 3200) {
    outcome = 'All-Pro WR1 Alpha';
    verdict = 'Record Extension · 4 yrs, $140M ($35M/yr)';
    isGold = true;
  } else if (totYds >= 2500) {
    outcome = 'Pro Bowl WR1 Boom';
    verdict = '5th-Year Option + $28M/yr Extension';
    isGold = true;
  } else if (totYds >= 1500) {
    outcome = 'Quality NFL Starter (WR2)';
    verdict = '2nd Contract · 3 yrs, $54M ($18M/yr)';
  } else if (totYds < 500) {
    outcome = 'Rookie-Contract Bust';
    verdict = 'Waived Before Year 4 · Practice Squad';
  }

  return {
    totYds, totRec, totTd,
    totYpc: totRec > 0 ? (totYds / totRec).toFixed(1) : '0.0',
    seasons: [s1, s2, s3],
    outcome, verdict, isGold, pctile,
    binIdx: Math.min(9, Math.floor(totYds / 500)),
    teamCode: m.code
  };
}

function eff(p) {
  if (p.mock_fit) return p.mock_fit;
  if (st.view === 'fit') {
    return {
      p10: p.s_p10, p50: p.s_p50, p90: p.s_p90,
      hit: p.s_hit, miss: p.s_miss,
      starter: p.s_st, boom: p.s_bm, bust: p.s_bu,
      deltaBaseHit: p.s_hit - p.n_hit
    };
  }
  return {
    p10: p.n_p10, p50: p.n_p50, p90: p.n_p90,
    hit: p.n_hit, miss: p.n_miss,
    starter: p.n_st, boom: p.n_bm, bust: p.n_bu,
    deltaBaseHit: 0
  };
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
    case 'actual': return ['Actual', p.done ? fmt(p.actual) : '–'];
    default: return ['Grade', p.grade.toFixed(1)];
  }
}

function rangeTrackHtml(p, e) {
  const x = v => clip((v / SCALE) * 100, 0, 100);
  let actDot = '';
  let actTxt = '';
  if (p.done) {
    const inside = p.actual >= e.p10 && p.actual <= e.p90;
    actDot = '<i class="rng-actual ' + (inside ? 'in' : 'out') + '" style="left:' + x(p.actual) + '%" title="Actual Y1–3: ' + fmt(p.actual) + ' yds"></i>';
    actTxt = '<span class="' + (inside ? 'pos' : 'neg') + '">Actual: ' + fmt(p.actual) + ' yds</span>';
  } else {
    actTxt = '<span>' + pct(e.starter) + ' Start · ' + pct(e.boom) + ' Boom</span>';
  }

  return '<div class="rng-wrap">' +
    '<div class="rng-top mono">' +
      '<span class="rng-endpoints">' + fmt(e.p10) + ' <span style="color:var(--dim)">floor</span></span>' +
      '<span class="rng-hit-lbl">If Hits: <span>' + fmt(e.hit) + '</span> <span style="color:var(--dim);font-weight:400">(Base ' + fmt(p.n_hit) + ')</span></span>' +
      '<span class="rng-endpoints"><span style="color:var(--dim)">ceil</span> ' + fmt(e.p90) + '</span>' +
    '</div>' +
    '<div class="rng-track">' +
      '<div class="rng-band" style="left:' + x(e.p10) + '%;width:' + Math.max(1.5, x(e.p90) - x(e.p10)) + '%"></div>' +
      '<div class="rng-marker" style="left:' + x(e.hit) + '%"></div>' +
      actDot +
    '</div>' +
    '<div class="rng-sub mono">' +
      '<span>If Misses: ~' + fmt(e.miss) + ' yds (' + pct(e.bust) + ' risk)</span>' +
      actTxt +
    '</div>' +
  '</div>';
}

function renderBoard() {
  const list = visible();
  $('count').textContent = list.length;
  $('sortdesc').textContent = SORT_DESC[st.sort];

  const done = list.filter(x => x.p.done);
  if (done.length >= 5) {
    const inside = done.filter(x => x.p.actual >= x.e.p10 && x.p.actual <= x.e.p90).length;
    $('calib').innerHTML = 'Completed classes (n=' + done.length + '): <b>' + Math.round(inside / done.length * 100) + '%</b> landed inside p10&ndash;p90 range';
  } else { $('calib').textContent = ''; }

  const mocked = P.filter(p => p.mock_fit).length;
  $('mock-pill').style.display = mocked ? '' : 'none';
  $('btn-reset-mocks').style.display = mocked ? '' : 'none';
  $('mock-pill').textContent = mocked + ' mock' + (mocked > 1 ? 's' : '') + ' pinned';

  $('empty').style.display = list.length ? 'none' : 'block';
  $('rows').innerHTML = list.map((x, i) => {
    const p = x.p, e = x.e, m = metric(p, e);
    const pickTxt = (p.pick && p.pick < 500) ? 'Rd ' + p.round + ' #' + p.pick : 'Pre-Draft';
    const activeTeam = p.mock_fit ? p.mock_fit.code : p.team;
    const tmCol = activeTeam ? ((ENVS[activeTeam + '_' + p.year] || {}).color || '#27272a') : '';
    const tmBadge = activeTeam
      ? ' &middot; <span class="tm-badge" style="background:' + tmCol + ';color:' + ink(tmCol) + '">' + esc(activeTeam) + '</span>'
      : '';
    const tags = (p.no40 ? '<span class="tag" title="No 40-yard dash on record">No 40</span>' : '') +
      (p.mock_fit ? '<span class="tag gold">Pinned: ' + esc(p.mock_fit.code) + ' \'' + String(p.mock_fit.envYear).slice(-2) + '</span>' : '');

    const deltaBase = e.hit - p.n_hit;
    const deltaHtml = (st.view === 'neutral' && !p.mock_fit)
      ? '<small>Base Neutral</small>'
      : '<small class="' + (deltaBase >= 0 ? 'pos' : 'neg') + '">' + sgnYds(deltaBase) + ' vs Base</small>';

    return '<div class="row" data-id="' + p.id + '">' +
      '<div class="rk mono">' + String(i + 1).padStart(2, '0') + '</div>' +
      '<div style="min-width:0"><div class="nm">' + esc(p.name) + '<span class="tag">' + esc(p.arch) + '</span>' + tags + '</div>' +
      '<div class="sub">' + esc(p.college) + ' &middot; ' + p.year + tmBadge + ' &middot; ' + pickTxt + '</div></div>' +
      '<div>' + rangeTrackHtml(p, e) + '</div>' +
      '<div class="hit-col mono"><b>' + fmt(e.hit) + ' yds</b>' + deltaHtml + '</div>' +
      '<div class="met"><small>' + m[0] + '</small><b class="mono">' + m[1] + '</b></div></div>';
  }).join('');
}

function findExtremeTeam(p, yr, best) {
  let chosen = 'KC', bestVal = best ? -1e9 : 1e9;
  TEAM_CODES.forEach(c => {
    const res = mockFit(p, c, yr);
    if ((best && res.hit > bestVal) || (!best && res.hit < bestVal)) {
      bestVal = res.hit;
      chosen = c;
    }
  });
  return chosen;
}

// ---- Tab 1 Quick Team-Swap Drawer (Deltas Strictly vs. Base!) ----------------
function renderDrawer() {
  const p = st.sel; if (!p) return;
  const m = mockFit(p, st.preview, p.year);
  const neutral = { p10: p.n_p10, p50: p.n_p50, p90: p.n_p90, hit: p.n_hit, miss: p.n_miss, starter: p.n_st, boom: p.n_bm, bust: p.n_bu };
  const drafted = { p10: p.s_p10, p50: p.s_p50, p90: p.s_p90, hit: p.s_hit, miss: p.s_miss, starter: p.s_st, boom: p.s_bm, bust: p.s_bu };
  const shiftCls = v => (v >= 0 ? 'pos' : 'neg');

  const mockVsBase = m.hit - neutral.hit;
  const draftedVsBase = drafted.hit - neutral.hit;
  const incNames = (m.team.incumbents || []).slice(0, 3).map(i => i.name + ' (' + Math.round(i.target_share*100) + '%)').join(', ');

  const row = (lbl, e, dBase) => '<tr><td>' + lbl + '</td><td class="mono">' + fmt(e.p10) + ' &ndash; ' + fmt(e.p90) +
    '</td><td class="mono"><b class="pos">' + fmt(e.hit) + '</b></td><td class="mono ' + shiftCls(dBase) + '">' + (dBase === 0 ? 'Base' : sgnYds(dBase)) +
    '</td><td class="mono">' + pct(e.starter) + '</td><td class="mono">' + pct(e.boom) + '</td></tr>';

  const ath = '<div class="grid4">' +
    '<div class="cell"><small>Ht / Wt</small><b class="mono">' + ftIn(p.ht) + ' / ' + (p.wt == null ? '–' : Math.round(p.wt)) + '</b></div>' +
    '<div class="cell"><small>40-Yard</small><b class="mono">' + (p.forty == null ? 'No 40' : p.forty.toFixed(2) + 's') + '</b></div>' +
    '<div class="cell"><small>HaSS</small><b class="mono">' + (p.hass == null ? '–' : p.hass.toFixed(1)) + '</b></div>' +
    '<div class="cell"><small>Burst</small><b class="mono">' + (p.burst == null ? '–' : p.burst.toFixed(0)) + '</b></div></div>';

  $('drawer-body').innerHTML =
    '<div class="dh"><div><span class="lbl">Boom or Bust &middot; Prospect Overview</span><h2>' + esc(p.name) + '</h2>' +
    '<div class="sub">' + esc(p.college) + ' &middot; Class of ' + p.year + ((p.pick && p.pick < 500) ? ' &middot; Rd ' + p.round + ' #' + p.pick : ' &middot; Pre-Draft') + (p.team ? ' &middot; ' + esc(p.team) : '') + '</div>' +
    '<div style="margin-top:6px"><span class="tag" style="margin-left:0">' + esc(p.arch) + '</span><span class="tag gold">Grade ' + p.grade.toFixed(1) + '</span></div></div>' +
    '<button class="btn" id="btn-close" style="align-self:flex-start">&times;</button></div>' +

    '<div class="sec"><h4>3-Year Outcome Comparison (Deltas vs. Base)</h4>' +
    '<table class="tbl"><tr><th>Context</th><th>p10 &ndash; p90 Range</th><th>If Hits</th><th>vs Base</th><th>Start</th><th>Boom</th></tr>' +
    row('Base (Neutral)', neutral, 0) +
    row('Drafted' + (p.team ? ' (' + esc(p.team) + ')' : ''), drafted, draftedVsBase) +
    row('What-If (' + esc(m.code) + ')', m, mockVsBase) +
    '</table></div>' +

    '<div class="sec"><h4>Quick Landing-Spot Swap (' + p.year + ' Rosters)</h4>' +
    '<div class="teams">' + TEAM_CODES.map(c => {
      const col = (ENVS[c + '_' + p.year] || {}).color || '#3f3f46';
      return '<button class="tb' + (c === st.preview ? ' on' : '') + '" data-team="' + c + '" style="background:' + col + ';color:' + ink(col) + '">' + c + '</button>';
    }).join('') + '</div>' +
    '<div class="grid3" style="margin-top:8px">' +
      '<div class="cell gold-cell"><small>What-If "If Hits"</small><b class="mono pos">' + fmt(m.hit) + ' yds</b><span class="sub-note">Base: ' + fmt(neutral.hit) + ' yds</span></div>' +
      '<div class="cell"><small>Change vs. Base</small><b class="mono ' + shiftCls(mockVsBase) + '">' + sgnYds(mockVsBase) + ' yds</b><span class="sub-note">' + sgn((m.hit - neutral.hit) / Math.max(1, neutral.hit)) + ' shift</span></div>' +
      '<div class="cell"><small>Fit Multiplier</small><b class="mono">' + m.mult.toFixed(2) + 'x</b><span class="sub-note">' + esc(m.roleLabel) + '</span></div>' +
    '</div>' +
    '<div class="note"><b>' + esc(m.team.city) + ' ' + esc(m.team.name) + ' (' + p.year + ')</b> &middot; QB: ' + esc(m.team.qb) + ' &middot; ' + Math.round(m.team.vacated * 100) + '% vacated' + (incNames ? ' &middot; Returning: ' + esc(incNames) : '') + '</div>' +
    '<div class="acts">' +
      '<button class="btn gold" id="btn-apply" style="flex:1">Pin to Board</button>' +
      '<button class="btn primary" id="btn-open-sim">Open in Lineup Lab &rarr;</button>' +
      '<button class="btn" id="btn-clear-mock">Reset</button>' +
    '</div></div>' +

    '<div class="sec"><h4>Combine &amp; Biomechanics</h4>' + ath + '</div>';
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

// ---- Tab 2: Clean Search + Scrollable Prospect Picker & Lineup Lab ----------
function getFilteredSimProspects() {
  const q = (st.simQ || '').toLowerCase().trim();
  return P.filter(p => {
    if (st.simCls !== 'all' && String(p.year) !== st.simCls) return false;
    if (!q) return true;
    return (
      p.name.toLowerCase().includes(q) ||
      String(p.college).toLowerCase().includes(q) ||
      String(p.year).includes(q) ||
      String(p.team).toLowerCase().includes(q)
    );
  });
}

function selectSimProspect(id, scrollIntoView) {
  const p = P.find(x => x.id === id);
  if (!p) return;
  st.simProspectId = p.id;
  st.simTeam = p.mock_fit ? p.mock_fit.code : (p.team || 'KC');
  st.simYear = p.mock_fit ? p.mock_fit.envYear : p.year;
  st.benchedVets = {};
  st.simRoleOverride = 'auto';
  st.simQbMode = 'actual';
  st.singleRoll = null;
  st.rollHistory = [];
  $('sim-env-year').value = String(st.simYear);
  document.querySelectorAll('#sim-role-seg button').forEach(b => b.classList.toggle('on', b.dataset.role === st.simRoleOverride));
  renderSimProspectList(scrollIntoView);
  renderSimTab();
}

function renderSimProspectList(scrollSelected) {
  const matches = getFilteredSimProspects();
  const activeP = P.find(x => x.id === st.simProspectId) || P[0];
  $('sim-match-cnt').textContent = matches.length + ' found';
  $('sim-search-clear').style.display = st.simQ ? 'block' : 'none';

  // Always ensure currently selected prospect is visible or show top 60 matches
  const shown = matches.slice(0, 60);
  if (activeP && !shown.some(x => x.id === activeP.id) && !st.simQ && st.simCls === 'all') {
    shown.unshift(activeP);
  }

  if (!shown.length) {
    $('sim-prospect-list').innerHTML = '<div style="padding:18px 8px;text-align:center;color:var(--dim);font-size:11.5px">No receivers match "' + esc(st.simQ) + '"</div>';
    return;
  }

  const shortArch = a => (a === 'Boundary X' ? 'X' : (a === 'Flanker Z' ? 'Z' : 'SL'));

  $('sim-prospect-list').innerHTML = shown.map(p => {
    const isOn = p.id === st.simProspectId;
    return '<div class="p-item' + (isOn ? ' on' : '') + '" data-pid="' + p.id + '">' +
      '<div class="p-item-l">' +
        '<div class="p-item-nm">' + esc(p.name) + '</div>' +
        '<div class="p-item-sub">' + esc(p.college) + ' &middot; ' + p.year + (p.team ? ' &middot; ' + esc(p.team) : '') + '</div>' +
      '</div>' +
      '<div class="p-item-r">' +
        '<span class="tag" style="margin:0;padding:0 5px">' + shortArch(p.arch) + '</span>' +
        '<span class="p-item-grd mono">' + p.grade.toFixed(1) + '</span>' +
      '</div>' +
    '</div>';
  }).join('');

  if (scrollSelected) {
    const activeEl = $('sim-prospect-list').querySelector('.p-item.on');
    if (activeEl) activeEl.scrollIntoView({ block: 'nearest' });
  }
}

function populateSimSelectors() {
  const years = ['all'].concat(META.years.map(String));
  $('sim-cls-filter').innerHTML = years.map(y => '<option value="' + y + '"' + (y === st.simCls ? ' selected' : '') + '>' + (y === 'all' ? 'All Yrs' : y) + '</option>').join('');
  $('sim-env-year').innerHTML = META.years.map(y => '<option value="' + y + '"' + (y === st.simYear ? ' selected' : '') + '>' + y + ' NFL Roster &amp; Environment</option>').join('');
  $('sim-search').value = st.simQ;
  document.querySelectorAll('#sim-role-seg button').forEach(b => b.classList.toggle('on', b.dataset.role === st.simRoleOverride));
  renderSimProspectList(true);
}

function renderSimTab() {
  const p = P.find(x => x.id === st.simProspectId) || P[0];
  if (!p) return;

  const m = mockFit(p, st.simTeam, st.simYear, {
    roleOverride: st.simRoleOverride,
    qbMode: st.simQbMode,
    benchedVets: st.benchedVets
  });
  const shiftCls = v => (v >= 0 ? 'pos' : 'neg');

  $('sim-teams-grid').innerHTML = TEAM_CODES.map(c => {
    const col = (ENVS[c + '_' + st.simYear] || {}).color || '#3f3f46';
    return '<button class="tb' + (c === st.simTeam ? ' on' : '') + '" data-simteam="' + c + '" style="background:' + col + ';color:' + ink(col) + '">' + c + '</button>';
  }).join('');

  const lineup = buildStartingLineup(p, m);
  const lineupCardsHtml = lineup.slots.map(s => {
    const pl = s.player;
    if (!pl) {
      return '<div class="p-card"><div class="p-card-top"><span class="slot-pill">' + esc(s.slot) + '</span><span class="ovr-badge mono">--</span></div><div class="p-card-name" style="color:var(--dim)">Open Depth Slot</div><div class="p-card-arch">Vacated Role</div></div>';
    }
    const adjShare = Math.round(pl.target_share * lineup.scale * 100);
    if (pl.isProspect) {
      return '<div class="p-card prospect">' +
        '<div>' +
          '<div class="p-card-top"><span class="slot-pill">' + esc(s.slot) + '</span><span class="ovr-badge mono">' + pl.ovr + ' GRD</span></div>' +
          '<div class="p-card-name">' + esc(pl.name) + '</div>' +
          '<div class="p-card-arch">ROOKIE &middot; ' + esc(pl.arch) + '</div>' +
        '</div>' +
        '<div>' +
          '<div class="p-card-stats mono"><span>~' + adjShare + '% Tgt Share</span><b class="pos">' + fmt(pl.yds) + ' yds/yr</b></div>' +
          '<div class="sh-bar"><div class="sh-fill" style="width:' + Math.min(100, adjShare * 2.8) + '%"></div></div>' +
          '<div class="act-hint" style="color:var(--gold)">★ Inserted Prospect</div>' +
        '</div>' +
      '</div>';
    }
    return '<div class="p-card vet' + (pl.benched ? ' benched' : '') + '" data-vet="' + esc(pl.name) + '" title="Click to Trade / Bench ' + esc(pl.name) + ' and vacate their targets">' +
      '<div>' +
        '<div class="p-card-top"><span class="slot-pill">' + esc(s.slot) + '</span><span class="ovr-badge mono">' + pl.ovr + ' OVR</span></div>' +
        '<div class="p-card-name">' + esc(pl.name) + '</div>' +
        '<div class="p-card-arch">' + esc(pl.pos) + ' &middot; ' + esc(pl.arch) + '</div>' +
      '</div>' +
      '<div>' +
        '<div class="p-card-stats mono"><span>' + adjShare + '% Share</span><b>' + fmt(pl.yds) + ' yds</b></div>' +
        '<div class="sh-bar"><div class="sh-fill" style="width:' + Math.min(100, adjShare * 2.8) + '%"></div></div>' +
        '<div class="act-hint">' + (pl.benched ? '↺ Click to Restore' : '✕ Click to Trade / Bench') + '</div>' +
      '</div>' +
    '</div>';
  }).join('');

  const benchHtml = lineup.benchList.length
    ? '<div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap;align-items:center">' +
        '<span class="lbl" style="margin-right:4px">WR4 / Bench / Traded:</span>' +
        lineup.benchList.map(b =>
          '<button class="btn" data-vet="' + esc(b.name) + '" style="padding:3px 8px;font-size:11px;' + (b.benched ? 'border-color:var(--gold-bd);color:var(--gold)' : '') + '">' +
            esc(b.name) + ' (' + Math.round(b.target_share*100) + '% · ' + fmt(b.yds) + ' yds) ' + (b.benched ? '[Traded &mdash; Restore]' : '[Bumped to Bench &mdash; Trade]') +
          '</button>'
        ).join('') +
      '</div>'
    : '';

  const mc = runMonteCarlo1000(m);
  const maxBin = Math.max(1, ...mc.bins);
  const binLabels = ['<500', '500–999', '1.0–1.5k', '1.5–2.0k', '2.0–2.5k', '2.5–3.0k', '3.0–3.5k', '3.5–4.0k', '4.0–4.5k', '4.5k+'];

  if (!st.singleRoll) {
    st.singleRoll = rollSingleCareer(p, m);
  }
  const sr = st.singleRoll;

  const histHtml = mc.bins.map((cnt, idx) => {
    const hPct = Math.max(3, Math.round((cnt / maxBin) * 100));
    const cls = (idx === 0 ? 'bust' : (idx >= 5 ? 'boom' : '')) + (sr && sr.binIdx === idx ? ' active-bin' : '');
    const pctVal = (cnt / 10).toFixed(1) + '%';
    return '<div class="hbar-col"><div class="hbar ' + cls + '" style="height:' + hPct + '%"><span class="mono">' + pctVal + '</span></div></div>';
  }).join('');

  const tmCol = m.team.color || '#27272a';
  const basePctShift = ((m.hit - p.n_hit) / Math.max(1, p.n_hit));

  $('sim-main-panel').innerHTML =
    '<div class="sim-card">' +
      /* ---- Header Banner ---- */
      '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:14px">' +
        '<div>' +
          '<div style="display:flex;align-items:center;gap:8px">' +
            '<span class="tm-badge" style="background:' + tmCol + ';color:' + ink(tmCol) + ';padding:2px 8px;font-size:11px">' + esc(m.code) + ' ' + st.simYear + '</span>' +
            '<span class="lbl">Boom or Bust &middot; Lineup &amp; Career Sim Lab</span>' +
          '</div>' +
          '<h2 style="margin:5px 0 2px;font-size:22px;font-weight:800">' + esc(p.name) + ' &rarr; ' + esc(m.team.city) + ' ' + esc(m.team.name) + '</h2>' +
          '<div class="sub">' + esc(p.college) + ' &middot; Grade <b>' + p.grade.toFixed(1) + '</b> &middot; Base "If Hits": <b class="mono">' + fmt(p.n_hit) + ' yds</b> (' + fmt(p.n_p10) + '&ndash;' + fmt(p.n_p90) + ')</div>' +
        '</div>' +
        '<div style="text-align:right">' +
          '<span class="lbl">Change vs. Base (Neutral)</span>' +
          '<div class="mono ' + shiftCls(m.deltaBaseHit) + '" style="font-size:22px;font-weight:900">' + sgnYds(m.deltaBaseHit) + ' yds (' + sgn(basePctShift) + ')</div>' +
          '<div class="sub mono">Fit Multiplier: <b>' + m.mult.toFixed(2) + 'x</b></div>' +
        '</div>' +
      '</div>' +

      /* ---- Base vs. Landing-Spot Comparison KPIs (Strictly vs. Base!) ---- */
      '<div class="grid4" style="margin-top:14px">' +
        '<div class="cell"><small>Base "If Hits" (Neutral)</small><b class="mono">' + fmt(p.n_hit) + ' yds</b><span class="sub-note">Base Range: ' + fmt(p.n_p10) + '&ndash;' + fmt(p.n_p90) + '</span></div>' +
        '<div class="cell gold-cell"><small>Landing Spot "If Hits"</small><b class="mono pos">' + fmt(m.hit) + ' yds</b><span class="sub-note">' + sgnYds(m.deltaBaseHit) + ' yds vs Base</span></div>' +
        '<div class="cell"><small>New 80% Range (p10&ndash;p90)</small><b class="mono">' + fmt(m.p10) + ' &ndash; ' + fmt(m.p90) + '</b><span class="sub-note">Ceil ' + sgnYds(m.deltaBaseP90) + ' vs Base</span></div>' +
        '<div class="cell"><small>Start / Boom / Bust</small><b class="mono"><span class="acc">' + pct(m.starter) + '</span> / <span class="pos">' + pct(m.boom) + '</span> / <span class="neg">' + pct(m.bust) + '</span></b><span class="sub-note">Base Boom: ' + pct(p.n_bm) + '</span></div>' +
      '</div>' +

      /* ---- Interactive Madden-Style 11-Personnel Starting Lineup ---- */
      '<div class="field-board">' +
        '<div class="field-hdr">' +
          '<div><b style="font-size:12px;text-transform:uppercase;letter-spacing:.06em">' + st.simYear + ' ' + esc(m.team.city) + ' ' + esc(m.team.name) + ' Starting 11-Personnel Lineup</b><div class="sub" style="margin-top:1px">Click any returning WR/TE card below to Trade / Bench them and vacate their targets</div></div>' +
          '<div class="mono" style="font-size:11px;color:var(--mut)">Vacated Target Pool: <b class="pos">' + Math.round(m.effVacated * 100) + '%</b></div>' +
        '</div>' +
        '<div class="los-line"><span>Line of Scrimmage</span></div>' +
        '<div class="lineup-row">' + lineupCardsHtml + '</div>' +
        benchHtml +
        '<div class="qb-bar">' +
          '<div><span class="lbl">Starting Quarterback &amp; Scheme</span><div style="font-weight:700;font-size:13px;margin-top:2px">QB1: ' + esc(m.team.qb) + ' <span style="color:var(--mut);font-weight:400">(' + fmt(m.team.qb_yds) + ' pass yds, ' + m.team.qb_td + ' TDs &middot; ' + esc(m.team.style) + ')</span></div></div>' +
          '<div style="display:flex;align-items:center;gap:6px">' +
            '<span class="lbl">QB Override:</span>' +
            '<div class="seg" id="sim-qb-seg">' +
              '<button data-qb="actual"' + (st.simQbMode === 'actual' ? ' class="on"' : '') + '>Actual QB (' + sgn(m.dQb) + ')</button>' +
              '<button data-qb="elite"' + (st.simQbMode === 'elite' ? ' class="on"' : '') + '>All-Pro QB</button>' +
              '<button data-qb="bridge"' + (st.simQbMode === 'bridge' ? ' class="on"' : '') + '>Bridge QB</button>' +
            '</div>' +
          '</div>' +
        '</div>' +
        '<div class="grid3" style="margin-top:10px">' +
          '<div class="cell"><small>Pillar 1: QB &amp; Volume</small><b class="mono ' + shiftCls(m.dQb) + '">' + sgn(m.dQb) + ' vs Base</b></div>' +
          '<div class="cell"><small>Pillar 2: Vacated Targets</small><b class="mono ' + shiftCls(m.dVac) + '">' + sgn(m.dVac) + ' vs Base</b></div>' +
          '<div class="cell"><small>Pillar 3: Lineup Synergy</small><b class="mono ' + shiftCls(m.dRole) + '">' + sgn(m.dRole) + ' (' + esc(m.roleLabel) + ')</b></div>' +
        '</div>' +
      '</div>' +

      /* ---- Interactive Rookie Contract Career Roller Game ---- */
      '<div class="career-box">' +
        '<div class="career-hdr">' +
          '<div>' +
            '<span class="lbl">Boom or Bust &middot; Rookie Contract Sim Game (Years 1&ndash;3)</span>' +
            '<div style="font-weight:800;font-size:16px;margin-top:2px">' + esc(p.name) + ' &mdash; Simulated 3-Year Outcome on ' + esc(m.code) + ' <span class="tag gold">' + sr.pctile + 'th Percentile Roll</span></div>' +
          '</div>' +
          '<div style="display:flex;gap:6px;align-items:center">' +
            '<button class="btn gold" id="btn-roll-single">&#9861; Roll 3-Year Career</button>' +
            '<button class="btn" id="btn-roll-5x">Roll 5x Fast</button>' +
          '</div>' +
        '</div>' +
        '<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 11px;background:var(--surface);border:1px solid var(--line);border-radius:5px;margin-bottom:10px;flex-wrap:wrap;gap:8px">' +
          '<div><span class="lbl">Career Tier:</span> <b style="margin-left:6px;color:' + (sr.isGold ? 'var(--gold)' : 'var(--txt)') + '">' + esc(sr.outcome) + '</b></div>' +
          '<div class="verdict-pill' + (sr.isGold ? ' gold' : '') + '">GM Verdict: ' + esc(sr.verdict) + '</div>' +
        '</div>' +
        '<table class="tbl mono">' +
          '<tr><th>Season</th><th>Team</th><th>Receptions</th><th>Receiving Yds</th><th>Yds / Catch</th><th>Rec TDs</th></tr>' +
          sr.seasons.map((s, idx) => '<tr><td>Year ' + (idx + 1) + ' (' + s.yr + ')</td><td>' + esc(m.code) + '</td><td>' + s.rec + '</td><td><b>' + fmt(s.yds) + '</b></td><td>' + s.ypc + '</td><td>' + s.td + '</td></tr>').join('') +
          '<tr style="font-weight:800;border-top:2px solid var(--line-hi)"><td>3-Year Total</td><td>' + esc(m.code) + '</td><td>' + sr.totRec + '</td><td class="' + (sr.isGold ? 'pos' : 'acc') + '">' + fmt(sr.totYds) + ' yds (' + sgnYds(sr.totYds - p.n_hit) + ' vs Base Hit)</td><td>' + sr.totYpc + '</td><td>' + sr.totTd + '</td></tr>' +
        '</table>' +
        (st.rollHistory.length ? (
          '<div class="roll-strip"><span class="lbl">Recent Rolls:</span>' +
          st.rollHistory.map((h, idx) => '<span class="roll-chip mono' + (h.isGold ? ' gold' : '') + '" data-histidx="' + idx + '">' + fmt(h.totYds) + ' yds (' + h.pctile + 'th%)</span>').join('') +
          '</div>'
        ) : '') +
      '</div>' +

      /* ---- 1,000-Trial Monte Carlo Distribution Histogram ---- */
      '<div class="hist-wrap">' +
        '<div style="display:flex;justify-content:space-between;align-items:center">' +
          '<b style="font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim)">1,000-Season Monte Carlo Distribution (Highlighted Bar = Current Roll)</b>' +
          '<button class="btn" id="btn-rerun-mc" style="padding:3px 8px;font-size:11px">Re-Sample 1,000x</button>' +
        '</div>' +
        '<div class="hist-bars">' + histHtml + '</div>' +
        '<div class="hist-lbls mono">' + binLabels.map(l => '<div>' + l + '</div>').join('') + '</div>' +
        '<div class="note" style="margin-top:8px">Dark Gray = Bust (&lt;500 yds) &middot; White/Silver = Rotational/Starter (500&ndash;2,499 yds) &middot; Gold = Boom (&ge;2,500 yds). Outlined bar shows where your current simulated career (' + fmt(sr.totYds) + ' yds) landed.</div>' +
      '</div>' +
    '</div>';
}

function switchTab(tabName) {
  st.tab = tabName;
  document.querySelectorAll('#nav-tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === tabName));
  $('tab-board').style.display = tabName === 'board' ? 'block' : 'none';
  $('tab-sim').style.display = tabName === 'sim' ? 'block' : 'none';
  if (tabName === 'sim') {
    populateSimSelectors();
    renderSimTab();
  } else {
    renderBoard();
  }
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

$('nav-tabs').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  switchTab(b.dataset.tab);
});

$('seg-class').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; st.cls = b.dataset.cls; syncControls(); renderBoard(); });
$('seg-view').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; st.view = b.dataset.view; syncControls(); renderBoard(); });
$('sel-sort').addEventListener('change', e => { st.sort = e.target.value; renderBoard(); });
$('sel-arch').addEventListener('change', e => { st.arch = e.target.value; renderBoard(); });
$('inp-q').addEventListener('input', e => { st.q = e.target.value.toLowerCase().trim(); renderBoard(); });
$('btn-reset').addEventListener('click', () => { st.sort = 'grade'; st.view = 'fit'; st.cls = 'all'; st.arch = 'all'; st.q = ''; syncControls(); renderBoard(); });
$('btn-reset-mocks').addEventListener('click', () => { P.forEach(p => { p.mock_fit = null; }); renderBoard(); });
$('rows').addEventListener('click', e => { const r = e.target.closest('.row'); if (r) openDrawer(r.dataset.id); });
$('backdrop').addEventListener('click', closeDrawer);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

// Drawer Events
$('drawer-body').addEventListener('click', e => {
  const t = e.target.closest('[data-team]');
  if (t) { st.preview = t.dataset.team; renderDrawer(); return; }
  if (e.target.closest('#btn-close')) { closeDrawer(); return; }
  if (e.target.closest('#btn-apply')) { st.sel.mock_fit = mockFit(st.sel, st.preview, st.sel.year); closeDrawer(); renderBoard(); return; }
  if (e.target.closest('#btn-clear-mock')) { st.sel.mock_fit = null; closeDrawer(); renderBoard(); return; }
  if (e.target.closest('#btn-open-sim')) {
    st.simCls = 'all';
    st.simQ = '';
    st.simProspectId = st.sel.id;
    st.simTeam = st.preview;
    st.simYear = st.sel.year;
    st.benchedVets = {};
    st.simRoleOverride = 'auto';
    st.simQbMode = 'actual';
    st.singleRoll = null;
    st.rollHistory = [];
    closeDrawer();
    switchTab('sim');
  }
});

// Tab 2 Search + Scrollable Prospect List Events
$('sim-search').addEventListener('input', e => {
  st.simQ = e.target.value;
  renderSimProspectList(false);
});
$('sim-search').addEventListener('keydown', e => {
  const matches = getFilteredSimProspects();
  if (!matches.length) return;
  const curIdx = matches.findIndex(x => x.id === st.simProspectId);
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    const next = matches[Math.min(matches.length - 1, curIdx + 1)] || matches[0];
    selectSimProspect(next.id, true);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    const prev = matches[Math.max(0, curIdx - 1)] || matches[0];
    selectSimProspect(prev.id, true);
  } else if (e.key === 'Enter') {
    e.preventDefault();
    const pick = curIdx >= 0 ? matches[curIdx] : matches[0];
    selectSimProspect(pick.id, true);
  }
});
$('sim-search-clear').addEventListener('click', () => {
  st.simQ = '';
  $('sim-search').value = '';
  renderSimProspectList(true);
  $('sim-search').focus();
});
$('sim-cls-filter').addEventListener('change', e => {
  st.simCls = e.target.value;
  const matches = getFilteredSimProspects();
  if (matches.length && !matches.some(x => x.id === st.simProspectId)) {
    selectSimProspect(matches[0].id, true);
  } else {
    renderSimProspectList(true);
  }
});
$('sim-prospect-list').addEventListener('click', e => {
  const item = e.target.closest('[data-pid]');
  if (!item) return;
  selectSimProspect(item.dataset.pid, false);
});

$('sim-env-year').addEventListener('change', e => {
  st.simYear = parseInt(e.target.value, 10);
  st.benchedVets = {};
  st.singleRoll = null;
  renderSimTab();
});
$('sim-teams-grid').addEventListener('click', e => {
  const b = e.target.closest('[data-simteam]');
  if (!b) return;
  st.simTeam = b.dataset.simteam;
  st.benchedVets = {};
  st.singleRoll = null;
  renderSimTab();
});
$('sim-role-seg').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  st.simRoleOverride = b.dataset.role;
  document.querySelectorAll('#sim-role-seg button').forEach(btn => btn.classList.toggle('on', btn.dataset.role === st.simRoleOverride));
  st.singleRoll = null;
  renderSimTab();
});
$('sim-btn-best').addEventListener('click', () => {
  const p = P.find(x => x.id === st.simProspectId); if (!p) return;
  st.simTeam = findExtremeTeam(p, st.simYear, true);
  st.benchedVets = {};
  st.singleRoll = null;
  renderSimTab();
});
$('sim-btn-worst').addEventListener('click', () => {
  const p = P.find(x => x.id === st.simProspectId); if (!p) return;
  st.simTeam = findExtremeTeam(p, st.simYear, false);
  st.benchedVets = {};
  st.singleRoll = null;
  renderSimTab();
});
$('sim-btn-drafted').addEventListener('click', () => {
  const p = P.find(x => x.id === st.simProspectId); if (!p) return;
  st.simTeam = p.team || 'KC';
  st.simYear = p.year;
  st.benchedVets = {};
  st.simRoleOverride = 'auto';
  st.simQbMode = 'actual';
  st.singleRoll = null;
  $('sim-env-year').value = String(st.simYear);
  document.querySelectorAll('#sim-role-seg button').forEach(b => b.classList.toggle('on', b.dataset.role === 'auto'));
  renderSimTab();
});
$('sim-btn-pin').addEventListener('click', () => {
  const p = P.find(x => x.id === st.simProspectId); if (!p) return;
  p.mock_fit = mockFit(p, st.simTeam, st.simYear, {
    roleOverride: st.simRoleOverride,
    qbMode: st.simQbMode,
    benchedVets: st.benchedVets
  });
  switchTab('board');
});
$('sim-btn-reset-room').addEventListener('click', () => {
  st.benchedVets = {};
  st.simQbMode = 'actual';
  st.singleRoll = null;
  renderSimTab();
});

// Interactive Roster Moves & Career Roller Inside Main Sim Panel
$('sim-main-panel').addEventListener('click', e => {
  const p = P.find(x => x.id === st.simProspectId) || P[0];
  if (!p) return;

  const vetCard = e.target.closest('[data-vet]');
  if (vetCard) {
    const vName = vetCard.dataset.vet;
    if (st.benchedVets[vName]) delete st.benchedVets[vName];
    else st.benchedVets[vName] = true;
    st.singleRoll = null;
    renderSimTab();
    return;
  }

  const qbBtn = e.target.closest('[data-qb]');
  if (qbBtn) {
    st.simQbMode = qbBtn.dataset.qb;
    st.singleRoll = null;
    renderSimTab();
    return;
  }

  const m = mockFit(p, st.simTeam, st.simYear, {
    roleOverride: st.simRoleOverride,
    qbMode: st.simQbMode,
    benchedVets: st.benchedVets
  });

  if (e.target.closest('#btn-roll-single')) {
    const r = rollSingleCareer(p, m);
    st.singleRoll = r;
    st.rollHistory.unshift(r);
    if (st.rollHistory.length > 8) st.rollHistory.pop();
    renderSimTab();
  } else if (e.target.closest('#btn-roll-5x')) {
    for (let k = 0; k < 5; k++) {
      const r = rollSingleCareer(p, m);
      st.singleRoll = r;
      st.rollHistory.unshift(r);
    }
    st.rollHistory = st.rollHistory.slice(0, 8);
    renderSimTab();
  } else if (e.target.closest('#btn-rerun-mc')) {
    renderSimTab();
  } else if (e.target.closest('[data-histidx]')) {
    const idx = parseInt(e.target.closest('[data-histidx]').dataset.histidx, 10);
    if (st.rollHistory[idx]) {
      st.singleRoll = st.rollHistory[idx];
      renderSimTab();
    }
  }
});

$('lastdone').textContent = META.last_complete;
$('yr-span').textContent = META.years[META.years.length - 1] + '–' + META.years[0];
buildChips();
syncControls();
renderBoard();
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
        f"Generated Boom or Bust 2-tab board with {len(prospects)} prospects across classes "
        f"{years[-1]}–{years[0]} -> {out.resolve()} ({out.stat().st_size / 1024:.0f} KB)"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Boom or Bust NFL WR prospect board & lineup lab HTML app.")
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