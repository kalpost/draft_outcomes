"""
Stage 2: Year-Specific Ex-Ante NFL Team Environment & Fit (`stage2_team_fit.py`)
================================================================================
1. Computes every NFL team's Ex-Ante Environment `(team, draft_year)` directly
   from `nfl_stats_player.parquet` and `combine.parquet` across
   `DRAFT_START_YEAR` to `DRAFT_END_YEAR + 1`.
2. Evaluates the 3-Pillar Team Fit Multiplier (`fit_mult`, `delta_qb_vol`,
   `delta_vacancy`, `delta_role`) with Alpha Target Insulation.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from config import (
    COMBINE_RAW_PATH,
    DRAFT_END_YEAR,
    DRAFT_START_YEAR,
    NFL_STATS_PLAYER_PATH,
    SCORE_UNDRAFTED_CLASS_YEAR,
    norm_nfl_team,
    parse_height_inches,
)


def build_nfl_team_environments() -> dict[tuple[str, int], dict]:
    """
    Computes every NFL team's exact Ex-Ante Environment for each draft year
    directly from `nfl_stats_player.parquet` and `combine.parquet`.
    """
    print("[3/5] Computing year-specific NFL Team Environments from nfl_stats_player.parquet...")

    nfl_stats = pl.read_parquet(NFL_STATS_PLAYER_PATH)
    combine_raw = pl.read_parquet(COMBINE_RAW_PATH)

    vet_size = {}
    for r in combine_raw.iter_rows(named=True):
        nm = str(r.get("player_name") or "").lower().strip()
        ht = parse_height_inches(r.get("ht"))
        wt = float(r["wt"]) if r.get("wt") is not None else None
        if nm and ht and wt:
            vet_size[nm] = (ht, wt)

    team_col = "recent_team" if "recent_team" in nfl_stats.columns else "team"

    reg = nfl_stats.filter(pl.col("season_type") == "REG").with_columns(
        team_norm=pl.col(team_col).map_elements(norm_nfl_team, return_dtype=pl.String),
        season_int=pl.col("season").cast(pl.Int64),
    )

    player_season = (
        reg.group_by(["season_int", "team_norm", "player_id", "player_display_name", "position"])
        .agg(
            attempts=pl.col("attempts").fill_null(0).sum(),
            pass_yds=pl.col("passing_yards").fill_null(0).sum(),
            pass_epa=pl.col("passing_epa").fill_null(0.0).sum(),
            targets=pl.col("targets").fill_null(0).sum(),
            receptions=pl.col("receptions").fill_null(0).sum(),
            rec_yds=pl.col("receiving_yards").fill_null(0.0).sum(),
            rec_tds=pl.col("receiving_tds").fill_null(0.0).sum(),
            air_yds=pl.col("receiving_air_yards").fill_null(0.0).sum(),
            yac_yds=pl.col("receiving_yards_after_catch").fill_null(0.0).sum(),
        )
    )

    team_season_totals = (
        player_season.group_by(["season_int", "team_norm"])
        .agg(
            team_pass_att=pl.col("attempts").sum(),
            team_pass_epa=pl.col("pass_epa").sum(),
            team_targets=pl.col("targets").sum(),
            team_air_yds=pl.col("air_yds").clip(lower_bound=0.0).sum(),
        )
        .with_columns(
            qb_epa_per_att=(pl.col("team_pass_epa") / pl.col("team_pass_att").clip(lower_bound=100))
        )
    )

    season_norms = (
        team_season_totals.group_by("season_int")
        .agg(
            att_mean=pl.col("team_pass_att").mean(),
            att_std=pl.col("team_pass_att").std(),
            epa_mean=pl.col("qb_epa_per_att").mean(),
            epa_std=pl.col("qb_epa_per_att").std(),
        )
    )
    team_season_totals = (
        team_season_totals.join(season_norms, on="season_int", how="left")
        .with_columns(
            pass_vol_z=((pl.col("team_pass_att") - pl.col("att_mean")) / pl.col("att_std").clip(lower_bound=25.0)).clip(-2.2, 2.2),
            qb_epa_z=((pl.col("qb_epa_per_att") - pl.col("epa_mean")) / pl.col("epa_std").clip(lower_bound=0.04)).clip(-2.2, 2.2),
        )
    )

    ts_lookup = {}
    for r in team_season_totals.iter_rows(named=True):
        ts_lookup[(int(r["season_int"]), r["team_norm"])] = r

    active_rosters: dict[tuple[int, str], set[str]] = {}
    players_by_st: dict[tuple[int, str], list[dict]] = {}
    for r in player_season.iter_rows(named=True):
        key = (int(r["season_int"]), r["team_norm"])
        players_by_st.setdefault(key, []).append(r)
        if (r["targets"] or 0) >= 8 or (r["attempts"] or 0) >= 15:
            active_rosters.setdefault(key, set()).add(r["player_id"])

    max_stat_season = int(player_season["season_int"].max())
    env_map = {}
    all_teams = sorted(list(set(team_season_totals["team_norm"].to_list())))
    max_draft_yr = max(DRAFT_END_YEAR + 1, (SCORE_UNDRAFTED_CLASS_YEAR or DRAFT_END_YEAR) + 1)

    for draft_yr in range(DRAFT_START_YEAR, max_draft_yr + 1):
        prev_s = min(draft_yr - 1, max_stat_season)
        curr_s = draft_yr if draft_yr <= max_stat_season else None

        for tm in all_teams:
            ts = ts_lookup.get((prev_s, tm))
            if not ts:
                continue

            tot_tgts = max(float(ts["team_targets"]), 200.0)
            p_list = players_by_st.get((prev_s, tm), [])
            next_roster = active_rosters.get((curr_s, tm)) if curr_s is not None else None

            vacated_tgts = 0.0
            incumbents = []

            wr_te_list = [p for p in p_list if p["position"] in ("WR", "TE") and p["targets"] >= 15]
            wr_te_list.sort(key=lambda x: x["targets"], reverse=True)

            for p in wr_te_list:
                pid = p["player_id"]
                t_share = float(p["targets"]) / tot_tgts

                departed = (next_roster is not None) and (pid not in next_roster)
                if departed:
                    vacated_tgts += float(p["targets"])
                else:
                    if (p["position"] == "WR" and t_share >= 0.11) or (p["position"] == "TE" and t_share >= 0.20):
                        adot = float(p["air_yds"]) / max(float(p["targets"]), 1.0)
                        yac_ratio = float(p["yac_yds"]) / max(float(p["rec_yds"]), 1.0)
                        ypc = float(p["rec_yds"]) / max(float(p["receptions"]), 1.0)
                        p_nm = str(p.get("player_display_name") or "").lower().strip()
                        ht_v, wt_v = vet_size.get(p_nm, (72.5, 200.0))
                        bmi_v = (wt_v * 703.0) / (ht_v ** 2)

                        if p["position"] == "TE":
                            vec = np.array([0.15, 0.15, 0.70])
                        else:
                            vx = (adot - 10.5) * 0.35 + (0.35 - yac_ratio) * 3.5 + (ht_v - 72.0) * 0.35 + (bmi_v - 26.0) * 0.45
                            vz = (ypc - 13.0) * 0.35 + (adot - 10.0) * 0.25 - (bmi_v - 26.0) * 0.35
                            vslot = (10.2 - adot) * 0.45 + (yac_ratio - 0.32) * 4.5 - (ht_v - 71.5) * 0.25
                            sc = np.array([vx, vz, vslot], dtype=float)
                            ex = np.exp((sc - np.max(sc)) / 1.15)
                            vec = ex / np.sum(ex)

                        incumbents.append({
                            "name": p["player_display_name"] or "Veteran WR",
                            "pos": p["position"],
                            "target_share": round(t_share, 3),
                            "adot": round(adot, 1),
                            "vector": np.round(vec, 3),
                        })

            vac_pct = (vacated_tgts / tot_tgts) if next_roster is not None else 0.24
            env_map[(tm, draft_yr)] = {
                "pass_vol_z": round(float(ts["pass_vol_z"]), 3),
                "qb_epa_z": round(float(ts["qb_epa_z"]), 3),
                "vacated_target_pct": round(float(np.clip(vac_pct, 0.02, 0.75)), 3),
                "incumbents": incumbents[:3],
            }

    return env_map


def compute_stage2_team_fit(rookie_arch: dict, team_env: dict, c_grade: float) -> dict:
    """
    Computes the 3-Pillar Ex-Ante Team Fit Multiplier with Alpha Target Insulation.
    """
    r_vec = rookie_arch["vector"]
    pass_vol_z = float(team_env.get("pass_vol_z", 0.0))
    qb_epa_z = float(team_env.get("qb_epa_z", 0.0))
    vacated_pct = float(team_env.get("vacated_target_pct", 0.24))

    raw_qb_vol = (pass_vol_z * 0.055) + (qb_epa_z * 0.075)
    raw_vacancy = (vacated_pct - 0.24) * 0.38

    incumbents = team_env.get("incumbents", [])
    wr_inc_share = sum(inc["target_share"] for inc in incumbents if inc.get("pos") == "WR")

    congestion_pen = 0.0
    synergy_bon = 0.0
    notes = []

    for inc in incumbents:
        v_vec = inc["vector"]
        share = float(inc["target_share"])
        overlap = float(np.dot(r_vec, v_vec))

        if share >= 0.19 and overlap >= 0.46:
            pen = (share - 0.14) * (overlap ** 1.3) * 0.48
            congestion_pen += pen
            notes.append(f"Clash w/ {inc['name']} ({overlap:.0%}, -{pen*100:.1f}%)")
        elif share >= 0.19 and overlap <= 0.28:
            syn = share * (1.0 - overlap) * 0.22
            synergy_bon += syn
            notes.append(f"Synergy w/ {inc['name']} ({overlap:.0%}, +{syn*100:.1f}%)")

    if wr_inc_share <= 0.28:
        runway_bon = 0.065 if c_grade >= 90.0 else 0.035
        synergy_bon += runway_bon
        notes.append("Open WR1 Runway")
    elif vacated_pct >= 0.34 and wr_inc_share <= 0.35:
        synergy_bon += 0.045
        notes.append(f"High Vacancy ({vacated_pct:.0%})")

    raw_role = synergy_bon - congestion_pen

    upside_factor = max(0.75, min(1.35, (c_grade / 80.0) ** 1.25))
    downside_insulation = max(0.45, min(1.0, 1.0 - max(0.0, (c_grade - 86.0) * 0.045)))

    if raw_qb_vol >= 0:
        delta_qb_vol = raw_qb_vol * upside_factor
    else:
        delta_qb_vol = raw_qb_vol * downside_insulation

    delta_vacancy = raw_vacancy * (upside_factor if raw_vacancy >= 0 else downside_insulation)
    delta_role = raw_role * (1.10 if raw_role > 0 else 0.85)

    fit_mult = float(np.clip(1.0 + delta_qb_vol + delta_vacancy + delta_role, 0.82, 1.28))

    return {
        "fit_mult": round(fit_mult, 3),
        "delta_qb_vol": round(delta_qb_vol, 3),
        "delta_vacancy": round(delta_vacancy, 3),
        "delta_role": round(delta_role, 3),
        "notes": "; ".join(notes) if notes else "Balanced Room",
    }