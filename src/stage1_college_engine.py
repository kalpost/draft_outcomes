"""
Stage 1A & 1B: New M2 College Rating Engine & Outcome/Combine Linker (`stage1_college_engine.py`)
=================================================================================================
Key Upgrades:
  1. Eliminating the "Historic Offense Denominator Penalty" (`eff_denom_team_yds` & `eff_denom_team_td`):
     - Previously, in historic high-YPA offenses (e.g. 2019 LSU throwing for 5,671 yds & 60 TDs with
       Joe Burrow; 2013 Texas A&M throwing for 4,593 yds & 39 TDs with Johnny Manziel; 2018–2019
       Oklahoma with Kyler Murray/Jalen Hurts; 2012 Clemson with Tajh Boyd; 2023 LSU with Jayden Daniels),
       dividing a receiver's yards and TDs by raw `team_pass_yds` and `team_rec_td` penalized stars for
       their QB's extreme YPA (`10.8 YPA` at 2019 LSU gave Justin Jefferson's `1,540 yds & 18 TDs` only
       a `28.5%` raw dominator!).
     - Now caps the team yardage denominator at a normal `8.05 YPA` baseline (`denom_team_att * 8.05`)
       and the team TD denominator at a normal `6.6% TD/att` baseline (`denom_team_att * 0.066`).
     - This stops punishing Justin Jefferson, CeeDee Lamb, Mike Evans, DeAndre Hopkins, and Malik Nabers
       for playing in historic high-YPA offenses, while STILL penalizing high-attempt Air Raid offenses
       (like Washington State throwing 53 times/game for Marquess Wilson or West Virginia throwing 44
       times/game for Stedman Bailey) because `denom_team_att` remains high!

  2. Dynamic Multi-Season Trajectory & Air-Raid Attempt Deflation:
     - Deflates high-attempt Air Raid volume (`team_pass_att_per_g > 37.5`, e.g. Washington State &
       West Virginia) and penalizes final-season walk-offs/regressions (Marquess Wilson 2012).
     - Removes the 0.58 redshirt-freshman penalty when a 2-season Power-4 alpha averages `>= 85 yds/g`
       in both seasons (Mike Evans at Texas A&M: `1,105 yds` in 2012 and `1,394 yds` in 2013).
"""

from __future__ import annotations

import math
import numpy as np
import polars as pl

from config import (
    BOOM_YDS_THRESHOLD,
    BUST_YDS_THRESHOLD,
    CFB_END_YEAR,
    CFB_START_YEAR,
    COMBINE_RAW_PATH,
    LEGACY_RATINGS_PATH,
    MIN_PASS_ATT_GAME,
    MIN_UNDRAFTED_SEASON_YDS,
    NFL_DRAFT_PICKS_PATH,
    NFL_STATS_PLAYER_PATH,
    PANEL_V2_FEAT_PATH,
    RECEIVER_GAMES_PATH,
    RECEIVER_SEASONS_PATH,
    SCORE_UNDRAFTED_CLASS_YEAR,
    STARTER_YDS_THRESHOLD,
    norm_nfl_team,
    norm_player_key,
    parse_height_inches,
)


def _ensure_panel_synced() -> None:
    if not RECEIVER_GAMES_PATH.exists():
        return
    rg_seasons = pl.read_parquet(RECEIVER_GAMES_PATH, columns=["season"])
    rg_min, rg_max = int(rg_seasons["season"].min()), int(rg_seasons["season"].max())

    needs_rebuild = not PANEL_V2_FEAT_PATH.exists()
    if not needs_rebuild:
        pf_seasons = pl.read_parquet(PANEL_V2_FEAT_PATH, columns=["season"])
        pf_min, pf_max = int(pf_seasons["season"].min()), int(pf_seasons["season"].max())
        if pf_min > rg_min or pf_max < rg_max:
            print(
                f"      [Sync Check] {PANEL_V2_FEAT_PATH.name} ({pf_min}–{pf_max}) is behind "
                f"{RECEIVER_GAMES_PATH.name} ({rg_min}–{rg_max}). Rebuilding panel_v2_feat..."
            )
            needs_rebuild = True

    if needs_rebuild:
        from build_panel_v2 import build_panel_v2
        build_panel_v2(update_features=True)


def append_undrafted_college_prospects(links_df: pl.DataFrame) -> pl.DataFrame:
    if SCORE_UNDRAFTED_CLASS_YEAR is None:
        return links_df

    rs = pl.read_parquet(RECEIVER_SEASONS_PATH).with_columns(pl.col("athlete_id").cast(pl.String))
    existing_ids = set(links_df["athlete_id"].cast(pl.String).drop_nulls().to_list())

    career_span = (
        rs.filter(pl.col("yds") >= 100)
        .group_by("athlete_id")
        .agg(
            n_logged_seasons=pl.len(),
            first_logged_season=pl.col("season").min(),
            tot_logged_games=pl.col("games").sum(),
        )
    )

    cands = (
        rs.filter(
            (pl.col("season") == CFB_END_YEAR)
            & (pl.col("yds") >= MIN_UNDRAFTED_SEASON_YDS)
            & (pl.col("avg_opp_elo").is_not_null())
            & (pl.col("avg_opp_elo") >= 1380.0)
            & (~pl.col("athlete_id").is_in(existing_ids))
        )
        .join(career_span, on="athlete_id", how="left")
        .sort("yds", descending=True)
        .unique(subset=["athlete_id"], keep="first")
    )
    if cands.height == 0:
        return links_df

    max_wr_id = int(links_df["wr_id"].max() or 0) + 1
    new_rows = []
    for idx, r in enumerate(cands.iter_rows(named=True)):
        n_s = int(r.get("n_logged_seasons") or 1)
        tot_g = int(r.get("tot_logged_games") or r.get("games") or 12)
        new_rows.append({
            "wr_id": max_wr_id + idx,
            "draft_year": int(SCORE_UNDRAFTED_CLASS_YEAR),
            "round": 0,
            "pick": 999,
            "gsis_id": None,
            "pfr_player_id": None,
            "pname": str(r["name"]),
            "pcollege": str(r["team"]),
            "nm": None,
            "sc": None,
            "athlete_id": str(r["athlete_id"]),
            "school_ok": True,
            "n_cands": 1,
            "n_school_ok": 1,
            "manual": False,
            "college_games": tot_g,
            "college_seasons": max(3, n_s + 1) if n_s == 1 else max(3, n_s),
            "thin": tot_g < 12,
        })

    extra_df = pl.DataFrame(new_rows).select(links_df.columns)
    print(
        f"      Appended {extra_df.height} FBS pre-draft college prospects for {SCORE_UNDRAFTED_CLASS_YEAR} "
        f"(season={CFB_END_YEAR}, yds>={MIN_UNDRAFTED_SEASON_YDS})"
    )
    return pl.concat([links_df, extra_df], how="vertical_relaxed")


def build_new_m2_college_ratings(links_df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes Era-Normalized, Bad-QB-Rescued, Historic-Offense-Fair, On-Field Per-Game Dominator,
    and Sustained Power-Alpha M2 Ratings across all college receiver-games.
    """
    _ensure_panel_synced()

    day12_drafted_ids = set(
        links_df.filter((pl.col("round") >= 1) & (pl.col("round") <= 3))["athlete_id"]
        .cast(pl.String)
        .drop_nulls()
        .to_list()
    )
    day3_drafted_ids = set(
        links_df.filter((pl.col("round") >= 4) & (pl.col("round") <= 7))["athlete_id"]
        .cast(pl.String)
        .drop_nulls()
        .to_list()
    )

    links_df = append_undrafted_college_prospects(links_df)

    panel = pl.read_parquet(PANEL_V2_FEAT_PATH).with_columns(
        pl.col("athlete_id").cast(pl.String)
    )
    rec_seasons = pl.read_parquet(RECEIVER_SEASONS_PATH).with_columns(
        pl.col("athlete_id").cast(pl.String)
    )

    p_min_s, p_max_s = int(panel["season"].min()), int(panel["season"].max())
    print(
        f"\n[1/5] Solving Stage 1 New M2 College Rating Engine on {PANEL_V2_FEAT_PATH.name} "
        f"({panel.height:,} games, {p_min_s}–{p_max_s})..."
    )
    if p_min_s > CFB_START_YEAR:
        print(
            f"      [NOTE] {PANEL_V2_FEAT_PATH.name} starts in {p_min_s} (CFB_START_YEAR={CFB_START_YEAR}). "
            f"Run `python3 src/pull_cfbd.py` and `python3 src/parse_boxes.py` to pull {CFB_START_YEAR}–{p_min_s - 1} box scores!"
        )

    if LEGACY_RATINGS_PATH.exists():
        legacy_ratings = pl.read_parquet(LEGACY_RATINGS_PATH).with_columns(
            pl.col("athlete_id").cast(pl.String)
        )
    else:
        legacy_ratings = pl.DataFrame(
            schema={
                "athlete_id": pl.String,
                "final_season_rating": pl.Float64,
                "career_rating": pl.Float64,
                "final_season_games_share": pl.Float64,
            }
        )

    # 1A. Program Schedule Elo Lookup across ALL available college seasons
    team_elo_lookup = (
        rec_seasons.group_by("team")
        .agg(program_opp_elo=pl.col("avg_opp_elo").mean())
    )

    # 1B. Exact Team Season Passing Totals (deduplicated by `game_id`)
    team_unique_games = (
        panel.filter(pl.col("pass_att") > 0)
        .select(["game_id", "season", "team", "pass_att", "pass_yds"])
        .unique(subset=["game_id", "team"])
        .group_by(["season", "team"])
        .agg(
            true_team_games=pl.len(),
            true_team_pass_att=pl.col("pass_att").sum(),
            true_team_pass_yds=pl.col("pass_yds").clip(lower_bound=0).sum(),
        )
    )

    # Teammate NFL WR Room Density weighted by true NFL Draft caliber
    panel_player_season = (
        panel.filter(pl.col("pass_att") > 0)
        .group_by(["season", "team", "athlete_id"])
        .agg(
            ps_games=pl.len(),
            ps_rec=pl.col("rec").clip(lower_bound=0).sum(),
            ps_yds=pl.col("yds").clip(lower_bound=0).sum(),
            ps_td=pl.col("td").clip(lower_bound=0).sum(),
        )
        .join(team_elo_lookup, on="team", how="left")
        .with_columns(
            nfl_wr_weight=(
                pl.when(pl.col("athlete_id").is_in(day12_drafted_ids) & (pl.col("ps_yds") >= 450))
                .then(1.0)
                .when(
                    (pl.col("season") >= 2025)
                    & (pl.col("program_opp_elo").fill_null(1400.0) >= 1590.0)
                    & (pl.col("ps_yds") >= 850)
                )
                .then(0.85)
                .when(pl.col("athlete_id").is_in(day3_drafted_ids) & (pl.col("ps_yds") >= 650))
                .then(0.50)
                .otherwise(0.0)
            ),
        )
        .with_columns(
            weighted_nfl_yds=(pl.col("ps_yds").cast(pl.Float64) * pl.col("nfl_wr_weight")),
            weighted_nfl_rec=(pl.col("ps_rec").cast(pl.Float64) * pl.col("nfl_wr_weight")),
        )
    )

    room_context = (
        panel_player_season.group_by(["season", "team"])
        .agg(
            room_active_nfl_wrs=pl.col("nfl_wr_weight").sum(),
            room_nfl_wr_yds=pl.col("weighted_nfl_yds").sum(),
            room_nfl_wr_rec=pl.col("weighted_nfl_rec").sum(),
            true_team_rec_td=pl.col("ps_td").sum(),
        )
        .join(team_unique_games, on=["season", "team"], how="left")
    )

    player_nfl_wt = panel_player_season.select([
        "season", "team", "athlete_id",
        pl.col("nfl_wr_weight").alias("self_nfl_wt"),
        pl.col("weighted_nfl_yds").alias("self_weighted_yds"),
        pl.col("weighted_nfl_rec").alias("self_weighted_rec"),
    ])

    # 1C. Game-Level Opponent-Adjusted & Pace-Deflated M2 Residuals
    def_q25 = float(panel["def_rating"].quantile(0.25))

    panel_games = (
        panel.filter(pl.col("pass_att") >= MIN_PASS_ATT_GAME)
        .with_columns(
            eff_pass_att=(0.80 * pl.col("pass_att").clip(lower_bound=20).cast(pl.Float64) + 0.20 * 28.5),
            played_game=((pl.col("rec") > 0) | (pl.col("yds") > 0)),
        )
        .with_columns(
            game_exp_wt=pl.when(pl.col("played_game"))
            .then(pl.col("pass_att").cast(pl.Float64))
            .otherwise(pl.col("pass_att").cast(pl.Float64) * 0.04),
            y_pa=(pl.col("yds").clip(lower_bound=0) / pl.col("eff_pass_att")),
            rec_pa=(pl.col("rec").clip(lower_bound=0) / pl.col("eff_pass_att")),
            td_pa=(pl.col("td").clip(lower_bound=0) / pl.col("eff_pass_att")),
            game_yds_share=(
                pl.col("yds").clip(lower_bound=0) / pl.col("pass_yds").clip(lower_bound=125.0)
            ).clip(0.0, 0.75),
            is_tough_def=(pl.col("def_rating") <= def_q25),
            is_fbs=(pl.col("opp_class") == "fbs").fill_null(True),
            # Only deflate extreme Air Raid attempt games (> 39 pass attempts/game, e.g. Washington St / West Virginia / Texas Tech),
            # NOT normal 32-38 att/g pro-spread games (2019 LSU, 2012 Clemson, 2013 Texas A&M)!
            pace_deflator=(39.0 / pl.col("pass_att").clip(lower_bound=39.0)) ** 0.45,
        )
        .with_columns(
            adj_game_yptpa=(
                (
                    pl.col("y_pa") * pl.col("pace_deflator")
                    - pl.col("def_rating").fill_null(0.0) * 0.44
                )
                * pl.when(pl.col("is_fbs")).then(1.0).otherwise(0.68)
            ),
            adj_game_share=(
                pl.col("game_yds_share") * pl.when(pl.col("is_fbs")).then(1.0).otherwise(0.68)
                - pl.col("def_rating").fill_null(0.0) * 0.038
            ),
            consistent_hit=((pl.col("y_pa") - pl.col("def_rating").fill_null(0.0) * 0.30) >= 1.75).cast(pl.Float64),
        )
    )

    season_panel_agg = (
        panel_games.group_by(["athlete_id", "season", "team"])
        .agg(
            s_panel_rows=pl.len(),
            s_active_games=pl.col("played_game").cast(pl.Int32).sum().clip(lower_bound=1),
            s_fbs_share=pl.col("is_fbs").cast(pl.Float64).mean(),
            s_rec=pl.col("rec").clip(lower_bound=0).sum(),
            s_yds=pl.col("yds").clip(lower_bound=0).sum(),
            s_td=pl.col("td").clip(lower_bound=0).sum(),
            s_pass_att=pl.col("game_exp_wt").sum(),
            s_adj_yptpa=((pl.col("adj_game_yptpa") * pl.col("game_exp_wt")).sum() / pl.col("game_exp_wt").sum().clip(lower_bound=1.0)),
            s_adj_share=((pl.col("adj_game_share") * pl.col("game_exp_wt")).sum() / pl.col("game_exp_wt").sum().clip(lower_bound=1.0)),
            s_raw_yptpa=((pl.col("y_pa") * pl.col("game_exp_wt")).sum() / pl.col("game_exp_wt").sum().clip(lower_bound=1.0)),
            s_def_rating=pl.col("def_rating").mean(),
            s_tough_yptpa=pl.col("adj_game_yptpa").filter(pl.col("is_tough_def") & pl.col("played_game")).mean(),
            s_consistency=(
                (pl.col("consistent_hit").filter(pl.col("played_game")).mean()).fill_null(0.50)
            ),
        )
        .join(room_context, on=["season", "team"], how="left")
        .join(player_nfl_wt, on=["season", "team", "athlete_id"], how="left")
        .join(team_elo_lookup, on="team", how="left")
    )

    rs_sub = rec_seasons.select([
        "athlete_id", "season", "team",
        pl.col("games").alias("rs_games"),
        pl.col("rec").alias("rs_rec"),
        pl.col("yds").alias("rs_yds"),
        pl.col("td").alias("rs_td"),
        pl.col("team_att").alias("rs_team_att"),
        pl.col("dominator").alias("rs_dominator"),
        pl.col("yds_share").alias("rs_yds_share"),
        pl.col("td_share").alias("rs_td_share"),
        pl.col("avg_opp_elo").alias("rs_opp_elo"),
        pl.col("team_games").alias("rs_team_games"),
        pl.col("team_yds").alias("rs_team_yds"),
    ])

    season_panel_agg = (
        rs_sub.join(season_panel_agg, on=["athlete_id", "season", "team"], how="full", coalesce=True)
        .join(team_elo_lookup.rename({"program_opp_elo": "prog_elo_fb"}), on="team", how="left")
        .with_columns(
            s_games=pl.coalesce(["rs_games", "s_active_games"]).clip(lower_bound=1),
            s_rec=pl.coalesce(["s_rec", "rs_rec"]).fill_null(0),
            s_yds=pl.coalesce(["s_yds", "rs_yds"]).fill_null(0),
            s_td=pl.coalesce(["s_td", "rs_td"]).fill_null(0),
            s_pass_att=pl.coalesce(["s_pass_att", pl.col("rs_team_att").cast(pl.Float64)]).fill_null(250.0),
            eff_opp_elo=pl.when(pl.col("s_fbs_share").fill_null(1.0) < 0.45)
            .then(pl.coalesce(["rs_opp_elo", "program_opp_elo", "prog_elo_fb"]).fill_null(1240.0).clip(upper_bound=1340.0))
            .otherwise(pl.coalesce(["rs_opp_elo", "program_opp_elo", "prog_elo_fb"]).fill_null(1260.0)),
            raw_denom_team_yds=pl.coalesce(["true_team_pass_yds", "rs_team_yds"]).clip(lower_bound=1750.0).cast(pl.Float64),
            denom_team_att=pl.coalesce(["true_team_pass_att", "rs_team_att"]).clip(lower_bound=240).cast(pl.Float64),
            eff_games_share=(
                pl.coalesce(["rs_games", "s_active_games"]).clip(lower_bound=1).cast(pl.Float64)
                / pl.coalesce(["rs_team_games", "true_team_games"]).clip(lower_bound=6).cast(pl.Float64)
            ).clip(0.15, 1.0),
            team_pass_att_per_g=(
                pl.coalesce(["true_team_pass_att", "rs_team_att"]).cast(pl.Float64)
                / pl.coalesce(["rs_team_games", "true_team_games"]).clip(lower_bound=6).cast(pl.Float64)
            ).fill_null(30.0),
        )
        .with_columns(
            # ELIMINATE THE HISTORIC-OFFENSE DENOMINATOR PENALTY:
            # In historic high-YPA offenses (2019 LSU w/ Joe Burrow `10.8 YPA, 60 TDs`, 2013 Texas A&M
            # w/ Manziel `9.6 YPA`, 2018-19 Oklahoma `10.5 YPA`, 2012 Clemson w/ Boyd, 2023 LSU w/ Daniels),
            # cap the yardage denominator at `denom_team_att * 8.05` (with a 30% slope above 8.05 YPA) and
            # the TD denominator at `denom_team_att * 0.065` (with a 30% slope above).
            # High-attempt Air Raid teams (Washington St `53 att/g`, West Virginia `43 att/g`) still have
            # a huge `denom_team_att`, so they get NO boost, while Justin Jefferson, Mike Evans, DeAndre Hopkins,
            # CeeDee Lamb, and Malik Nabers are no longer penalized for playing with Heisman QBs!
            denom_team_yds=pl.when(pl.col("raw_denom_team_yds") > pl.col("denom_team_att") * 8.05)
            .then(
                pl.col("denom_team_att") * 8.05
                + 0.30 * (pl.col("raw_denom_team_yds") - pl.col("denom_team_att") * 8.05)
            )
            .otherwise(pl.col("raw_denom_team_yds")),
            denom_team_td=pl.when(
                pl.col("true_team_rec_td").fill_null(14).cast(pl.Float64) > pl.col("denom_team_att") * 0.065
            )
            .then(
                pl.col("denom_team_att") * 0.065
                + 0.30 * (pl.col("true_team_rec_td").fill_null(14).cast(pl.Float64) - pl.col("denom_team_att") * 0.065)
            )
            .otherwise(pl.col("true_team_rec_td").fill_null(14).cast(pl.Float64)),
        )
        .with_columns(
            calc_yds_share=(
                pl.col("s_yds")
                / (pl.col("denom_team_yds") * pl.col("eff_games_share").clip(lower_bound=0.55)).clip(lower_bound=1050.0)
            ).clip(0.02, 0.62),
            calc_td_share=(
                pl.col("s_td")
                / (pl.col("denom_team_td") * pl.col("eff_games_share").clip(lower_bound=0.55)).clip(lower_bound=6.0)
            ).clip(0.0, 0.76),
            raw_co_star_nfl_wrs=(pl.col("room_active_nfl_wrs").fill_null(0.0) - pl.col("self_nfl_wt").fill_null(0.0)).clip(0.0, 3.0),
            co_star_nfl_yds=(pl.col("room_nfl_wr_yds").fill_null(0.0) - pl.col("self_weighted_yds").fill_null(0.0)).clip(lower_bound=0.0),
            co_star_nfl_rec=(pl.col("room_nfl_wr_rec").fill_null(0.0) - pl.col("self_weighted_rec").fill_null(0.0)).clip(lower_bound=0.0),
        )
        .with_columns(
            non_star_oth_yds=(pl.col("raw_denom_team_yds") - pl.col("s_yds") - pl.col("co_star_nfl_yds")).clip(lower_bound=150.0),
            non_star_oth_att=(
                pl.col("denom_team_att") - 1.52 * (pl.col("s_rec").cast(pl.Float64) + pl.col("co_star_nfl_rec"))
            ).clip(lower_bound=60.0),
        )
        .with_columns(
            true_non_star_oth_ypa=(
                (pl.col("non_star_oth_yds") + 75.0 * 6.85) / (pl.col("non_star_oth_att") + 75.0)
            ).clip(4.30, 9.80),
            s_raw_yptpa=pl.coalesce(["s_raw_yptpa", pl.col("s_yds") / pl.col("s_pass_att").clip(lower_bound=50.0)]),
            s_adj_yptpa=pl.coalesce(["s_adj_yptpa", (pl.col("s_yds") / pl.col("s_pass_att").clip(lower_bound=50.0)) * 1.02]),
            eff_yds_share=pl.when(pl.col("team_pass_att_per_g") < 17.5)
            .then(pl.col("calc_yds_share"))
            .otherwise(pl.max_horizontal(pl.col("rs_yds_share").fill_null(0.0), pl.col("calc_yds_share")))
            .clip(0.02, 0.62),
            eff_dominator=pl.when(pl.col("team_pass_att_per_g") < 17.5)
            .then(0.62 * pl.col("calc_yds_share") + 0.38 * pl.col("calc_td_share"))
            .otherwise(
                pl.max_horizontal(
                    pl.col("rs_dominator").fill_null(0.0),
                    0.62 * pl.col("calc_yds_share") + 0.38 * pl.col("calc_td_share"),
                )
            ).clip(0.02, 0.65),
            raw_rec_per_game=(pl.col("s_rec") / pl.col("s_games").clip(lower_bound=1)),
            raw_yds_per_game=(pl.col("s_yds") / pl.col("s_games").clip(lower_bound=1)),
            td_per_game=(pl.col("s_td") / pl.col("s_games").clip(lower_bound=1)),
            season_ypc=(pl.col("s_yds") / pl.col("s_rec").clip(lower_bound=1)),
            era_att_baseline=pl.when(pl.col("season") <= 2012).then(16.5).otherwise(20.0),
            # Pass-Rate Normalizer (`pro_style_vol_scale`):
            # Boosts per-game output in <28 att/g run-first offenses (Julio Jones, A.J. Green, Calvin Johnson)
            # AND deflates per-game output in >38 att/g extreme Air Raid offenses (Washington St / Marquess Wilson,
            # West Virginia / Stedman Bailey)!
            pro_style_vol_scale=((32.0 / pl.col("team_pass_att_per_g").clip(20.5, 48.0)) ** 0.45).clip(0.82, 1.22),
        )
        .with_columns(
            rec_per_game=(pl.col("raw_rec_per_game") * pl.col("pro_style_vol_scale")),
            yds_per_game=(pl.col("raw_yds_per_game") * pl.col("pro_style_vol_scale")),
            pass_vol_scale=(
                ((pl.col("team_pass_att_per_g") - pl.col("era_att_baseline")) / 13.0).clip(0.25, 1.0) ** 1.30
            ),
            raw_co_star_yds_share=(
                pl.col("co_star_nfl_yds") / pl.col("raw_denom_team_yds")
            ).clip(0.0, 0.55),
        )
        .with_columns(
            co_star_yds_share=(pl.col("raw_co_star_yds_share") * pl.col("pass_vol_scale")),
            co_star_nfl_wrs=(pl.col("raw_co_star_nfl_wrs") * pl.col("pass_vol_scale")),
        )
        .with_columns(
            pie_adj_dominator=(
                pl.col("eff_dominator") / (1.0 - 0.65 * pl.col("co_star_yds_share")).clip(lower_bound=0.58)
            ).clip(0.04, 0.65),
        )
    )

    first_s_lookup = (
        season_panel_agg.group_by("athlete_id")
        .agg(first_s=pl.col("season").min())
    )

    era_norms = (
        season_panel_agg.filter((pl.col("s_games") >= 6) & (pl.col("s_yds") >= 400))
        .group_by("season")
        .agg(season_med_yptpa=pl.col("s_adj_yptpa").median())
    )
    global_med_yptpa = float(era_norms["season_med_yptpa"].median() or 2.05)
    prior_yptpa = 1.58

    season_panel_agg = (
        season_panel_agg.join(era_norms, on="season", how="left")
        .join(first_s_lookup, on="athlete_id", how="left")
        .with_columns(
            season_career_yr=(pl.col("season") - pl.col("first_s") + 1).clip(1, 6),
            era_yptpa_scale=(
                global_med_yptpa / pl.col("season_med_yptpa").fill_null(global_med_yptpa).clip(1.65, 2.45)
            ).clip(0.92, 1.18),
            sos_bonus_gate=((pl.col("eff_opp_elo") - 1340.0) / 220.0).clip(0.20, 1.15),
            qb_rescue_norm=pl.when(pl.col("true_non_star_oth_ypa") < 6.55)
            .then(((6.60 / pl.col("true_non_star_oth_ypa")) ** 0.52).clip(1.0, 1.24))
            .otherwise(1.0),
            rec_vol_gate=pl.when(pl.col("raw_rec_per_game") < 3.6)
            .then(((pl.col("raw_rec_per_game") - 1.2) / 2.4).clip(0.50, 1.0))
            .otherwise(1.0),
            senior_yr_discount=pl.when(pl.col("season") - pl.col("first_s") >= 3)
            .then(0.90)
            .otherwise(1.0),
            # Air-Raid Extreme Attempt Deflator (`air_raid_deflator`):
            # Deflates offenses throwing >39 times per game (e.g. Washington St 53 att/g for Marquess Wilson,
            # West Virginia 43 att/g for Stedman Bailey, Baylor/Texas Tech) without touching 30-37 att/g teams!
            air_raid_deflator=pl.when(pl.col("team_pass_att_per_g") > 39.0)
            .then(((39.0 / pl.col("team_pass_att_per_g")) ** 0.28).clip(0.86, 1.0))
            .otherwise(1.0),
        )
        .with_columns(
            shrunk_adj_yptpa=(
                (
                    pl.col("s_adj_yptpa")
                    * pl.col("era_yptpa_scale")
                    * pl.col("qb_rescue_norm")
                    * pl.col("rec_vol_gate")
                    * pl.col("senior_yr_discount")
                    * pl.col("air_raid_deflator")
                    * pl.col("s_games").cast(pl.Float64)
                    + prior_yptpa * 2.0
                )
                / (pl.col("s_games").cast(pl.Float64) + 2.0)
            ),
            sos_mult=(
                1.0
                + ((pl.col("eff_opp_elo") - 1530.0) / 600.0).clip(-0.34, 0.24)
                - (pl.col("s_def_rating").fill_null(0.0) * 0.10).clip(-0.10, 0.14)
            ),
            teammate_mult=(
                1.0
                + (pl.col("co_star_yds_share") * 0.54).clip(0.0, 0.27)
                + (pl.col("co_star_nfl_wrs") * 0.048)
            ),
            alpha_dom_mult=(
                1.0
                + ((pl.col("pie_adj_dominator") - 0.31).clip(0.0, 0.30) * 0.52)
                * pl.col("sos_bonus_gate")
                * pl.col("rec_vol_gate")
            ),
            chain_explosive_mult=(
                1.0
                + ((pl.col("yds_per_game") - 68.0) / 290.0).clip(-0.10, 0.16) * pl.col("sos_bonus_gate")
                + pl.when(pl.col("season_ypc") < 11.8)
                .then((pl.col("season_ypc") - 11.8) * 0.050)
                .when((pl.col("season_ypc") > 18.8) & (pl.col("raw_rec_per_game") < 4.8))
                .then(-0.070)
                .otherwise(((pl.col("season_ypc") - 13.5) / 46.0).clip(-0.04, 0.09))
                + ((pl.col("td_per_game") - 0.55) / 8.0).clip(-0.04, 0.09) * pl.col("sos_bonus_gate")
            ),
            s_room_credit=(pl.col("co_star_yds_share") * 1.32 + pl.col("co_star_nfl_wrs") * 0.14).clip(0.0, 0.72),
            s_solo_alpha=(
                (
                    (pl.col("eff_dominator") - 0.27).clip(0.0, 0.35) * 1.45
                    + ((pl.col("yds_per_game") - 72.0).clip(0.0, 50.0) / 100.0) * 0.38
                    + ((6.65 - pl.col("true_non_star_oth_ypa")).clip(0.0, 2.2) * 0.09)
                )
                * ((pl.col("eff_opp_elo") - 1475.0) / 125.0).clip(0.0, 1.25)
                * (1.0 - 1.4 * pl.col("co_star_yds_share")).clip(0.20, 1.0)
                * pl.col("rec_vol_gate")
            ).clip(0.0, 0.64),
            is_power_alpha_season=(
                (pl.col("s_games") >= 6)
                & (pl.col("eff_opp_elo") >= 1510.0)
                & (pl.col("raw_rec_per_game") >= 4.2)
                & ((pl.col("pie_adj_dominator") >= 0.29) | (pl.col("yds_per_game") >= 74.0))
            ).cast(pl.Int32),
        )
        .with_columns(
            s_alpha_or_room_credit=pl.max_horizontal("s_room_credit", "s_solo_alpha").clip(0.0, 0.72),
            s_m2_rating=(
                pl.col("shrunk_adj_yptpa")
                * pl.col("sos_mult")
                * pl.col("teammate_mult")
                * pl.col("alpha_dom_mult")
                * pl.col("chain_explosive_mult")
            ),
            career_season_wt=(
                pl.col("s_games").cast(pl.Float64) * (pl.col("s_rec").cast(pl.Float64) / 24.0).clip(0.25, 1.50)
            ),
        )
    )

    # 1E. Step A: Extract Chronological Career Span & Early Superstar Breakout Flag BEFORE Filtering
    chronological_first = (
        season_panel_agg.sort(["athlete_id", "season"])
        .group_by("athlete_id")
        .agg(
            n_total_seasons=pl.len(),
            n_power_alpha_seasons=pl.col("is_power_alpha_season").sum(),
            chrono_first_season=pl.col("season").first(),
            chrono_last_season=pl.col("season").last(),
            chrono_last_games=pl.col("s_games").last(),
            raw_yr1_dominator=pl.col("pie_adj_dominator").first(),
            yr1_rec=pl.col("s_rec").first(),
            yr1_yds=pl.col("s_yds").first(),
            min_season_yds=pl.col("s_yds").min(),
            max_early_yds=pl.col("s_yds").slice(0, pl.len() - 1).max(),
            max_early_m2=pl.col("s_m2_rating").slice(0, pl.len() - 1).max(),
            career_rec=pl.col("s_rec").sum(),
            career_yds=pl.col("s_yds").sum(),
            career_td=pl.col("s_td").sum(),
            career_pass_att=pl.col("s_pass_att").sum(),
            career_active_games=pl.col("s_games").sum(),
        )
    )

    # 1E. Step B: Filter Out Low-Snap Bench Cameos while Preserving High-Output Short Seasons
    base_s = season_panel_agg.filter(pl.col("s_games") >= 3).sort(["athlete_id", "season"])
    player_s_meta = (
        base_s.group_by("athlete_id")
        .agg(
            max_s=pl.col("season").max(),
            n_starter_s=(
                (pl.col("s_games") >= 5)
                & ((pl.col("s_rec") >= 18) | (pl.col("eff_yds_share") >= 0.15) | (pl.col("yds_per_game") >= 65.0))
            ).cast(pl.Int32).sum(),
            n_healthy_s=(pl.col("s_games") >= 6).cast(pl.Int32).sum(),
        )
    )

    valid_seasons = (
        base_s.join(player_s_meta, on="athlete_id", how="left")
        .filter(
            pl.when(pl.col("n_starter_s") >= 1)
            .then(
                (pl.col("s_games") >= 5)
                & ((pl.col("s_rec") >= 18) | (pl.col("eff_yds_share") >= 0.15) | (pl.col("yds_per_game") >= 65.0))
            )
            .otherwise(
                ~(
                    (pl.col("season") == pl.col("max_s"))
                    & (pl.col("s_games") <= 5)
                    & (pl.col("n_healthy_s") >= 1)
                )
            )
        )
        .sort(["athlete_id", "season"])
    )

    career_agg = (
        valid_seasons.group_by("athlete_id")
        .agg(
            active_starter_seasons=pl.len(),
            last_season_panel=pl.col("season").last(),
            m2_peak=pl.col("s_m2_rating").max(),
            m2_top2_mean=pl.col("s_m2_rating").sort(descending=True).head(2).mean(),
            m2_final=pl.col("s_m2_rating").last(),
            m2_penultimate=pl.col("s_m2_rating").slice(-2, 1).first(),
            m2_wt_career=((pl.col("s_m2_rating") * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
            peak_dominator=pl.col("pie_adj_dominator").max(),
            raw_peak_dominator=pl.col("eff_dominator").max(),
            final_dominator=pl.col("pie_adj_dominator").last(),
            peak_rec_per_g=pl.col("rec_per_game").max(),
            peak_yds_per_g=pl.col("yds_per_game").max(),
            final_rec_per_g=pl.col("rec_per_game").last(),
            final_yds_per_g=pl.col("yds_per_game").last(),
            final_ypc=pl.col("season_ypc").last(),
            final_td=pl.col("s_td").last(),
            final_games_played=pl.col("s_games").last(),
            final_games_share=pl.col("eff_games_share").last(),
            max_co_star_wrs=pl.col("co_star_nfl_wrs").max(),
            max_co_star_yds=pl.col("co_star_nfl_yds").max(),
            max_co_star_share=pl.col("co_star_yds_share").max(),
            max_raw_co_star_share=pl.col("raw_co_star_yds_share").max(),
            career_alpha_room_credit=(
                0.65 * pl.col("s_alpha_or_room_credit").max()
                + 0.35 * ((pl.col("s_alpha_or_room_credit") * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0))
            ),
            mean_opp_elo=((pl.col("eff_opp_elo") * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
            mean_def_rating=((pl.col("s_def_rating").fill_null(0.0) * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
            mean_oth_ypa=((pl.col("true_non_star_oth_ypa").fill_null(6.85) * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
            mean_tough_yptpa=pl.col("s_tough_yptpa").mean(),
            mean_adj_yptpa=((pl.col("s_adj_yptpa") * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
            mean_consistency=((pl.col("s_consistency").fill_null(0.50) * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
        )
        .with_columns(
            m2_career=pl.when(pl.col("active_starter_seasons") >= 2)
            .then(0.58 * pl.col("m2_top2_mean") + 0.42 * pl.col("m2_wt_career"))
            .otherwise(pl.col("m2_wt_career")),
        )
        .join(chronological_first, on="athlete_id", how="left")
    )

    df = (
        links_df.with_columns(pl.col("athlete_id").cast(pl.String))
        .join(career_agg, on="athlete_id", how="left")
        .join(
            legacy_ratings.select([
                "athlete_id",
                pl.col("final_season_rating").alias("leg_final"),
                pl.col("career_rating").alias("leg_career"),
                pl.col("final_season_games_share").alias("leg_g_share"),
            ]),
            on="athlete_id",
            how="left",
        )
    )

    df = df.with_columns(
        m2_peak=pl.coalesce(["m2_peak", pl.col("leg_final") * 1.35 + 1.85]).fill_null(1.75),
        m2_final=pl.coalesce(["m2_final", pl.col("leg_final") * 1.35 + 1.80]).fill_null(1.70),
        m2_career=pl.coalesce(["m2_career", pl.col("leg_career") * 1.35 + 1.75]).fill_null(1.68),
        peak_dominator=pl.col("peak_dominator").fill_null(0.24),
        raw_peak_dominator=pl.col("raw_peak_dominator").fill_null(pl.col("peak_dominator")).fill_null(0.22),
        final_dominator=pl.col("final_dominator").fill_null(0.22),
        n_total_seasons=pl.col("n_total_seasons").fill_null(1),
        n_power_alpha_seasons=pl.col("n_power_alpha_seasons").fill_null(0),
        active_starter_seasons=pl.col("active_starter_seasons").fill_null(1),
        college_seasons=pl.when(
            (pl.col("draft_year").cast(pl.Int64) - pl.col("chrono_first_season").cast(pl.Int64)) <= 1
        )
        .then(pl.lit(4))
        .otherwise(
            (pl.col("draft_year").cast(pl.Int64) - pl.col("chrono_first_season").cast(pl.Int64)).clip(3, 5)
        )
        .fill_null(4),
        peak_rec_per_g=pl.col("peak_rec_per_g").fill_null(4.4),
        peak_yds_per_g=pl.col("peak_yds_per_g").fill_null(62.0),
        final_rec_per_g=pl.col("final_rec_per_g").fill_null(4.2),
        final_yds_per_g=pl.col("final_yds_per_g").fill_null(60.0),
        final_td=pl.col("final_td").fill_null(5),
        final_games_played=pl.col("final_games_played").fill_null(11),
        final_games_share=pl.coalesce(["final_games_share", "leg_g_share"]).fill_null(0.85),
        max_co_star_wrs=pl.col("max_co_star_wrs").fill_null(0.0),
        max_co_star_yds=pl.col("max_co_star_yds").fill_null(0.0),
        max_co_star_share=pl.col("max_co_star_share").fill_null(0.0),
        max_raw_co_star_share=pl.col("max_raw_co_star_share").fill_null(0.0),
        career_alpha_room_credit=pl.col("career_alpha_room_credit").fill_null(0.0),
        career_rec=pl.col("career_rec").fill_null(85),
        career_yds=pl.col("career_yds").fill_null(1150),
        career_td=pl.col("career_td").fill_null(9),
        career_pass_att=pl.col("career_pass_att").fill_null(750),
        career_active_games=pl.col("career_active_games").fill_null(20),
        mean_opp_elo=pl.col("mean_opp_elo").fill_null(1380.0),
        mean_def_rating=pl.col("mean_def_rating").fill_null(0.0),
        mean_consistency=pl.col("mean_consistency").fill_null(0.48),
    )

    df = df.with_columns(
        is_verified_freshman_season=(
            (
                (pl.col("chrono_first_season") <= (pl.col("draft_year") - 3))
                & (pl.col("n_total_seasons") >= 3)
                & (pl.col("yr1_yds").fill_null(0) >= 320)
            )
            | (
                (pl.col("chrono_first_season") == (pl.col("draft_year") - 2))
                & (pl.col("n_total_seasons") >= 2)
                & (pl.col("yr1_yds").fill_null(0) >= 480)
            )
            | (
                (pl.col("draft_year") == (CFB_END_YEAR + 1))
                & (pl.col("mean_opp_elo") >= 1560.0)
                & (pl.col("yr1_yds").fill_null(0) >= 750)
            )
        ),
        is_covid_optout_2021=(
            (pl.col("draft_year") == 2021) & (pl.col("last_season_panel") <= 2019)
        ),
        is_early_breakout_injured_star=(
            (pl.col("last_season_panel") < (pl.col("draft_year") - 1))
            & (pl.col("mean_opp_elo") >= 1545.0)
            & (pl.col("max_co_star_wrs") >= 0.85)
            & ((pl.col("max_early_yds").fill_null(0) >= 1100) | (pl.col("m2_peak") >= 2.75))
        ),
    ).with_columns(
        had_short_final_year=(
            (~pl.col("is_covid_optout_2021"))
            & (~pl.col("is_early_breakout_injured_star"))
            & (
                (pl.col("chrono_last_games").fill_null(12) <= 4)
                | (pl.col("last_season_panel") < (pl.col("draft_year") - 1))
            )
        ),
        # If a 2-season redshirt freshman posted >= 1,050 yards in BOTH seasons against an SEC/B1G
        # schedule (e.g. Mike Evans at Texas A&M: 1,105 yds in 2012 & 1,394 yds in 2013), give full 1.0x
        # freshman weight instead of the 0.58x discount meant for declining 2-year players (Sidney Rice)!
        yr1_dominator=pl.when(pl.col("is_verified_freshman_season"))
        .then(
            pl.col("raw_yr1_dominator").fill_null(0.15)
            * pl.when(
                (pl.col("n_total_seasons") >= 3)
                | ((pl.col("min_season_yds").fill_null(0) >= 1050) & (pl.col("mean_opp_elo") >= 1550.0))
            )
            .then(1.0)
            .otherwise(0.58)
        )
        .when(pl.col("is_early_breakout_injured_star"))
        .then(0.32)
        .otherwise(0.14),
    ).with_columns(
        standout_freshman_bonus=(
            (pl.col("yr1_dominator") - 0.20).clip(lower_bound=0.0, upper_bound=0.26)
            * ((pl.col("mean_opp_elo") - 1380.0) / 200.0).clip(0.30, 1.10)
        ),
        is_optout_or_injured_final=(
            pl.col("is_covid_optout_2021")
            | pl.col("is_early_breakout_injured_star")
            | (
                (pl.col("final_games_played") <= 5)
                & (pl.col("m2_penultimate").fill_null(0.0) >= 2.25)
            )
            | (
                (pl.col("final_games_share") <= 0.78)
                & (pl.col("final_yds_per_g") >= 82.0)
                # Require `m2_final >= m2_peak * 0.90` so players who regressed and quit mid-season
                # (like Marquess Wilson in 2012 at Washington State) don't get injury protection!
                & (pl.col("m2_final") >= pl.col("m2_peak") * 0.90)
            )
            | (
                (pl.col("final_yds_per_g") >= 96.0)
                & (pl.col("raw_peak_dominator") >= 0.38)
            )
        ),
        one_year_wonder_pen=pl.when(
            (pl.col("active_starter_seasons") <= 1)
            & (~pl.col("is_covid_optout_2021"))
            & (~pl.col("is_early_breakout_injured_star"))
        )
        .then(
            pl.when(pl.col("n_total_seasons") <= 1).then(0.32).otherwise(0.24)
        )
        .otherwise(0.0),
        sustained_alpha_bonus=(
            pl.when(pl.col("n_power_alpha_seasons") >= 3)
            .then(0.25)
            .when(
                ((pl.col("n_power_alpha_seasons") == 2) & (pl.col("mean_opp_elo") >= 1510.0))
                | pl.col("is_early_breakout_injured_star")
                | (pl.col("is_covid_optout_2021") & (pl.col("m2_peak") >= 2.70))
            )
            .then(0.17)
            .otherwise(0.0)
        ),
    ).with_columns(
        ascension_bonus=pl.when(
            (~pl.col("had_short_final_year"))
            & (pl.col("m2_final") >= pl.col("m2_career") * 0.96)
            & (pl.col("final_yds_per_g") >= 82.0)
            & ((pl.col("active_starter_seasons") >= 2) | pl.col("is_early_breakout_injured_star"))
        )
        .then(
            (
                (pl.col("m2_final") - pl.col("m2_career")).clip(0.0, 0.45) * 0.30
                + ((pl.col("final_yds_per_g") - 80.0).clip(0.0, 50.0) / 100.0) * 0.28
                + ((pl.col("final_td").cast(pl.Float64) - 10.0).clip(0.0, 10.0) / 10.0) * 0.08
            ).clip(0.0, 0.26)
        )
        .otherwise(0.0),
        final_regression_pen=pl.when(
            pl.col("had_short_final_year")
        )
        .then(0.22)
        .when(
            (~pl.col("is_optout_or_injured_final")) & (pl.col("m2_final") < pl.col("m2_peak") * 0.93)
        )
        .then((pl.col("m2_peak") - pl.col("m2_final")) * 0.82)
        .when(
            (pl.col("n_total_seasons") >= 3)
            & (~pl.col("is_covid_optout_2021"))
            & (~pl.col("is_early_breakout_injured_star"))
            & (pl.col("final_yds_per_g") < 70.0)
            & (pl.col("m2_final") < pl.col("m2_peak"))
        )
        .then((pl.col("m2_peak") - pl.col("m2_final")) * 0.85 + 0.14)
        .otherwise(0.0),
    ).with_columns(
        m2_final_eff=(
            pl.when(pl.col("is_optout_or_injured_final"))
            .then(0.85 * pl.col("m2_peak") + 0.15 * pl.col("m2_final"))
            .otherwise(0.76 * pl.col("m2_final") + 0.24 * pl.col("m2_peak"))
            + pl.col("ascension_bonus")
            + pl.col("sustained_alpha_bonus") * 0.75
            - pl.col("one_year_wonder_pen")
        ),
        m2_career=pl.when(
            (pl.col("active_starter_seasons") <= 1)
            & (~pl.col("is_covid_optout_2021"))
            & (~pl.col("is_early_breakout_injured_star"))
        )
        .then(0.68 * pl.col("m2_career") + 0.32 * 1.68)
        .otherwise(pl.col("m2_career") + pl.col("sustained_alpha_bonus") * 0.85),
    )

    df = df.with_columns(
        college_ypc=(pl.col("career_yds") / pl.col("career_rec").clip(lower_bound=1)).clip(8.5, 23.5),
        college_rec_per_att=(pl.col("career_rec") / pl.col("career_pass_att").clip(lower_bound=50)).clip(0.04, 0.28),
        college_td_per_rec=(pl.col("career_td") / pl.col("career_rec").clip(lower_bound=1)).clip(0.0, 0.35),
        vs_top_def_ratio=(
            pl.col("mean_tough_yptpa").fill_null(pl.col("mean_adj_yptpa"))
            / pl.col("mean_adj_yptpa").clip(lower_bound=0.6)
        ).fill_null(1.0).clip(0.70, 1.42),
        qb_independence_ratio=(
            pl.col("mean_adj_yptpa").fill_null(1.8)
            / (pl.col("mean_oth_ypa").fill_null(6.85).clip(lower_bound=4.3) * 0.26)
        ).clip(0.55, 1.95),
        opp_defense_difficulty=(
            ((pl.col("mean_opp_elo") - 1525.0) / 160.0)
            - pl.col("mean_def_rating") * 0.45
        ).clip(-1.15, 0.98),
        teammate_wr_boost=(
            pl.col("career_alpha_room_credit") + pl.col("sustained_alpha_bonus") * 0.55
        ).clip(0.0, 0.78),
    )

    print(
        f"      Solved New M2 Ratings for all {len(df)} prospects across "
        f"{df['draft_year'].min()}–{df['draft_year'].max()}"
    )
    return df


def attach_nfl_outcomes_and_combine(df: pl.DataFrame) -> pl.DataFrame:
    """
    1. Attaches ground-truth NFL Year 1-3 receiving outcomes from `nfl_stats_player.parquet`
       and draft age / team from `nfl_draft_picks.parquet`.
    2. Links Combine/Pro-Day measurables from `combine.parquet` and computes both
       `pure_speed_score` and `hass`.
    """
    print("[2/5] Linking ground-truth NFL outcomes, draft age, and Combine measurables...")

    draft_picks = pl.read_parquet(NFL_DRAFT_PICKS_PATH)
    nfl_stats = pl.read_parquet(NFL_STATS_PLAYER_PATH)
    combine_raw = pl.read_parquet(COMBINE_RAW_PATH)

    dp_sub = (
        draft_picks.select([
            pl.col("season").cast(pl.Int64).alias("draft_year"),
            pl.col("pick").cast(pl.Int64),
            pl.col("team").alias("nfl_draft_team"),
            pl.col("age").cast(pl.Float64).alias("draft_age"),
        ])
        .unique(subset=["draft_year", "pick"])
    )
    df = df.join(dp_sub, on=["draft_year", "pick"], how="left")
    df = df.with_columns(
        nfl_team=pl.col("nfl_draft_team").map_elements(norm_nfl_team, return_dtype=pl.String)
    )

    reg_stats = nfl_stats.filter(pl.col("season_type") == "REG").select([
        pl.col("player_id").alias("gsis_id"),
        pl.col("season").cast(pl.Int64).alias("nfl_season"),
        pl.col("receiving_yards").fill_null(0.0),
        pl.col("receptions").fill_null(0.0),
        pl.col("targets").fill_null(0.0),
    ])

    y1_3_outcomes = (
        df.select(["wr_id", "gsis_id", "draft_year"])
        .join(reg_stats, on="gsis_id", how="inner")
        .filter(
            (pl.col("nfl_season") >= pl.col("draft_year"))
            & (pl.col("nfl_season") <= pl.col("draft_year") + 2)
        )
        .group_by("wr_id")
        .agg(
            stats_y1_3_yds=pl.col("receiving_yards").sum(),
            nfl_rec_y1_3=pl.col("receptions").sum(),
            nfl_tgt_y1_3=pl.col("targets").sum(),
            nfl_seasons_logged=pl.col("nfl_season").n_unique(),
        )
    )
    df = df.join(y1_3_outcomes, on="wr_id", how="left").with_columns(
        nfl_actual_y1_3=pl.col("stats_y1_3_yds").fill_null(0.0),
        nfl_rec_y1_3=pl.col("nfl_rec_y1_3").fill_null(0.0),
        nfl_tgt_y1_3=pl.col("nfl_tgt_y1_3").fill_null(0.0),
    )

    c_records = []
    for r in combine_raw.iter_rows(named=True):
        yr = r.get("draft_year") or r.get("season")
        if yr is None:
            continue
        yr_int = int(float(yr))
        ht_in = parse_height_inches(r.get("ht"))
        wt = float(r["wt"]) if r.get("wt") is not None else None
        forty = float(r["forty"]) if r.get("forty") is not None else None
        vert = float(r["vertical"]) if r.get("vertical") is not None else None
        broad = float(r["broad_jump"]) if r.get("broad_jump") is not None else None
        cone = float(r["cone"]) if r.get("cone") is not None else None
        shuttle = float(r["shuttle"]) if r.get("shuttle") is not None else None

        pfr_id = r.get("pfr_id")
        pname_clean = norm_player_key(str(r.get("player_name") or ""))

        c_records.append({
            "pfr_player_id": str(pfr_id) if pfr_id else None,
            "name_yr_key": f"{pname_clean}_{yr_int}",
            "ht_in": ht_in,
            "wt_lbs": wt,
            "forty_sec": forty,
            "vertical_in": vert,
            "broad_in": broad,
            "cone_sec": cone,
            "shuttle_sec": shuttle,
        })

    c_df = pl.DataFrame(c_records)
    c_by_pfr = c_df.filter(pl.col("pfr_player_id").is_not_null()).unique(subset=["pfr_player_id"]).drop("name_yr_key")
    c_by_name = c_df.unique(subset=["name_yr_key"]).drop("pfr_player_id").rename({
        "ht_in": "ht_in_nm", "wt_lbs": "wt_lbs_nm", "forty_sec": "forty_sec_nm",
        "vertical_in": "vertical_in_nm", "broad_in": "broad_in_nm",
        "cone_sec": "cone_sec_nm", "shuttle_sec": "shuttle_sec_nm",
    })

    df = df.with_columns(
        name_yr_key=(
            pl.col("pname").map_elements(norm_player_key, return_dtype=pl.String)
            + "_"
            + pl.col("draft_year").cast(pl.String)
        )
    )
    df = df.join(c_by_pfr, on="pfr_player_id", how="left").join(c_by_name, on="name_yr_key", how="left")

    df = df.with_columns(
        ht_in=pl.coalesce(["ht_in", "ht_in_nm"]),
        wt_lbs=pl.coalesce(["wt_lbs", "wt_lbs_nm"]),
        forty_sec=pl.coalesce(["forty_sec", "forty_sec_nm"]),
        vertical_in=pl.coalesce(["vertical_in", "vertical_in_nm"]),
        broad_in=pl.coalesce(["broad_in", "broad_in_nm"]),
        cone_sec=pl.coalesce(["cone_sec", "cone_sec_nm"]),
        shuttle_sec=pl.coalesce(["shuttle_sec", "shuttle_sec_nm"]),
    ).drop([
        "ht_in_nm", "wt_lbs_nm", "forty_sec_nm", "vertical_in_nm",
        "broad_in_nm", "cone_sec_nm", "shuttle_sec_nm", "name_yr_key",
    ])

    df = df.with_columns(
        bmi=pl.when(pl.col("wt_lbs").is_not_null() & pl.col("ht_in").is_not_null())
        .then(((pl.col("wt_lbs") * 703.0) / (pl.col("ht_in") ** 2)).round(1))
        .otherwise(None),
        has_testing=pl.col("forty_sec").is_not_null().cast(pl.Int32),
        hass=pl.when(
            pl.col("forty_sec").is_not_null()
            & pl.col("wt_lbs").is_not_null()
            & pl.col("ht_in").is_not_null()
            & (pl.col("forty_sec") >= 4.15)
            & (pl.col("forty_sec") <= 5.0)
        )
        .then((((pl.col("wt_lbs") * 200.0) / (pl.col("forty_sec") ** 4)) * (pl.col("ht_in") / 73.0)).round(1))
        .otherwise(None),
        pure_speed_score=pl.when(
            pl.col("forty_sec").is_not_null()
            & (pl.col("forty_sec") >= 4.15)
            & (pl.col("forty_sec") <= 5.0)
        )
        .then((100.0 + (4.48 - pl.col("forty_sec")) * 115.0 + (4.38 - pl.col("forty_sec")).clip(lower_bound=0.0) * 65.0).round(1))
        .otherwise(None),
        burst_score=pl.when(pl.col("vertical_in").is_not_null() & pl.col("broad_in").is_not_null())
        .then((((pl.col("vertical_in") + pl.col("broad_in") / 10.0) / 47.6) * 100.0).round(1))
        .when(pl.col("vertical_in").is_not_null())
        .then(((pl.col("vertical_in") / 35.5) * 100.0).round(1))
        .otherwise(None),
        agility_sum=pl.when(
            pl.col("cone_sec").is_not_null() & pl.col("shuttle_sec").is_not_null()
            & (pl.col("cone_sec") >= 6.2) & (pl.col("cone_sec") <= 8.0)
            & (pl.col("shuttle_sec") >= 3.8) & (pl.col("shuttle_sec") <= 4.9)
        )
        .then((pl.col("cone_sec") + pl.col("shuttle_sec")).round(2))
        .otherwise(None),
    )

    return df


def build_stage1_skill_vector(r: dict) -> list[float]:
    """
    Constructs the 7 orthogonal, non-negative-signed Stage 1 College Skill signals.
    """
    m2_fin = float(r["m2_final_eff"]) - float(r.get("final_regression_pen") or 0.0)
    m2_car = float(r["m2_career"])

    c_seasons = int(r.get("college_seasons") or 4)
    n_starter = int(r.get("active_starter_seasons") or 1)
    n_total = int(r.get("n_total_seasons") or 1)
    is_early_star = bool(r.get("is_early_breakout_injured_star") or r.get("is_covid_optout_2021"))
    d_age = (
        float(r["draft_age"])
        if r.get("draft_age") is not None
        else (21.2 if (c_seasons <= 3 and (n_total >= 3 or is_early_star)) else (21.9 if c_seasons <= 3 else 22.6))
    )
    eff_c_seasons = max(c_seasons, 4) if d_age >= 22.45 else c_seasons
    fresh_bon = float(r.get("standout_freshman_bonus") or 0.0)
    sust_alpha = float(r.get("sustained_alpha_bonus") or 0.0)

    sample_mult = 0.40 if (n_starter <= 1 and not is_early_star) else 1.0
    age_score = (
        ((22.1 - d_age) * 0.35 + (3.6 - min(5, max(3, eff_c_seasons))) * 0.40) * sample_mult
        + fresh_bon * 1.15
        + sust_alpha * 0.80
    )

    opp_elo = float(r.get("mean_opp_elo") or 1450.0)
    def_rat = float(r.get("mean_def_rating") or 0.0)
    sos_score = ((opp_elo - 1530.0) / 120.0) - (def_rat * 0.45)

    tm_boost = float(r.get("teammate_wr_boost") or 0.0)

    pk_dom = float(r.get("peak_dominator") or 0.24)
    pk_yds_g = float(r.get("peak_yds_per_g") or 65.0)
    pk_rec_g = float(r.get("peak_rec_per_g") or 4.5)
    c_ypc = float(r.get("college_ypc") or 13.8)
    qb_ind = float(r.get("qb_independence_ratio") or 1.0)
    sos_weight = float(np.clip((opp_elo - 1340.0) / 260.0, 0.25, 1.22))
    ypc_route_factor = (
        0.70 if c_ypc < 11.5
        else (0.85 if c_ypc < 12.5 else (0.80 if (c_ypc > 18.8 and pk_rec_g < 4.8) else min(1.12, 0.95 + (c_ypc - 12.5) * 0.028)))
    )
    yds_g_factor = float(np.clip(pk_yds_g / 86.0, 0.65, 1.28))
    rec_g_factor = float(np.clip(pk_rec_g / 5.2, 0.60, 1.12))
    sos_adj_dom = pk_dom * sos_weight * ypc_route_factor * yds_g_factor * rec_g_factor * (0.82 + 0.18 * min(1.7, qb_ind))

    top_def = float(r.get("vs_top_def_ratio") or 1.0)
    g_share = float(r.get("final_games_share") or 0.85)
    eff_g_share = max(0.88, g_share) if (float(r.get("final_yds_per_g") or 60.0) >= 84.0 or is_early_star) else g_share
    cons_rate = float(r.get("mean_consistency") or 0.50)
    resilience = (
        (top_def - 1.0) * 0.55
        + (eff_g_share - 0.85) * 0.40
        + (cons_rate - 0.52) * 0.75
        + sust_alpha * 0.85
    )

    return [m2_fin, m2_car, age_score, sos_score, tm_boost, sos_adj_dom, resilience]


def derive_consistent_probabilities(p10: float, p50: float, p90: float) -> tuple[float, float, float]:
    """
    Derives P(Starter >= 1500), P(Boom >= 2500), and P(Bust < 500) directly from
    the prospect's continuous (p10, p50, p90) quantile distribution on the sqrt scale.
    """
    s10 = math.sqrt(max(0.0, float(p10)))
    s50 = math.sqrt(max(25.0, float(p50)))
    s90 = math.sqrt(max(100.0, float(p90)))

    sigma_left = max(3.0, (s50 - s10) / 1.38)
    sigma_right = max(3.8, (s90 - s50) / 1.38)

    def cdf_sqrt(y_thresh: float) -> float:
        sy = math.sqrt(max(0.0, y_thresh))
        sigma = sigma_left if sy < s50 else sigma_right
        z = (sy - s50) / sigma
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    p_bust = float(np.clip(cdf_sqrt(BUST_YDS_THRESHOLD), 0.03, 0.94))
    p_starter = float(np.clip(1.0 - cdf_sqrt(STARTER_YDS_THRESHOLD), 0.03, 0.94))
    p_boom = float(np.clip(1.0 - cdf_sqrt(BOOM_YDS_THRESHOLD), 0.01, min(0.88, p_starter - 0.04)))

    return round(p_starter, 3), round(p_boom, 3), round(p_bust, 3)