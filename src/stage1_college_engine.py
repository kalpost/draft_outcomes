"""
Stage 1A & 1B: New M2 College Rating Engine & Outcome/Combine Linker (`stage1_college_engine.py`)
=================================================================================================
1. Solves Stage 1A New M2 College Ratings across all games in `panel_v2_feat.parquet`
   (Opponent-Defense & Pace-Adjusted Game Efficiency, Active-Game Exposure Weighting,
   Pass-Volume-Scaled Teammate Cannibalization, Verified Solo-Alpha Funnel Credit,
   and One-Sided Standout Freshman Bonus).
2. Optionally appends pre-draft / undrafted college WRs for `SCORE_UNDRAFTED_CLASS_YEAR`
   when configured in `config.py`.
3. Links NFL Year 1–3 ground-truth outcomes and Combine/Pro-Day measurables.
4. Constructs the 7 orthogonal Stage 1 skill features and derives continuous probabilities.
"""

from __future__ import annotations

import math
import numpy as np
import polars as pl

from config import (
    BOOM_YDS_THRESHOLD,
    BUST_YDS_THRESHOLD,
    CFB_END_YEAR,
    COMBINE_RAW_PATH,
    LEGACY_RATINGS_PATH,
    MIN_PASS_ATT_GAME,
    MIN_UNDRAFTED_SEASON_YDS,
    NFL_DRAFT_PICKS_PATH,
    NFL_STATS_PLAYER_PATH,
    PANEL_V2_FEAT_PATH,
    RECEIVER_SEASONS_PATH,
    SCORE_UNDRAFTED_CLASS_YEAR,
    STARTER_YDS_THRESHOLD,
    norm_nfl_team,
    norm_player_key,
    parse_height_inches,
)


def append_undrafted_college_prospects(links_df: pl.DataFrame) -> pl.DataFrame:
    """
    If `SCORE_UNDRAFTED_CLASS_YEAR` is set in `config.py` (e.g. 2026), appends all
    college receivers from `CFB_END_YEAR` with `yds >= MIN_UNDRAFTED_SEASON_YDS`
    who are not already in `links_df`, allowing the frozen model to score an entire
    upcoming draft class prior to the NFL Draft.
    """
    if SCORE_UNDRAFTED_CLASS_YEAR is None:
        return links_df

    rs = pl.read_parquet(RECEIVER_SEASONS_PATH).with_columns(pl.col("athlete_id").cast(pl.String))
    existing_ids = set(links_df["athlete_id"].cast(pl.String).drop_nulls().to_list())

    cands = (
        rs.filter(
            (pl.col("season") == CFB_END_YEAR)
            & (pl.col("yds") >= MIN_UNDRAFTED_SEASON_YDS)
            & (~pl.col("athlete_id").is_in(existing_ids))
        )
        .sort("yds", descending=True)
        .unique(subset=["athlete_id"], keep="first")
    )
    if cands.height == 0:
        return links_df

    max_wr_id = int(links_df["wr_id"].max() or 0) + 1
    new_rows = []
    for idx, r in enumerate(cands.iter_rows(named=True)):
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
            "college_games": int(r["games"] or 12),
            "college_seasons": 3,
            "thin": False,
        })

    extra_df = pl.DataFrame(new_rows).select(links_df.columns)
    print(
        f"      Appended {extra_df.height} pre-draft college prospects for {SCORE_UNDRAFTED_CLASS_YEAR} "
        f"(season={CFB_END_YEAR}, yds>={MIN_UNDRAFTED_SEASON_YDS})"
    )
    return pl.concat([links_df, extra_df], how="vertical_relaxed")


def build_new_m2_college_ratings(links_df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes Opponent-, Schedule-, QB-, and Pass-Volume-Scaled Teammate-Adjusted M2 Ratings
    across all college receiver-games in `panel_v2_feat.parquet`.
    """
    links_df = append_undrafted_college_prospects(links_df)

    panel = pl.read_parquet(PANEL_V2_FEAT_PATH).with_columns(
        pl.col("athlete_id").cast(pl.String)
    )
    rec_seasons = pl.read_parquet(RECEIVER_SEASONS_PATH).with_columns(
        pl.col("athlete_id").cast(pl.String)
    )

    print(
        f"\n[1/5] Solving Stage 1 New M2 College Rating Engine on {PANEL_V2_FEAT_PATH.name} "
        f"({panel.height:,} games, {panel['season'].min()}–{panel['season'].max()})..."
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

    drafted_ids = set(links_df["athlete_id"].cast(pl.String).drop_nulls().to_list())

    # 1A. Program Schedule Elo Lookup (from receiver_seasons.parquet)
    team_elo_lookup = (
        rec_seasons.filter(pl.col("season") >= 2018)
        .group_by("team")
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

    # Teammate NFL WR Room Density & Total Tracked Receiving TDs per (season, team)
    panel_player_season = (
        panel.filter(pl.col("pass_att") > 0)
        .group_by(["season", "team", "athlete_id"])
        .agg(
            ps_games=pl.len(),
            ps_rec=pl.col("rec").clip(lower_bound=0).sum(),
            ps_yds=pl.col("yds").clip(lower_bound=0).sum(),
            ps_td=pl.col("td").clip(lower_bound=0).sum(),
        )
        .with_columns(
            is_active_nfl_wr=(
                (pl.col("athlete_id").is_in(drafted_ids) & (pl.col("ps_yds") >= 480))
                | ((pl.col("season") >= 2023) & (pl.col("ps_yds") >= 700))
            ).cast(pl.Int32),
        )
    )

    room_context = (
        panel_player_season.group_by(["season", "team"])
        .agg(
            room_active_nfl_wrs=pl.col("is_active_nfl_wr").sum(),
            room_nfl_wr_yds=pl.col("ps_yds").filter(pl.col("is_active_nfl_wr") == 1).sum(),
            true_team_rec_td=pl.col("ps_td").sum(),
        )
        .join(team_unique_games, on=["season", "team"], how="left")
    )

    # 1C. Game-Level Opponent-Adjusted M2 Residuals (`panel_v2_feat.parquet`)
    def_q25 = float(panel["def_rating"].quantile(0.25))

    panel_games = (
        panel.filter(pl.col("pass_att") >= MIN_PASS_ATT_GAME)
        .with_columns(
            eff_pass_att=(0.70 * pl.col("pass_att").cast(pl.Float64) + 0.30 * 32.0),
            played_game=((pl.col("rec") > 0) | (pl.col("yds") > 0)),
        )
        .with_columns(
            game_exp_wt=pl.when(pl.col("played_game"))
            .then(pl.col("pass_att").cast(pl.Float64))
            .otherwise(pl.col("pass_att").cast(pl.Float64) * 0.22),
            y_pa=(pl.col("yds").clip(lower_bound=0) / pl.col("eff_pass_att")),
            rec_pa=(pl.col("rec").clip(lower_bound=0) / pl.col("eff_pass_att")),
            td_pa=(pl.col("td").clip(lower_bound=0) / pl.col("eff_pass_att")),
            game_yds_share=(
                pl.col("yds").clip(lower_bound=0) / pl.col("pass_yds").clip(lower_bound=120.0)
            ).clip(0.0, 0.75),
            is_tough_def=(pl.col("def_rating") <= def_q25),
            is_fbs=(pl.col("opp_class") == "fbs").fill_null(True),
            pace_deflator=(35.0 / pl.col("pass_att").clip(lower_bound=35.0)) ** 0.35,
        )
        .with_columns(
            adj_game_yptpa=(
                (
                    pl.col("y_pa") * pl.col("pace_deflator")
                    - pl.col("def_rating").fill_null(0.0) * 0.38
                    - ((pl.col("off_oth_ypa").fill_null(7.4) - 7.6).clip(-1.0, 2.5) * 0.06)
                )
                * pl.when(pl.col("is_fbs")).then(1.0).otherwise(0.75)
            ),
            adj_game_share=(
                pl.col("game_yds_share") * pl.when(pl.col("is_fbs")).then(1.0).otherwise(0.75)
                - pl.col("def_rating").fill_null(0.0) * 0.035
            ),
            consistent_hit=((pl.col("y_pa") - pl.col("def_rating").fill_null(0.0) * 0.25) >= 1.85).cast(pl.Float64),
        )
    )

    season_panel_agg = (
        panel_games.group_by(["athlete_id", "season", "team"])
        .agg(
            s_panel_rows=pl.len(),
            s_active_games=pl.col("played_game").cast(pl.Int32).sum().clip(lower_bound=1),
            s_rec=pl.col("rec").clip(lower_bound=0).sum(),
            s_yds=pl.col("yds").clip(lower_bound=0).sum(),
            s_td=pl.col("td").clip(lower_bound=0).sum(),
            s_pass_att=pl.col("game_exp_wt").sum(),
            s_adj_yptpa=((pl.col("adj_game_yptpa") * pl.col("game_exp_wt")).sum() / pl.col("game_exp_wt").sum().clip(lower_bound=1.0)),
            s_adj_share=((pl.col("adj_game_share") * pl.col("game_exp_wt")).sum() / pl.col("game_exp_wt").sum().clip(lower_bound=1.0)),
            s_raw_yptpa=((pl.col("y_pa") * pl.col("game_exp_wt")).sum() / pl.col("game_exp_wt").sum().clip(lower_bound=1.0)),
            s_def_rating=pl.col("def_rating").mean(),
            s_oth_ypa=pl.col("off_oth_ypa").mean(),
            s_tough_yptpa=pl.col("adj_game_yptpa").filter(pl.col("is_tough_def") & pl.col("played_game")).mean(),
            s_consistency=((pl.col("consistent_hit") * pl.col("game_exp_wt")).sum() / pl.col("game_exp_wt").sum().clip(lower_bound=1.0)),
        )
        .join(room_context, on=["season", "team"], how="left")
        .join(team_elo_lookup, on="team", how="left")
    )

    rs_sub = rec_seasons.select([
        "athlete_id", "season", "team",
        pl.col("games").alias("rs_games"),
        pl.col("dominator").alias("rs_dominator"),
        pl.col("yds_share").alias("rs_yds_share"),
        pl.col("td_share").alias("rs_td_share"),
        pl.col("avg_opp_elo").alias("rs_opp_elo"),
        pl.col("team_games").alias("rs_team_games"),
        pl.col("team_yds").alias("rs_team_yds"),
    ])

    season_panel_agg = (
        season_panel_agg.join(rs_sub, on=["athlete_id", "season", "team"], how="left")
        .with_columns(
            s_games=pl.coalesce(["rs_games", "s_active_games"]).clip(lower_bound=1),
            eff_opp_elo=pl.coalesce(["rs_opp_elo", "program_opp_elo"]).fill_null(1500.0),
            denom_team_yds=pl.coalesce(["rs_team_yds", "true_team_pass_yds"]).clip(lower_bound=1200.0),
            calc_yds_share=(pl.col("s_yds") / pl.col("true_team_pass_yds").clip(lower_bound=1200.0)).clip(0.02, 0.52),
            calc_td_share=(pl.col("s_td") / pl.col("true_team_rec_td").clip(lower_bound=8.0)).clip(0.0, 0.55),
            team_pass_att_per_g=(
                pl.col("true_team_pass_att").cast(pl.Float64)
                / pl.coalesce(["rs_team_games", "true_team_games"]).clip(lower_bound=6).cast(pl.Float64)
            ).fill_null(32.0),
        )
        .with_columns(
            eff_yds_share=pl.coalesce(["rs_yds_share", "calc_yds_share"]),
            eff_dominator=pl.coalesce([
                "rs_dominator",
                0.5 * (pl.col("calc_yds_share") + pl.col("calc_td_share")),
            ]),
            eff_games_share=(
                pl.col("s_games").cast(pl.Float64)
                / pl.coalesce(["rs_team_games", "true_team_games"]).clip(lower_bound=6).cast(pl.Float64)
            ).clip(0.15, 1.0),
            raw_co_star_nfl_wrs=(pl.col("room_active_nfl_wrs").fill_null(1) - 1).clip(0, 3),
            co_star_nfl_yds=(pl.col("room_nfl_wr_yds").fill_null(0) - pl.col("s_yds")).clip(lower_bound=0),
            rec_per_game=(pl.col("s_rec") / pl.col("s_games").clip(lower_bound=1)),
            yds_per_game=(pl.col("s_yds") / pl.col("s_games").clip(lower_bound=1)),
            season_ypc=(pl.col("s_yds") / pl.col("s_rec").clip(lower_bound=1)),
            pass_vol_scale=(
                ((pl.col("team_pass_att_per_g") - 20.0) / 13.0).clip(0.20, 1.0) ** 1.40
            ),
        )
        .with_columns(
            raw_co_star_yds_share=(
                pl.col("co_star_nfl_yds") / pl.col("denom_team_yds")
            ).clip(0.0, 0.55),
        )
        .with_columns(
            co_star_yds_share=(pl.col("raw_co_star_yds_share") * pl.col("pass_vol_scale")),
            co_star_nfl_wrs=(pl.col("raw_co_star_nfl_wrs").cast(pl.Float64) * pl.col("pass_vol_scale")),
        )
        .with_columns(
            pie_adj_dominator=(
                pl.col("eff_dominator") / (1.0 - 0.62 * pl.col("co_star_yds_share")).clip(lower_bound=0.62)
            ).clip(0.04, 0.58),
        )
    )

    # 1D. Compute the True Season-Level New M2 Rating (`s_m2_rating`)
    prior_yptpa = 1.60
    k_att = 180.0

    season_panel_agg = season_panel_agg.with_columns(
        shrunk_adj_yptpa=(
            (pl.col("s_adj_yptpa") * pl.col("s_pass_att") + prior_yptpa * k_att)
            / (pl.col("s_pass_att") + k_att)
        ),
        sos_mult=(
            1.0
            + ((pl.col("eff_opp_elo") - 1510.0) / 920.0).clip(-0.22, 0.20)
            - (pl.col("s_def_rating").fill_null(0.0) * 0.085).clip(-0.10, 0.12)
        ),
        teammate_mult=(
            1.0
            + (pl.col("co_star_yds_share") * 0.42).clip(0.0, 0.22)
            + (pl.col("co_star_nfl_wrs") * 0.035)
        ),
        chain_explosive_mult=(
            1.0
            + ((pl.col("rec_per_game") - 4.2) / 20.0).clip(-0.14, 0.09)
            + ((pl.col("season_ypc") - 13.8) / 45.0).clip(-0.08, 0.09)
        ),
    ).with_columns(
        s_m2_rating=(
            pl.col("shrunk_adj_yptpa")
            * pl.col("sos_mult")
            * pl.col("teammate_mult")
            * pl.col("chain_explosive_mult")
        ),
        career_season_wt=(
            pl.col("s_pass_att") * (pl.col("s_rec").cast(pl.Float64) / 28.0).clip(0.20, 1.45)
        ),
    )

    # 1E. Step A: Extract True Chronological Year-1 (Freshman) Dominator BEFORE Filtering
    chronological_first = (
        season_panel_agg.sort(["athlete_id", "season"])
        .group_by("athlete_id")
        .agg(
            chrono_first_season=pl.col("season").first(),
            yr1_dominator=pl.col("pie_adj_dominator").first(),
            yr1_m2=pl.col("s_m2_rating").first(),
            yr1_rec=pl.col("s_rec").first(),
            career_rec=pl.col("s_rec").sum(),
            career_yds=pl.col("s_yds").sum(),
            career_td=pl.col("s_td").sum(),
            career_pass_att=pl.col("s_pass_att").sum(),
        )
    )

    # 1E. Step B: Filter Out Injured Final Cameos (<=5 games) & Low-Snap Bench Cameos
    base_s = season_panel_agg.filter(pl.col("s_games") >= 3).sort(["athlete_id", "season"])
    player_s_meta = (
        base_s.group_by("athlete_id")
        .agg(
            max_s=pl.col("season").max(),
            n_starter_s=((pl.col("s_games") >= 6) & ((pl.col("s_rec") >= 16) | (pl.col("eff_yds_share") >= 0.13))).cast(pl.Int32).sum(),
            n_healthy_s=(pl.col("s_games") >= 6).cast(pl.Int32).sum(),
        )
    )

    valid_seasons = (
        base_s.join(player_s_meta, on="athlete_id", how="left")
        .filter(
            pl.when(pl.col("n_starter_s") >= 1)
            .then((pl.col("s_games") >= 6) & ((pl.col("s_rec") >= 16) | (pl.col("eff_yds_share") >= 0.13)))
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
            m2_final=pl.col("s_m2_rating").last(),
            m2_career=((pl.col("s_m2_rating") * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
            peak_dominator=pl.col("pie_adj_dominator").max(),
            raw_peak_dominator=pl.col("eff_dominator").max(),
            final_dominator=pl.col("pie_adj_dominator").last(),
            peak_rec_per_g=pl.col("rec_per_game").max(),
            final_rec_per_g=pl.col("rec_per_game").last(),
            final_games_played=pl.col("s_games").last(),
            final_games_share=pl.col("eff_games_share").last(),
            max_co_star_wrs=pl.col("co_star_nfl_wrs").max(),
            max_co_star_yds=pl.col("co_star_nfl_yds").max(),
            max_co_star_share=pl.col("co_star_yds_share").max(),
            max_raw_co_star_share=pl.col("raw_co_star_yds_share").max(),
            mean_opp_elo=((pl.col("eff_opp_elo") * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
            mean_def_rating=((pl.col("s_def_rating") * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
            mean_oth_ypa=((pl.col("s_oth_ypa") * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
            mean_tough_yptpa=pl.col("s_tough_yptpa").mean(),
            mean_adj_yptpa=((pl.col("s_adj_yptpa") * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
            mean_consistency=((pl.col("s_consistency") * pl.col("career_season_wt")).sum() / pl.col("career_season_wt").sum().clip(lower_bound=1.0)),
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
        m2_peak=pl.coalesce(["m2_peak", pl.col("leg_final") * 1.35 + 1.85]).fill_null(1.80),
        m2_final=pl.coalesce(["m2_final", pl.col("leg_final") * 1.35 + 1.80]).fill_null(1.75),
        m2_career=pl.coalesce(["m2_career", pl.col("leg_career") * 1.35 + 1.75]).fill_null(1.70),
        peak_dominator=pl.col("peak_dominator").fill_null(0.28),
        raw_peak_dominator=pl.col("raw_peak_dominator").fill_null(pl.col("peak_dominator")).fill_null(0.26),
        final_dominator=pl.col("final_dominator").fill_null(0.26),
        yr1_dominator=pl.col("yr1_dominator").fill_null(0.16),
        active_starter_seasons=pl.col("active_starter_seasons").fill_null(2),
        college_seasons=(
            pl.col("draft_year").cast(pl.Int64) - pl.col("chrono_first_season").cast(pl.Int64)
        ).clip(3, 5).fill_null(3),
        peak_rec_per_g=pl.col("peak_rec_per_g").fill_null(4.8),
        final_rec_per_g=pl.col("final_rec_per_g").fill_null(4.5),
        final_games_played=pl.col("final_games_played").fill_null(11),
        final_games_share=pl.coalesce(["final_games_share", "leg_g_share"]).fill_null(0.88),
        max_co_star_wrs=pl.col("max_co_star_wrs").fill_null(0.0),
        max_co_star_yds=pl.col("max_co_star_yds").fill_null(0.0),
        max_co_star_share=pl.col("max_co_star_share").fill_null(0.0),
        max_raw_co_star_share=pl.col("max_raw_co_star_share").fill_null(0.0),
        career_rec=pl.col("career_rec").fill_null(95),
        career_yds=pl.col("career_yds").fill_null(1350),
        career_td=pl.col("career_td").fill_null(11),
        career_pass_att=pl.col("career_pass_att").fill_null(750),
        mean_opp_elo=pl.col("mean_opp_elo").fill_null(1510.0),
        mean_def_rating=pl.col("mean_def_rating").fill_null(0.0),
        mean_consistency=pl.col("mean_consistency").fill_null(0.55),
    )

    df = df.with_columns(
        standout_freshman_bonus=(pl.col("yr1_dominator") - 0.20).clip(lower_bound=0.0, upper_bound=0.25),
        is_optout_or_injured_final=(
            ((pl.col("draft_year") == 2021) & (pl.col("last_season_panel") <= 2019))
            | (pl.col("last_season_panel") < (pl.col("draft_year") - 1))
            | (pl.col("final_games_played") <= 5)
            | (pl.col("final_games_share") <= 0.50)
        ),
    ).with_columns(
        m2_final_eff=pl.when(pl.col("is_optout_or_injured_final"))
        .then(0.85 * pl.col("m2_peak") + 0.15 * pl.col("m2_final"))
        .otherwise(0.80 * pl.col("m2_final") + 0.20 * pl.col("m2_peak")),
        final_regression_pen=pl.when(
            (~pl.col("is_optout_or_injured_final")) & (pl.col("m2_final") < pl.col("m2_peak") * 0.88)
        )
        .then((pl.col("m2_peak") - pl.col("m2_final")) * 0.65)
        .otherwise(0.0),
    )

    df = df.with_columns(
        college_ypc=(pl.col("career_yds") / pl.col("career_rec").clip(lower_bound=1)).clip(8.5, 23.5),
        college_rec_per_att=(pl.col("career_rec") / pl.col("career_pass_att").clip(lower_bound=50)).clip(0.04, 0.28),
        college_td_per_rec=(pl.col("career_td") / pl.col("career_rec").clip(lower_bound=1)).clip(0.0, 0.35),
        vs_top_def_ratio=(
            pl.col("mean_tough_yptpa").fill_null(pl.col("mean_adj_yptpa"))
            / pl.col("mean_adj_yptpa").clip(lower_bound=0.6)
        ).fill_null(1.0).clip(0.65, 1.45),
        qb_independence_ratio=(
            pl.col("mean_adj_yptpa").fill_null(1.8)
            / (pl.col("mean_oth_ypa").fill_null(7.2).clip(lower_bound=4.5) * 0.25)
        ).clip(0.55, 1.75),
        opp_defense_difficulty=(
            ((pl.col("mean_opp_elo") - 1510.0) / 180.0)
            - pl.col("mean_def_rating") * 0.45
        ).clip(-0.90, 0.95),
    ).with_columns(
        solo_alpha_credit=pl.when(
            (pl.col("max_raw_co_star_share") < 0.15)
            & (pl.col("college_seasons") <= 3)
            & (pl.col("mean_opp_elo") >= 1525.0)
            & (pl.col("raw_peak_dominator") >= 0.31)
            & (pl.col("peak_rec_per_g") >= 6.0)
            & (pl.col("vs_top_def_ratio") >= 0.96)
        )
        .then(
            (
                (pl.col("raw_peak_dominator") - 0.29) * 1.65
                + (pl.col("peak_rec_per_g") - 5.8).clip(0.0, 2.5) * 0.08
            ).clip(0.0, 0.42)
        )
        .otherwise(0.0),
    ).with_columns(
        teammate_wr_boost=pl.max_horizontal(
            pl.col("max_co_star_share") * 1.15 + pl.col("max_co_star_wrs") * 0.10,
            pl.col("solo_alpha_credit"),
        ).clip(0.0, 0.75),
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
    2. Links Combine/Pro-Day measurables from `combine.parquet`.
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

    c_seasons = int(r.get("college_seasons") or 3)
    d_age = (
        float(r["draft_age"])
        if r.get("draft_age") is not None
        else (21.2 if c_seasons <= 3 else (22.3 if c_seasons == 4 else 23.1))
    )
    fresh_bon = float(r.get("standout_freshman_bonus") or 0.0)

    age_score = (22.1 - d_age) * 0.42 + (3.5 - min(5, max(3, c_seasons))) * 0.42 + fresh_bon * 1.15

    opp_elo = float(r.get("mean_opp_elo") or 1510.0)
    def_rat = float(r.get("mean_def_rating") or 0.0)
    sos_score = ((opp_elo - 1520.0) / 140.0) - (def_rat * 0.45)

    tm_boost = float(r.get("teammate_wr_boost") or 0.0)

    pk_dom = float(r.get("peak_dominator") or 0.28)
    sos_weight = float(np.clip((opp_elo - 1300.0) / 300.0, 0.35, 1.15))
    sos_adj_dom = pk_dom * sos_weight

    top_def = float(r.get("vs_top_def_ratio") or 1.0)
    g_share = float(r.get("final_games_share") or 0.88)
    cons_rate = float(r.get("mean_consistency") or 0.55)
    resilience = (
        (top_def - 1.0) * 0.55
        + (g_share - 0.85) * 0.45
        + (cons_rate - 0.55) * 0.65
        + fresh_bon * 0.75
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