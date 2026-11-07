"""
College Receiver Season Aggregator (`build_seasons.py`)
=======================================================
Aggregates `receiver_games.parquet` into season-level production, team passing
totals, opponent Elo, and dominator shares (`receiver_seasons.parquet`).
"""

from __future__ import annotations

import polars as pl

from config import RECEIVER_GAMES_PATH, RECEIVER_SEASONS_PATH


def build_seasons() -> pl.DataFrame:
    rg = pl.read_parquet(RECEIVER_GAMES_PATH).filter(pl.col("pass_att") > 0)

    team_season = (
        rg.group_by(["season", "team", "game_id"])
        .agg(
            att=pl.col("pass_att").first(),
            yds=pl.col("yds").sum(),
            td=pl.col("td").sum(),
        )
        .group_by(["season", "team"])
        .agg(
            team_att=pl.col("att").sum(),
            team_yds=pl.col("yds").sum(),
            team_td=pl.col("td").sum(),
            team_games=pl.len(),
        )
    )

    ps = (
        rg.group_by(["season", "team", "athlete_id", "name"])
        .agg(
            games=pl.len(),
            rec=pl.col("rec").sum(),
            yds=pl.col("yds").sum(),
            td=pl.col("td").sum(),
            avg_opp_elo=pl.col("opp_elo").mean(),
        )
        .join(team_season, on=["season", "team"], how="left")
        .with_columns(
            yptpa=pl.col("yds") / pl.col("team_att"),
            yds_share=pl.col("yds") / pl.col("team_yds"),
            td_share=pl.when(pl.col("team_td") > 0).then(pl.col("td") / pl.col("team_td")).otherwise(0.0),
        )
        .with_columns(dominator=(pl.col("yds_share") + pl.col("td_share")) / 2.0)
    )

    ps.write_parquet(RECEIVER_SEASONS_PATH)
    print(f"Successfully wrote {ps.height:,} receiver-seasons ({ps['season'].min()}–{ps['season'].max()}) to {RECEIVER_SEASONS_PATH}")
    return ps


if __name__ == "__main__":
    build_seasons()