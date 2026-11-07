"""
College Receiver Game-Panel Builder (`build_panel_v2.py`)
=========================================================
Constructs `game_split.parquet` and `panel_v2.parquet` from `receiver_games.parquet`
and `team_games.parquet` using global parameters from `config.py`.
"""

from __future__ import annotations

import polars as pl

from config import (
    CFB_END_YEAR,
    GAME_SPLIT_PATH,
    MIN_PASS_ATT_GAME,
    MIN_REC_SENSITIVITY,
    MIN_REC_TRAIN,
    PANEL_TRAIN_FRAC,
    PANEL_V2_PATH,
    RECEIVER_GAMES_PATH,
    TEAM_GAMES_PATH,
)

pl.Config.set_tbl_rows(25)


def build_panel_v2() -> pl.DataFrame:
    rg = pl.read_parquet(RECEIVER_GAMES_PATH)
    tg = pl.read_parquet(TEAM_GAMES_PATH)

    # Filter to real pass games
    base = tg.filter(pl.col("pass_att") >= MIN_PASS_ATT_GAME)

    # Calculate calendar cutoff per season strictly on regular season games
    cut = (
        base.filter(pl.col("seasonType") == "regular")
        .group_by("season")
        .agg(cut=pl.col("week").quantile(PANEL_TRAIN_FRAC, interpolation="nearest"))
    )

    # Join cutoff back and assign is_test flag.
    # All postseason games (bowls + CFP) and late regular-season games go to the test set.
    games = (
        base.join(cut, on="season", how="left")
        .with_columns(
            is_test=(
                (pl.col("seasonType") == "postseason")
                | (pl.col("week").fill_null(99) > pl.col("cut").fill_null(9))
            ).fill_null(True)
        )
        .select(
            "game_id", "season", "week", "seasonType", "team", "opponent", "opp_class",
            "is_home", "neutralSite", "pass_att", "team_points", "opp_points", "is_test",
        )
    )

    # Save the exact split so all models use the identical holdout definition
    games.select("game_id", "team", "season", "is_test").write_parquet(GAME_SPLIT_PATH)

    stats = rg.select("game_id", "team", "athlete_id", "name", "rec", "yds", "td")
    team_tot = rg.group_by(["game_id", "team"]).agg(team_yds=pl.col("yds").sum())
    train_games = games.filter(~pl.col("is_test")).select("game_id", "team", "season")

    def build_for_threshold(min_rec: int) -> pl.DataFrame:
        # A receiver joins his team's group only if he has enough catches in TRAINING games
        grp = (
            stats.join(train_games, on=["game_id", "team"], how="inner")
            .group_by(["season", "team", "athlete_id"])
            .agg(name=pl.col("name").first(), train_rec=pl.col("rec").sum())
            .filter(pl.col("train_rec") >= min_rec)
        )

        return (
            games.join(grp, on=["season", "team"], how="inner")
            .join(
                stats.select("game_id", "team", "athlete_id", "rec", "yds", "td"),
                on=["game_id", "team", "athlete_id"],
                how="left",
            )
            .with_columns(
                pl.col("rec").fill_null(0),
                pl.col("yds").fill_null(0),
                pl.col("td").fill_null(0),
            )
            .with_columns(y=pl.col("yds") / pl.col("pass_att"))
        )

    def coverage(p: pl.DataFrame, test: bool) -> float:
        gy = (
            p.filter(pl.col("is_test") == test)
            .group_by(["game_id", "team"])
            .agg(g=pl.col("yds").sum())
            .join(team_tot, on=["game_id", "team"])
            .filter(pl.col("team_yds") > 0)
        )
        return round(float(gy["g"].sum() / gy["team_yds"].sum()), 3)

    print("Team-games:", games.height, "| Overall share in test:", round(float(games["is_test"].mean()), 3))
    print(
        "Postseason share of test games:",
        round(float((games.filter(pl.col("is_test"))["seasonType"] == "postseason").mean()), 3),
    )

    print(f"\nRecent season game-split breakdown (>= {CFB_END_YEAR - 4}):")
    print(
        games.filter(pl.col("season") >= (CFB_END_YEAR - 4))
        .group_by("season")
        .agg(
            total_team_games=pl.len(),
            test_games=pl.col("is_test").sum(),
            test_pct=(pl.col("is_test").mean() * 100.0).round(1),
        )
        .sort("season")
    )

    for m in MIN_REC_SENSITIVITY:
        p = build_for_threshold(m)
        z = (
            p.group_by("is_test")
            .agg(n=pl.len(), zero_share=(pl.col("yds") == 0).mean())
            .sort("is_test")
        )
        print(f"\nmin catches in training games = {m}: rows {p.height}, receivers {p['athlete_id'].n_unique()}")
        print(z)
        print(
            "share of team receiving yards covered | train:",
            coverage(p, False),
            "| test:",
            coverage(p, True),
        )

    panel = build_for_threshold(MIN_REC_TRAIN)
    panel.write_parquet(PANEL_V2_PATH)
    print(f"\nSuccessfully wrote {panel.height} rows to {PANEL_V2_PATH}")
    return panel


if __name__ == "__main__":
    build_panel_v2()