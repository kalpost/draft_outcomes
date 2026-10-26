import polars as pl

rg = pl.read_parquet("data/processed/receiver_games.parquet").filter(pl.col("pass_att") > 0)

team_season = (rg.group_by(["season", "team", "game_id"])
    .agg(att=pl.col("pass_att").first(), yds=pl.col("yds").sum(), td=pl.col("td").sum())
    .group_by(["season", "team"])
    .agg(team_att=pl.col("att").sum(), team_yds=pl.col("yds").sum(),
         team_td=pl.col("td").sum(), team_games=pl.len()))

ps = (rg.group_by(["season", "team", "athlete_id", "name"])
    .agg(games=pl.len(), rec=pl.col("rec").sum(), yds=pl.col("yds").sum(),
         td=pl.col("td").sum(), avg_opp_elo=pl.col("opp_elo").mean())
    .join(team_season, on=["season", "team"], how="left")
    .with_columns(yptpa=pl.col("yds") / pl.col("team_att"),
                  yds_share=pl.col("yds") / pl.col("team_yds"),
                  td_share=pl.when(pl.col("team_td") > 0).then(pl.col("td") / pl.col("team_td")))
    .with_columns(dominator=(pl.col("yds_share") + pl.col("td_share")) / 2))

ps.write_parquet("data/processed/receiver_seasons.parquet")
print("receiver-seasons:", ps.height)
print(ps.filter((pl.col("name") == "Marvin Harrison Jr.") & (pl.col("season") == 2023))
        .select(["games", "yds", "team_att", "yptpa", "yds_share", "dominator"]))
print(ps.filter((pl.col("games") >= 8) & (pl.col("yds") >= 600)).sort("yptpa", descending=True)
        .select(["season", "team", "name", "yds", "team_att", "yptpa"]).head(8))