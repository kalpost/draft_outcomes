import polars as pl
pl.Config.set_tbl_rows(30)
tg = pl.read_parquet("data/processed/team_games.parquet")
print(tg.group_by("season").agg(n=pl.len(), zero=(pl.col("pass_att") == 0).sum(),
      zero_with_catches=((pl.col("pass_att") == 0) & (pl.col("rec_total") > 0)).sum()).sort("season"))
z = tg.filter((pl.col("season") == 2021) & (pl.col("pass_att") == 0))
print(z.group_by(["seasonType", "opp_class"]).len().sort("len", descending=True))
print(z.group_by("week").len().sort("week"))
print(z.select(["team", "opponent", "opp_class", "week", "team_points", "rec_total"]).head(12))