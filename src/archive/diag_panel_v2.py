import polars as pl
pl.Config.set_tbl_rows(30)

p = pl.read_parquet("data/processed/panel_v2.parquet")
t = p.filter(pl.col("is_test"))
print("test zero share by season type:")
print(t.group_by("seasonType").agg(n=pl.len(), zero_share=(pl.col("yds") == 0).mean()))
print("regular-season test zero share by year:")
print(t.filter(pl.col("seasonType") == "regular").group_by("season")
      .agg(n=pl.len(), zero_share=(pl.col("yds") == 0).mean()).sort("season"))

links = pl.read_parquet("data/processed/wr_links_final.parquet").filter(pl.col("athlete_id").is_not_null())
seasons = p.group_by("athlete_id").agg(panel_seasons=pl.col("season").n_unique())
x = links.join(seasons, on="athlete_id", how="left").with_columns(pl.col("panel_seasons").fill_null(0))
print("drafted receivers with at least one panel season:", (x["panel_seasons"] > 0).sum(), "of", x.height)
print(x.group_by("panel_seasons").len().sort("panel_seasons"))
print(x.filter(pl.col("panel_seasons") == 0).select("pname", "pcollege", "draft_year", "college_games"))