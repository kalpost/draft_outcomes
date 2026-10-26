import polars as pl

draft = pl.read_parquet("data/raw/draft.parquet")
combine = pl.read_parquet("data/raw/combine.parquet")

wr = draft.filter((pl.col("position") == "WR") & pl.col("season").is_between(2013, 2025))
print(wr.group_by("season").len().sort("season"))
print("total WRs:", wr.height, "| missing gsis_id:", wr["gsis_id"].null_count())

cols = ["ht", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
cw = wr.join(combine.select(["pfr_id"] + cols), left_on="pfr_player_id", right_on="pfr_id", how="left")
print("rows before/after join:", wr.height, cw.height)  # should match
for c in cols:
    print(c, "missing:", round(cw[c].is_null().mean(), 2))

print(wr.select(["pfr_player_name", "season", "pick", "college", "age"]).head(10))