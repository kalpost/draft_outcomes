import polars as pl

r = pl.read_parquet("data/processed/rookie_outcomes.parquet")
z = r.filter(pl.col("rec_yards") == 0)

print("zeros by round:")
print(z.group_by("round").len().sort("round"))

print("early-round zeros (rounds 1-3):")
print(z.filter(pl.col("round") <= 3).select(["pfr_player_name", "season", "pick", "games", "targets"]))