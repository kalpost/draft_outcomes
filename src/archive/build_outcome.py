import polars as pl
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)
draft = pl.read_parquet("data/raw/draft.parquet")
stats = pl.read_parquet("data/raw/stats.parquet")

wr = (draft.filter((pl.col("position") == "WR") & pl.col("season").is_between(2013, 2025))
      .select(["season", "round", "pick", "team", "gsis_id", "pfr_player_id", "pfr_player_name", "college"])
      .with_row_index("row"))

reg = stats.filter(pl.col("season_type") == "REG")

rookie = (wr.join(reg, left_on=["gsis_id", "season"], right_on=["player_id", "season"], how="left")
          .group_by("row")
          .agg(pl.col("season").first(), pl.col("pick").first(), pl.col("round").first(),
               pl.col("pfr_player_name").first(), pl.col("college").first(),
               rec_yards=pl.col("receiving_yards").sum(),
               targets=pl.col("targets").sum(),
               games=pl.col("week").drop_nulls().n_unique())
          .with_columns(team_games=pl.when(pl.col("season") >= 2021).then(17).otherwise(16))
          .with_columns(ypg=pl.col("rec_yards") / pl.col("team_games")))

print(rookie.height, "(should be 419)")
print("share with 0 yards:", (rookie["rec_yards"] == 0).mean())
print(rookie.select(pl.col("ypg").quantile(q).alias(f"q{q}") for q in (0.1, 0.25, 0.5, 0.75, 0.9)))
print(rookie.filter(pl.col("pfr_player_name").is_in(["Justin Jefferson", "Ja'Marr Chase"])))
rookie.write_parquet("data/processed/rookie_outcomes.parquet")