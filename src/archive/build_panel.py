import polars as pl
pl.Config.set_tbl_rows(20)

rg = pl.read_parquet("data/processed/receiver_games.parquet")
tg = pl.read_parquet("data/processed/team_games.parquet")
gm = pl.read_parquet("data/processed/games.parquet")
qb = pl.read_parquet("data/processed/team_game_qb.parquet").select("game_id", "team", "qb_id")

MIN_REC = 8  # catches in a season to join that team-season's receiving group

grp = (rg.group_by(["season", "team", "athlete_id"])
         .agg(name=pl.col("name").first(), season_rec=pl.col("rec").sum())
         .filter(pl.col("season_rec") >= MIN_REC))

games = (tg.filter(pl.col("pass_att") >= 10)
           .select("game_id", "season", "week", "seasonType", "team", "opponent", "opp_class",
                   "is_home", "neutralSite", "pass_att", "team_points", "opp_points"))

stats = rg.select("game_id", "team", "athlete_id", "rec", "yds", "td")
cls = gm.select(pl.col("id").alias("game_id"), "homeTeam", "homeClassification", "awayClassification")

panel = (games.join(grp, on=["season", "team"], how="inner")
         .join(stats, on=["game_id", "team", "athlete_id"], how="left")
         .with_columns(pl.col("rec").fill_null(0), pl.col("yds").fill_null(0), pl.col("td").fill_null(0))
         .join(cls, on="game_id", how="left")
         .with_columns(own_class=pl.when(pl.col("team") == pl.col("homeTeam"))
                       .then(pl.col("homeClassification")).otherwise(pl.col("awayClassification")))
         .drop(["homeTeam", "homeClassification", "awayClassification"])
         .join(qb, on=["game_id", "team"], how="left")
         .with_columns(y=pl.col("yds") / pl.col("pass_att")))
panel.write_parquet("data/processed/panel.parquet")

print("panel rows:", panel.height, "| receivers:", panel["athlete_id"].n_unique(),
      "| team-seasons:", panel.select("season", "team").unique().height)
print("share of rows with zero yards:", round((panel["yds"] == 0).mean(), 3))
print(panel.group_by("own_class").len().sort("len", descending=True))
print("rows with no QB id (non-FBS teams, mostly):", panel["qb_id"].is_null().sum())

cov = (panel.group_by(["game_id", "team"]).agg(g=pl.col("yds").sum())
       .join(rg.group_by(["game_id", "team"]).agg(t=pl.col("yds").sum()), on=["game_id", "team"])
       .filter(pl.col("t") > 0))
print("share of team receiving yards covered by the group:", round(cov["g"].sum() / cov["t"].sum(), 3))

h = panel.filter((pl.col("name") == "Marvin Harrison Jr.") & (pl.col("season") == 2023))
print("Harrison 2023 panel rows:", h.height, "| zero-yard games:", (h["yds"] == 0).sum(), "| total yards:", h["yds"].sum())