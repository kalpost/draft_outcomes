import polars as pl
from pathlib import Path

pl.Config.set_tbl_rows(20)
Path("data/processed").mkdir(parents=True, exist_ok=True)

MIN_REC_TRAIN = 5      # locked in advance. Sensitivity runs at 3 and 8
TRAIN_FRAC = 2 / 3

rg = pl.read_parquet("data/processed/receiver_games.parquet")
tg = pl.read_parquet("data/processed/team_games.parquet")

# Split by game: a calendar cutoff per season, so both teams' rows from one game
# always fall on the same side. All postseason games go to the test period.
base = tg.filter(pl.col("pass_att") >= 10)
cut = (base.filter(pl.col("seasonType") == "regular")
       .group_by("season")
       .agg(cut=pl.col("week").quantile(TRAIN_FRAC, interpolation="nearest")))

games = (base.join(cut, on="season", how="left")
         .with_columns(is_test=(pl.col("seasonType") == "postseason") | (pl.col("week") > pl.col("cut")))
         .select("game_id", "season", "week", "seasonType", "team", "opponent", "opp_class",
                 "is_home", "neutralSite", "pass_att", "team_points", "opp_points", "is_test"))

# Save the split so every later script uses exactly the same one
games.select("game_id", "team", "season", "is_test").write_parquet("data/processed/game_split.parquet")

stats = rg.select("game_id", "team", "athlete_id", "name", "rec", "yds", "td")
team_tot = rg.group_by(["game_id", "team"]).agg(team_yds=pl.col("yds").sum())
train_games = games.filter(~pl.col("is_test")).select("game_id", "team", "season")


def build(min_rec):
    # A receiver joins his team's group only if he has enough catches in TRAINING games
    grp = (stats.join(train_games, on=["game_id", "team"], how="inner")
           .group_by(["season", "team", "athlete_id"])
           .agg(name=pl.col("name").first(), train_rec=pl.col("rec").sum())
           .filter(pl.col("train_rec") >= min_rec))
    return (games.join(grp, on=["season", "team"], how="inner")
            .join(stats.select("game_id", "team", "athlete_id", "rec", "yds", "td"),
                  on=["game_id", "team", "athlete_id"], how="left")
            .with_columns(pl.col("rec").fill_null(0), pl.col("yds").fill_null(0), pl.col("td").fill_null(0))
            .with_columns(y=pl.col("yds") / pl.col("pass_att")))


def coverage(p, test):
    gy = (p.filter(pl.col("is_test") == test)
          .group_by(["game_id", "team"]).agg(g=pl.col("yds").sum())
          .join(team_tot, on=["game_id", "team"])
          .filter(pl.col("team_yds") > 0))
    return round(gy["g"].sum() / gy["team_yds"].sum(), 3)


print("team-games:", games.height, "| share in test:", round(games["is_test"].mean(), 3))
print("postseason share of test games:",
      round((games.filter(pl.col("is_test"))["seasonType"] == "postseason").mean(), 3))
print(games.filter(pl.col("season") == 2023)
      .group_by(["season", "team"]).agg(n=pl.len(), n_test=pl.col("is_test").sum())
      .select(mean_games=pl.col("n").mean(), mean_test=pl.col("n_test").mean()))

for m in (3, 5, 8):
    p = build(m)
    z = (p.group_by("is_test")
         .agg(n=pl.len(), zero_share=(pl.col("yds") == 0).mean())
         .sort("is_test"))
    print(f"\nmin catches in training games = {m}: rows {p.height}, receivers {p['athlete_id'].n_unique()}")
    print(z)
    print("share of team receiving yards covered | train:", coverage(p, False),
          "| test:", coverage(p, True))

panel = build(MIN_REC_TRAIN)
panel.write_parquet("data/processed/panel_v2.parquet")

h = panel.filter((pl.col("name") == "Marvin Harrison Jr.") & (pl.col("season") == 2023))
print("\nHarrison 2023:", h.height, "rows | test rows:", h["is_test"].sum(),
      "| test yards:", h.filter(pl.col("is_test"))["yds"].sum())