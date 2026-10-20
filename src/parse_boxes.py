import gzip, json, glob
from pathlib import Path
import polars as pl

Path("data/processed").mkdir(parents=True, exist_ok=True)

def to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None

# schedule
keep = ["id", "season", "week", "seasonType", "neutralSite", "homeTeam", "homeClassification",
        "homePoints", "homePregameElo", "awayTeam", "awayClassification", "awayPoints", "awayPregameElo"]
games = []
for f in sorted(glob.glob("data/raw/cfbd/games/*.json.gz")):
    games += [{k: g.get(k) for k in keep} for g in json.load(gzip.open(f, "rt"))]
games = pl.DataFrame(games, infer_schema_length=None).unique(subset=["id"])
games.write_parquet("data/processed/games.parquet")

# box scores
rec_rows, team_rows = [], []
for f in sorted(glob.glob("data/raw/cfbd/games_players/*.json.gz")):
    for g in json.load(gzip.open(f, "rt")):
        for t in g["teams"]:
            comp = att = 0
            rec = {}
            for cat in t["categories"]:
                if cat["name"] == "passing":
                    for ty in cat["types"]:
                        if ty["name"] == "C/ATT":
                            for a in ty["athletes"]:
                                try:
                                    c, n = a["stat"].split("/")
                                    comp += int(c); att += int(n)
                                except (ValueError, AttributeError):
                                    pass
                elif cat["name"] == "receiving":
                    for ty in cat["types"]:
                        if ty["name"] in ("REC", "YDS", "TD"):
                            for a in ty["athletes"]:
                                d = rec.setdefault(a["id"], {"athlete_id": a["id"], "name": a["name"]})
                                d[ty["name"].lower()] = to_int(a["stat"])
            team_rows.append({"game_id": g["id"], "team": t["team"], "pass_att": att,
                              "completions": comp,
                              "rec_total": sum(d.get("rec") or 0 for d in rec.values())})
            rec_rows += [{"game_id": g["id"], "team": t["team"], **d} for d in rec.values()]

rg = pl.DataFrame(rec_rows, infer_schema_length=None).unique(subset=["game_id", "team", "athlete_id"])
tg = pl.DataFrame(team_rows, infer_schema_length=None).unique(subset=["game_id", "team"])

def with_game(df):
    j = df.join(games, left_on="game_id", right_on="id", how="left")
    j = j.with_columns(
        team_matched=(pl.col("team") == pl.col("homeTeam")) | (pl.col("team") == pl.col("awayTeam")),
        is_home=pl.col("team") == pl.col("homeTeam"))
    return j.with_columns(
        opponent=pl.when(pl.col("is_home")).then(pl.col("awayTeam")).otherwise(pl.col("homeTeam")),
        opp_class=pl.when(pl.col("is_home")).then(pl.col("awayClassification")).otherwise(pl.col("homeClassification")),
        team_points=pl.when(pl.col("is_home")).then(pl.col("homePoints")).otherwise(pl.col("awayPoints")),
        opp_points=pl.when(pl.col("is_home")).then(pl.col("awayPoints")).otherwise(pl.col("homePoints")),
        team_elo=pl.when(pl.col("is_home")).then(pl.col("homePregameElo")).otherwise(pl.col("awayPregameElo")),
        opp_elo=pl.when(pl.col("is_home")).then(pl.col("awayPregameElo")).otherwise(pl.col("homePregameElo")),
    ).drop(["homeTeam", "awayTeam", "homeClassification", "awayClassification",
            "homePoints", "awayPoints", "homePregameElo", "awayPregameElo"])

tg, rg = with_game(tg), with_game(rg)
rg = rg.join(tg.select(["game_id", "team", "pass_att"]), on=["game_id", "team"], how="left")
tg.write_parquet("data/processed/team_games.parquet")
rg.write_parquet("data/processed/receiver_games.parquet")

print("receiver-game rows:", rg.height, "| team-games:", tg.height)
print("team name found in schedule:", round(tg["team_matched"].mean(), 4))
print("team-games where receptions == completions:", round((tg["completions"] == tg["rec_total"]).mean(), 3))
print(tg.group_by("season").agg(n=pl.len(), zero_pass_att=(pl.col("pass_att") == 0).sum()).sort("season"))
h = rg.filter((pl.col("name") == "Marvin Harrison Jr.") & (pl.col("season") == 2023))
print(h.select(rec=pl.col("rec").sum(), yds=pl.col("yds").sum(), td=pl.col("td").sum(), games=pl.len()))