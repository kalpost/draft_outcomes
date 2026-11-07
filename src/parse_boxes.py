"""
CFBD Schedule & Box Score Parser (`parse_boxes.py`)
===================================================
Parses compressed CFBD schedule (`games/*.json.gz`) and player box score
(`games_players/*.json.gz`) archives within `[CFB_START_YEAR, CFB_END_YEAR]`,
writing:
  - `data/processed/games.parquet`
  - `data/processed/team_games.parquet`
  - `data/processed/receiver_games.parquet`
  - `data/processed/receiver_seasons.parquet` (via `build_seasons()`)
"""

from __future__ import annotations

import gzip
import json
import polars as pl

from build_seasons import build_seasons
from config import (
    CFB_END_YEAR,
    CFB_START_YEAR,
    CFBD_BOXES_DIR,
    CFBD_GAMES_DIR,
    GAMES_PROCESSED_PATH,
    RECEIVER_GAMES_PATH,
    TEAM_GAMES_PATH,
)

SCHEDULE_COLS = [
    "id", "season", "week", "seasonType", "neutralSite",
    "homeTeam", "homeClassification", "homePoints", "homePregameElo",
    "awayTeam", "awayClassification", "awayPoints", "awayPregameElo",
]


def _to_int(x) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def parse_boxes(run_build_seasons: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    # 1. Parse schedule files
    game_records = []
    for f in sorted(CFBD_GAMES_DIR.glob("*.json.gz")):
        with gzip.open(f, "rt") as gz:
            for g in json.load(gz):
                season = _to_int(g.get("season"))
                if season is not None and (CFB_START_YEAR <= season <= CFB_END_YEAR):
                    game_records.append({k: g.get(k) for k in SCHEDULE_COLS})

    games = pl.DataFrame(game_records, infer_schema_length=None).unique(subset=["id"])
    games.write_parquet(GAMES_PROCESSED_PATH)

    # 2. Parse player box scores
    rec_rows, team_rows = [], []
    for f in sorted(CFBD_BOXES_DIR.glob("*.json.gz")):
        with gzip.open(f, "rt") as gz:
            for g in json.load(gz):
                for t in g.get("teams", []):
                    comp = att = 0
                    rec: dict[str, dict] = {}
                    for cat in t.get("categories", []):
                        if cat.get("name") == "passing":
                            for ty in cat.get("types", []):
                                if ty.get("name") == "C/ATT":
                                    for a in ty.get("athletes", []):
                                        try:
                                            c, n = str(a.get("stat", "")).split("/")
                                            comp += int(c)
                                            att += int(n)
                                        except (ValueError, AttributeError):
                                            pass
                        elif cat.get("name") == "receiving":
                            for ty in cat.get("types", []):
                                if ty.get("name") in ("REC", "YDS", "TD"):
                                    for a in ty.get("athletes", []):
                                        aid = str(a.get("id"))
                                        d = rec.setdefault(
                                            aid,
                                            {"athlete_id": aid, "name": a.get("name")},
                                        )
                                        d[ty["name"].lower()] = _to_int(a.get("stat"))
                    team_rows.append({
                        "game_id": g["id"],
                        "team": t["team"],
                        "pass_att": att,
                        "completions": comp,
                        "rec_total": sum(d.get("rec") or 0 for d in rec.values()),
                    })
                    rec_rows.extend({"game_id": g["id"], "team": t["team"], **d} for d in rec.values())

    rg = pl.DataFrame(rec_rows, infer_schema_length=None).unique(subset=["game_id", "team", "athlete_id"])
    tg = pl.DataFrame(team_rows, infer_schema_length=None).unique(subset=["game_id", "team"])

    def _with_game(df: pl.DataFrame) -> pl.DataFrame:
        j = df.join(games, left_on="game_id", right_on="id", how="inner")
        j = j.with_columns(
            team_matched=(pl.col("team") == pl.col("homeTeam")) | (pl.col("team") == pl.col("awayTeam")),
            is_home=pl.col("team") == pl.col("homeTeam"),
        )
        return j.with_columns(
            opponent=pl.when(pl.col("is_home")).then(pl.col("awayTeam")).otherwise(pl.col("homeTeam")),
            opp_class=pl.when(pl.col("is_home")).then(pl.col("awayClassification")).otherwise(pl.col("homeClassification")),
            team_points=pl.when(pl.col("is_home")).then(pl.col("homePoints")).otherwise(pl.col("awayPoints")),
            opp_points=pl.when(pl.col("is_home")).then(pl.col("awayPoints")).otherwise(pl.col("homePoints")),
            team_elo=pl.when(pl.col("is_home")).then(pl.col("homePregameElo")).otherwise(pl.col("awayPregameElo")),
            opp_elo=pl.when(pl.col("is_home")).then(pl.col("awayPregameElo")).otherwise(pl.col("homePregameElo")),
        ).drop([
            "homeTeam", "awayTeam", "homeClassification", "awayClassification",
            "homePoints", "awayPoints", "homePregameElo", "awayPregameElo",
        ])

    tg, rg = _with_game(tg), _with_game(rg)
    rg = rg.join(tg.select(["game_id", "team", "pass_att"]), on=["game_id", "team"], how="left")

    tg.write_parquet(TEAM_GAMES_PATH)
    rg.write_parquet(RECEIVER_GAMES_PATH)

    print(f"receiver-game rows: {rg.height:,} | team-games: {tg.height:,} ({tg['season'].min()}–{tg['season'].max()})")
    print("team name found in schedule:", round(float(tg["team_matched"].mean()), 4))
    print("team-games where receptions == completions:", round(float((tg["completions"] == tg["rec_total"]).mean()), 3))
    print(tg.group_by("season").agg(n=pl.len(), zero_pass_att=(pl.col("pass_att") == 0).sum()).sort("season"))

    if run_build_seasons:
        build_seasons()

    return tg, rg


if __name__ == "__main__":
    parse_boxes()