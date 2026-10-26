import gzip, json, glob
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

rows = []
for f in sorted(glob.glob("data/raw/cfbd/games_players/*.json.gz")):
    for g in json.load(gzip.open(f, "rt")):
        for t in g["teams"]:
            att = {}
            for cat in t["categories"]:
                if cat["name"] != "passing":
                    continue
                for ty in cat["types"]:
                    if ty["name"] != "C/ATT":
                        continue
                    for a in ty["athletes"]:
                        try:
                            att[a["id"]] = int(a["stat"].split("/")[1])
                        except (ValueError, IndexError, AttributeError):
                            pass
            tot = sum(att.values())
            if tot > 0:
                qid = max(att, key=att.get)
                rows.append({"game_id": g["id"], "team": t["team"], "qb_id": qid})
qb = pl.DataFrame(rows).unique(subset=["game_id", "team"])

gm = pl.read_parquet("data/processed/games.parquet").select(
    pl.col("id").alias("game_id"), "season", "homeTeam", "homeClassification", "awayClassification")
qb = (qb.join(gm, on="game_id")
        .with_columns(own=pl.when(pl.col("team") == pl.col("homeTeam"))
                      .then(pl.col("homeClassification")).otherwise(pl.col("awayClassification")))
        .filter(pl.col("own") == "fbs"))
qb.write_parquet("data/processed/team_game_qb.parquet")

ts = (qb.group_by(["season", "team", "qb_id"]).agg(n=pl.len()).filter(pl.col("n") >= 3)
        .group_by(["season", "team"]).agg(k=pl.len()))
print("FBS team-seasons with a QB in 3+ games:", ts.height,
      "| share with 2+ primary QBs:", round((ts["k"] >= 2).mean(), 3))

rg = pl.read_parquet("data/processed/receiver_games.parquet").select("game_id", "team", "athlete_id")
pairs = rg.join(qb.select("game_id", "team", "qb_id"), on=["game_id", "team"]).filter(pl.col("athlete_id") != pl.col("qb_id"))
per = (pairs.group_by(["athlete_id", "qb_id"]).agg(n=pl.len()).filter(pl.col("n") >= 3)
       .group_by("athlete_id").agg(k=pl.len()))
links = pl.read_parquet("data/processed/wr_links_final.parquet").filter(pl.col("athlete_id").is_not_null())
coh = links.select("athlete_id").join(per, on="athlete_id", how="left").with_columns(pl.col("k").fill_null(0))
print("drafted WRs by number of distinct QBs (3+ games each):")
print(coh.group_by("k").len().sort("k"))

pr = pairs.select("athlete_id", "qb_id").unique()
ids = pl.concat([pr["athlete_id"], pr["qb_id"]]).unique().to_list()
idx = {v: i for i, v in enumerate(ids)}
r = [idx[a] for a in pr["athlete_id"]]
c = [idx[q] for q in pr["qb_id"]]
m = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(ids), len(ids)))
ncomp, lab = connected_components(m, directed=False)
sizes = np.bincount(lab)
giant = sizes.argmax()
print("people:", len(ids), "| components:", ncomp, "| giant component share:", round(sizes[giant] / len(ids), 3))
inc = [lab[idx[a]] == giant for a in links["athlete_id"].to_list() if a in idx]
print("drafted WRs in the giant component:", sum(inc), "of", len(inc))