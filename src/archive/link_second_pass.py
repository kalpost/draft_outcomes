import re
import polars as pl
pl.Config.set_tbl_rows(80)

SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}
ALIAS = {"mississippi": "ole miss", "central florida": "ucf", "miami fl": "miami",
         "ala birmingham": "uab", "connecticut": "uconn", "hawaii": "hawai i",
         "north carolina state": "nc state", "middle tenn state": "middle tennessee",
         "la monroe": "ul monroe", "boston col": "boston college"}

def school(s):
    k = (s or "").lower().replace(".", "").replace("&", "and")
    k = re.sub(r"\bst\b", "state", k)
    k = re.sub(r"[^a-z0-9]+", " ", k).strip()
    return ALIAS.get(k, k)

def last(s):
    w = [x for x in re.sub(r"[^a-z ]", "", (s or "").lower().replace("-", " ")).split() if x not in SUFFIX]
    return w[-1] if w else ""

links = pl.read_parquet("data/processed/wr_links.parquet")
rs = (pl.read_parquet("data/processed/receiver_seasons.parquet")
      .select("season", "team", "athlete_id", "name", "games", "yds")
      .with_columns(last=pl.col("name").map_elements(last, return_dtype=pl.String),
                    cfb_sc=pl.col("team").map_elements(school, return_dtype=pl.String)))

un = (links.filter(pl.col("athlete_id").is_null())
      .with_columns(last=pl.col("pname").map_elements(last, return_dtype=pl.String)))
cand = (un.join(rs, on="last", how="inner")
        .filter(pl.col("season").is_between(pl.col("draft_year") - 5, pl.col("draft_year") - 1))
        .filter(pl.col("sc") == pl.col("cfb_sc")))
print("second-pass candidates (check each by eye):")
print(cand.select(["pname", "pcollege", "draft_year", "name", "team", "season", "yds"])
          .sort(["pname", "season"]))

m = links.filter(pl.col("athlete_id").is_not_null())
w = (m.join(rs.select("athlete_id", "season", "games"), on="athlete_id")
       .filter(pl.col("season").is_between(pl.col("draft_year") - 5, pl.col("draft_year") - 1))
       .group_by("wr_id").agg(seasons=pl.len(), games=pl.col("games").sum()))
print("college games per matched receiver:")
print(w["games"].describe())
print("matched receivers with under 10 college games:", (w["games"] < 10).sum())