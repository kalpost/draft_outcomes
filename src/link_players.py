import re
import polars as pl
pl.Config.set_tbl_rows(80)

SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}
def nn(s):
    s = re.sub(r"[^a-z ]", "", (s or "").lower().replace("-", " ").replace(".", ""))
    return " ".join(w for w in s.split() if w not in SUFFIX)

ALIAS = {"mississippi": "ole miss", "central florida": "ucf", "miami fl": "miami",
         "ala birmingham": "uab", "connecticut": "uconn", "hawaii": "hawai i",
         "north carolina state": "nc state", "middle tenn state": "middle tennessee",
         "la monroe": "ul monroe", "boston col": "boston college"}
def school(s):
    k = (s or "").lower().replace(".", "").replace("&", "and")
    k = re.sub(r"\bst\b", "state", k)
    k = re.sub(r"[^a-z0-9]+", " ", k).strip()
    return ALIAS.get(k, k)

wr = (pl.read_parquet("data/raw/draft.parquet")
      .filter((pl.col("position") == "WR") & pl.col("season").is_between(2013, 2025))
      .select(pl.col("season").alias("draft_year"), "round", "pick", "gsis_id", "pfr_player_id",
              pl.col("pfr_player_name").alias("pname"), pl.col("college").alias("pcollege"))
      .with_row_index("wr_id")
      .with_columns(nm=pl.col("pname").map_elements(nn, return_dtype=pl.String),
                    sc=pl.col("pcollege").map_elements(school, return_dtype=pl.String)))

rs = (pl.read_parquet("data/processed/receiver_seasons.parquet")
      .select(pl.col("season").alias("cfb_season"), pl.col("team").alias("cfb_team"),
              "athlete_id", pl.col("name").alias("cfb_name"))
      .with_columns(nm=pl.col("cfb_name").map_elements(nn, return_dtype=pl.String),
                    cfb_sc=pl.col("cfb_team").map_elements(school, return_dtype=pl.String)))

cand = (wr.join(rs, on="nm", how="inner")
        .filter((pl.col("cfb_season") >= pl.col("draft_year") - 5) &
                (pl.col("cfb_season") <= pl.col("draft_year") - 1))
        .with_columns(school_ok=pl.col("sc") == pl.col("cfb_sc")))

byp = cand.group_by(["wr_id", "athlete_id"]).agg(n_seasons=pl.len(), school_ok=pl.col("school_ok").any())
best = (byp.sort(["wr_id", "school_ok", "n_seasons"], descending=[False, True, True])
        .group_by("wr_id").agg(athlete_id=pl.col("athlete_id").first(), school_ok=pl.col("school_ok").first(),
                               n_cands=pl.len(), n_school_ok=pl.col("school_ok").sum()))
res = wr.join(best, on="wr_id", how="left")

print("matched by name:", res["athlete_id"].is_not_null().sum(), "of", res.height)
print("matched with school agreement:", res["school_ok"].fill_null(False).sum())
print("ambiguous (several school-agreeing candidates):", (res["n_school_ok"].fill_null(0) > 1).sum())
print("name-only matches (school disagrees):")
print(res.filter(pl.col("athlete_id").is_not_null() & ~pl.col("school_ok")).select(["pname", "pcollege", "draft_year"]))
print("unmatched:")
print(res.filter(pl.col("athlete_id").is_null()).select(["pname", "pcollege", "draft_year", "round"]))
res.write_parquet("data/processed/wr_links.parquet")