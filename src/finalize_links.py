import polars as pl
pl.Config.set_tbl_rows(40)

ACCEPT = {
    "Bub Means": ("Jerrod Means", "Pittsburgh"),
    "Devin Fuller": ("Devin Lewis Fuller", "UCLA"),
    "Gabe Davis": ("Gabriel Davis", "UCF"),
    "Mike Thomas": ("Michael Thomas", "Southern Miss"),
    "Pat Bryant": ("Patrick Bryant", "Illinois"),
    "Scotty Miller": ("Scott Miller", "Bowling Green"),
    "Shaquelle Evans": ("Shaq Evans", "UCLA"),
    "Tank Dell": ("Nathaniel Dell", "Houston"),
    "Tutu Atwell": ("Chatarius Atwell", "Louisville"),
    "Walt Powell": ("Walter Powell", "Murray State"),
}
links = pl.read_parquet("data/processed/wr_links.parquet")
rs = pl.read_parquet("data/processed/receiver_seasons.parquet").select("season", "team", "athlete_id", "name", "games", "yds")

fix = {}
for p, (cname, team) in ACCEPT.items():
    ids = rs.filter((pl.col("name") == cname) & (pl.col("team") == team))["athlete_id"].unique().to_list()
    if len(ids) == 1:
        fix[p] = ids[0]
    else:
        print("CHECK", p, cname, team, ids)

links = links.with_columns(
    manual=pl.col("pname").is_in(list(fix)) & pl.col("athlete_id").is_null(),
    athlete_id=pl.coalesce(pl.col("athlete_id"),
                           pl.col("pname").map_elements(lambda p: fix.get(p), return_dtype=pl.String)))

name_only = links.filter(pl.col("athlete_id").is_not_null() & ~pl.col("school_ok").fill_null(False) & ~pl.col("manual"))
print("name-only matches: which schools did the college stats come from?")
print(name_only.select("pname", "pcollege", "draft_year", "athlete_id")
      .join(rs.select("athlete_id", "season", "team", "yds"), on="athlete_id").sort(["pname", "season"]))

cov = (links.filter(pl.col("athlete_id").is_not_null())
       .join(rs.select("athlete_id", "season", "games"), on="athlete_id")
       .filter(pl.col("season").is_between(pl.col("draft_year") - 5, pl.col("draft_year") - 1))
       .group_by("wr_id").agg(college_games=pl.col("games").sum(), college_seasons=pl.len()))
final = (links.join(cov, on="wr_id", how="left")
         .with_columns(college_games=pl.col("college_games").fill_null(0),
                       thin=pl.col("college_games").fill_null(0) < 10))

linked = final.filter(pl.col("athlete_id").is_not_null())
print("linked:", linked.height, "of", final.height, "| thin (<10 games):", linked["thin"].sum())
print(linked.filter(pl.col("thin")).select("pname", "pcollege", "draft_year", "college_games"))
print("no link:")
print(final.filter(pl.col("athlete_id").is_null()).select("pname", "pcollege", "draft_year"))
final.write_parquet("data/processed/wr_links_final.parquet")