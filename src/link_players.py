"""
Unified NFL-to-College WR Entity Linker (`link_players.py`)
===========================================================
Consolidates `link_players.py` and `finalize_links.py` into a single pass:
  1. Tier 1 (Exact Normalized Name + School Priority): Links NFL drafted WRs
     (`DRAFT_START_YEAR`–`DRAFT_END_YEAR`) to CFBD `receiver_seasons.parquet`
     within `[draft_year - CFB_LINK_LOOKBACK_YEARS, draft_year - 1]`.
  2. Tier 2 (Explicit Nickname Overrides): Applies `PLAYER_NAME_OVERRIDES` from
     `config.py` (e.g., Tank Dell -> Nathaniel Dell, Tutu Atwell -> Chatarius Atwell).
  3. Tier 3 (Automatic Same-School + Same-Surname Fallback): For any remaining
     unmatched draft pick, checks if there is a unique receiver at the exact
     same normalized college in the lookback window with the same last name and
     >= 250 career receiving yards (automatically catching future nickname cases).
  4. Attaches `college_games`, `college_seasons`, and `thin` flags and writes
     `wr_links.parquet` and `wr_links_final.parquet`.
"""

from __future__ import annotations

import polars as pl

from config import (
    CFB_LINK_LOOKBACK_YEARS,
    DRAFT_END_YEAR,
    DRAFT_RAW_PATH,
    DRAFT_START_YEAR,
    PLAYER_NAME_OVERRIDES,
    RECEIVER_SEASONS_PATH,
    THIN_COLLEGE_GAMES_THRESHOLD,
    WR_LINKS_FINAL_PATH,
    WR_LINKS_PATH,
    extract_last_name,
    norm_player_name,
    norm_school,
)

pl.Config.set_tbl_rows(80)


def link_players() -> pl.DataFrame:
    wr = (
        pl.read_parquet(DRAFT_RAW_PATH)
        .filter(
            (pl.col("position") == "WR")
            & pl.col("season").is_between(DRAFT_START_YEAR, DRAFT_END_YEAR)
        )
        .select(
            pl.col("season").alias("draft_year"),
            "round",
            "pick",
            "gsis_id",
            "pfr_player_id",
            pl.col("pfr_player_name").alias("pname"),
            pl.col("college").alias("pcollege"),
        )
        .with_row_index("wr_id")
        .with_columns(
            nm=pl.col("pname").map_elements(norm_player_name, return_dtype=pl.String),
            last_nm=pl.col("pname").map_elements(extract_last_name, return_dtype=pl.String),
            sc=pl.col("pcollege").map_elements(norm_school, return_dtype=pl.String),
        )
    )

    rs_full = pl.read_parquet(RECEIVER_SEASONS_PATH).with_columns(
        pl.col("athlete_id").cast(pl.String)
    )
    rs = (
        rs_full.select(
            pl.col("season").alias("cfb_season"),
            pl.col("team").alias("cfb_team"),
            "athlete_id",
            pl.col("name").alias("cfb_name"),
            "games",
            "yds",
        )
        .with_columns(
            nm=pl.col("cfb_name").map_elements(norm_player_name, return_dtype=pl.String),
            last_nm=pl.col("cfb_name").map_elements(extract_last_name, return_dtype=pl.String),
            cfb_sc=pl.col("cfb_team").map_elements(norm_school, return_dtype=pl.String),
        )
    )

    # --------------------------------------------------------------------------
    # Tier 1: Exact Normalized Name Join (prioritizing school agreement & seasons)
    # --------------------------------------------------------------------------
    cand = (
        wr.join(rs, on="nm", how="inner")
        .filter(
            (pl.col("cfb_season") >= pl.col("draft_year") - CFB_LINK_LOOKBACK_YEARS)
            & (pl.col("cfb_season") <= pl.col("draft_year") - 1)
        )
        .with_columns(school_ok=(pl.col("sc") == pl.col("cfb_sc")))
    )

    byp = cand.group_by(["wr_id", "athlete_id"]).agg(
        n_seasons=pl.len(),
        school_ok=pl.col("school_ok").any(),
    )
    best = (
        byp.sort(["wr_id", "school_ok", "n_seasons"], descending=[False, True, True])
        .group_by("wr_id")
        .agg(
            athlete_id=pl.col("athlete_id").first(),
            school_ok=pl.col("school_ok").first(),
            n_cands=pl.len(),
            n_school_ok=pl.col("school_ok").sum(),
        )
    )
    links = wr.join(best, on="wr_id", how="left")
    links.drop("last_nm").write_parquet(WR_LINKS_PATH)

    # --------------------------------------------------------------------------
    # Tier 2: Explicit Nickname / Legal-Name Overrides (`PLAYER_NAME_OVERRIDES`)
    # --------------------------------------------------------------------------
    manual_fix: dict[str, str] = {}
    for p_nfl, (c_name, c_team) in PLAYER_NAME_OVERRIDES.items():
        ids = (
            rs_full.filter((pl.col("name") == c_name) & (pl.col("team") == c_team))[
                "athlete_id"
            ]
            .unique()
            .to_list()
        )
        if len(ids) == 1:
            manual_fix[p_nfl] = str(ids[0])

    # --------------------------------------------------------------------------
    # Tier 3: Automatic Same-School + Same-Last-Name Fallback for Unmatched WRs
    # --------------------------------------------------------------------------
    unmatched = links.filter(
        pl.col("athlete_id").is_null() & ~pl.col("pname").is_in(list(manual_fix.keys()))
    )
    if unmatched.height > 0:
        surname_cand = (
            unmatched.select(["wr_id", "pname", "draft_year", "last_nm", "sc"])
            .join(rs, on=["last_nm"], how="inner")
            .filter(
                (pl.col("sc") == pl.col("cfb_sc"))
                & (pl.col("cfb_season") >= pl.col("draft_year") - CFB_LINK_LOOKBACK_YEARS)
                & (pl.col("cfb_season") <= pl.col("draft_year") - 1)
            )
            .group_by(["wr_id", "pname", "athlete_id", "cfb_name"])
            .agg(tot_yds=pl.col("yds").sum())
            .filter(pl.col("tot_yds") >= 250)
        )
        unique_surname = (
            surname_cand.group_by("wr_id")
            .agg(
                n_matches=pl.len(),
                pname=pl.col("pname").first(),
                athlete_id=pl.col("athlete_id").first(),
                cfb_name=pl.col("cfb_name").first(),
            )
            .filter(pl.col("n_matches") == 1)
        )
        for r in unique_surname.iter_rows(named=True):
            manual_fix[r["pname"]] = str(r["athlete_id"])
            print(f"Auto-linked nickname via same school + surname: {r['pname']} -> {r['cfb_name']} ({r['athlete_id']})")

    links = links.with_columns(
        manual=pl.col("pname").is_in(list(manual_fix.keys())) & pl.col("athlete_id").is_null(),
        athlete_id=pl.coalesce(
            pl.col("athlete_id"),
            pl.col("pname").map_elements(lambda p: manual_fix.get(p), return_dtype=pl.String),
        ),
    ).drop("last_nm")

    # --------------------------------------------------------------------------
    # Coverage & Final Link Table (`wr_links_final.parquet`)
    # --------------------------------------------------------------------------
    cov = (
        links.filter(pl.col("athlete_id").is_not_null())
        .join(rs_full.select("athlete_id", "season", "games"), on="athlete_id")
        .filter(
            pl.col("season").is_between(
                pl.col("draft_year") - CFB_LINK_LOOKBACK_YEARS,
                pl.col("draft_year") - 1,
            )
        )
        .group_by("wr_id")
        .agg(
            college_games=pl.col("games").sum(),
            college_seasons=pl.len(),
        )
    )

    final = (
        links.join(cov, on="wr_id", how="left")
        .with_columns(
            college_games=pl.col("college_games").fill_null(0),
            thin=pl.col("college_games").fill_null(0) < THIN_COLLEGE_GAMES_THRESHOLD,
        )
    )

    linked = final.filter(pl.col("athlete_id").is_not_null())
    print(
        f"Linked: {linked.height} of {final.height} ({DRAFT_START_YEAR}–{DRAFT_END_YEAR}) "
        f"| Thin (<{THIN_COLLEGE_GAMES_THRESHOLD} games): {linked['thin'].sum()} "
        f"| Unlinked: {final.height - linked.height}"
    )
    if final.height > linked.height:
        print("Unlinked prospects (primarily non-FBS/FCS or position converts):")
        print(final.filter(pl.col("athlete_id").is_null()).select("pname", "pcollege", "draft_year", "round"))

    final.write_parquet(WR_LINKS_FINAL_PATH)
    print(f"Successfully wrote {final.height} rows to {WR_LINKS_FINAL_PATH}")
    return final


if __name__ == "__main__":
    link_players()