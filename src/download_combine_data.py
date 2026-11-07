"""
Download and Coalesce NFL Combine & Pro Day Workout Data (`download_combine_data.py`)
=====================================================================================
1. Downloads official `nflverse` Combine data (`data/raw/combine.parquet`).
2. Attempts to download the coalesced Combine + Pro Day dataset
   (`array-carpenter/nfl-draft-data`: `combine_pro_day.parquet`, 2007–2026) and
   backfills missing workout measurables (`ht`, `wt`, `forty`, `vertical`,
   `broad_jump`, `cone`, `shuttle`) so prospects who skipped drills at Indianapolis
   and tested only at their college Pro Day are covered.
"""

from __future__ import annotations

import urllib.request
import polars as pl

from config import (
    COMBINE_PRO_DAY_RAW_PATH,
    COMBINE_PRO_DAY_URL,
    COMBINE_RAW_PATH,
    COMBINE_URL,
    DRAFT_END_YEAR,
    DRAFT_START_YEAR,
    PRO_DAY_FORTY_OFFSET_SEC,
    norm_player_key,
    parse_height_inches,
)


def _fetch_url(url: str, dest_path) -> bool:
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        print(f"Saved {dest_path.stat().st_size / 1024:.1f} KB -> {dest_path}")
        return True
    except Exception as exc:
        print(f"Warning: could not fetch {url} ({exc})")
        return False


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    col_map = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None


def _merge_pro_day_backfill(base_df: pl.DataFrame, pd_df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalizes the `combine_pro_day.parquet` schema and backfills any null workout
    drills on `base_df` (and appends WR prospects present only in the Pro Day table).
    """
    cols = pd_df.columns
    yr_c = _find_col(cols, ["Year", "draft_year", "season", "Draft Year"])
    nm_c = _find_col(cols, ["Player", "player_name", "Name", "name"])
    pos_c = _find_col(cols, ["POS_GP", "Pos", "pos", "position"])
    school_c = _find_col(cols, ["School", "College", "school", "college"])
    ht_c = _find_col(cols, ["Height", "ht", "HT", "height_in"])
    wt_c = _find_col(cols, ["Weight", "wt", "WT", "weight_lb"])
    forty_c = _find_col(cols, ["40 Yard", "forty", "40_yd", "40", "forty_yard"])
    vert_c = _find_col(cols, ["Vert Leap", "Vertical", "vertical", "vert"])
    broad_c = _find_col(cols, ["Broad Jump", "broad_jump", "broad", "Broad"])
    cone_c = _find_col(cols, ["3Cone", "Three Cone", "cone", "3-Cone"])
    shut_c = _find_col(cols, ["Shuttle", "shuttle", "20 Yard Shuttle"])

    if not yr_c or not nm_c:
        print("Pro Day dataset schema unrecognized; skipping backfill.")
        return base_df

    pd_records = []
    for r in pd_df.iter_rows(named=True):
        yr_val = r.get(yr_c)
        nm_val = r.get(nm_c)
        if yr_val is None or not nm_val:
            continue
        try:
            yr_int = int(float(yr_val))
        except Exception:
            continue

        key = f"{norm_player_key(str(nm_val))}_{yr_int}"
        ht_in = parse_height_inches(r.get(ht_c)) if ht_c else None

        def _flt(col_name: str | None) -> float | None:
            if not col_name or r.get(col_name) is None:
                return None
            try:
                val = float(r[col_name])
                return val if val > 0 else None
            except Exception:
                return None

        pd_records.append({
            "name_yr_key": key,
            "pd_player_name": str(nm_val),
            "pd_season": yr_int,
            "pd_pos": str(r.get(pos_c) or "") if pos_c else "",
            "pd_school": str(r.get(school_c) or "") if school_c else "",
            "pd_ht": f"{int(ht_in // 12)}-{round(ht_in % 12, 1):g}" if ht_in else None,
            "pd_wt": _flt(wt_c),
            "pd_forty": _flt(forty_c),
            "pd_vertical": _flt(vert_c),
            "pd_broad_jump": _flt(broad_c),
            "pd_cone": _flt(cone_c),
            "pd_shuttle": _flt(shut_c),
        })

    if not pd_records:
        return base_df

    pd_norm = pl.DataFrame(pd_records).unique(subset=["name_yr_key"])

    season_col = "season" if "season" in base_df.columns else "draft_year"
    base_keyed = base_df.with_columns(
        name_yr_key=(
            pl.col("player_name").map_elements(norm_player_key, return_dtype=pl.String)
            + "_"
            + pl.col(season_col).cast(pl.Int64).cast(pl.String)
        )
    )

    merged = base_keyed.join(pd_norm, on="name_yr_key", how="left").with_columns(
        ht=pl.coalesce(["ht", "pd_ht"]),
        wt=pl.coalesce(["wt", "pd_wt"]),
        # When official Combine 40 is missing, backfill with Pro Day 40 + PRO_DAY_FORTY_OFFSET_SEC
        forty=pl.when(pl.col("forty").is_not_null())
        .then(pl.col("forty"))
        .when(pl.col("pd_forty").is_not_null())
        .then((pl.col("pd_forty") + PRO_DAY_FORTY_OFFSET_SEC).round(2))
        .otherwise(None),
        vertical=pl.coalesce(["vertical", "pd_vertical"]),
        broad_jump=pl.coalesce(["broad_jump", "pd_broad_jump"]),
        cone=pl.coalesce(["cone", "pd_cone"]),
        shuttle=pl.coalesce(["shuttle", "pd_shuttle"]),
    )

    drop_cols = [c for c in merged.columns if c.startswith("pd_") or c == "name_yr_key"]
    return merged.drop(drop_cols)


def download_combine(include_pro_day: bool = True) -> pl.DataFrame:
    print(f"Fetching official nflverse combine parquet from {COMBINE_URL}...")
    if not _fetch_url(COMBINE_URL, COMBINE_RAW_PATH):
        raise RuntimeError(f"Failed to download {COMBINE_URL}")

    df = pl.read_parquet(COMBINE_RAW_PATH)
    season_col = "season" if "season" in df.columns else "draft_year"
    pos_col = "pos" if "pos" in df.columns else "position"

    wr_before = df.filter(
        pl.col(season_col).is_between(DRAFT_START_YEAR - 1, DRAFT_END_YEAR)
        & (pl.col(pos_col) == "WR")
    )
    forty_before = wr_before.filter(pl.col("forty").is_not_null()).height

    if include_pro_day:
        print(f"Fetching Pro Day backfill dataset from {COMBINE_PRO_DAY_URL}...")
        if _fetch_url(COMBINE_PRO_DAY_URL, COMBINE_PRO_DAY_RAW_PATH):
            pd_df = pl.read_parquet(COMBINE_PRO_DAY_RAW_PATH)
            df = _merge_pro_day_backfill(df, pd_df)
            df.write_parquet(COMBINE_RAW_PATH)

    wr_after = df.filter(
        pl.col(season_col).is_between(DRAFT_START_YEAR - 1, DRAFT_END_YEAR)
        & (pl.col(pos_col) == "WR")
    )
    forty_after = wr_after.filter(pl.col("forty").is_not_null()).height

    print(
        f"Total combine records: {len(df):,} | WR records ({DRAFT_START_YEAR - 1}–{DRAFT_END_YEAR}): {len(wr_after):,} "
        f"| WRs with 40 time: {forty_before} -> {forty_after} (+{forty_after - forty_before} Pro Day backfills)"
    )
    return df


if __name__ == "__main__":
    download_combine()