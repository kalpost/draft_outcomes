"""
Unified NFL & Combine/Pro-Day Data Downloader (`pull_nfl.py`)
=============================================================
Consolidates `pull_nfl.py` and `download_combine_data.py`:
  1. Pulls NFL Draft Picks, Combine, Snap Counts, and Player Season Stats via
     `nflreadpy` across `[NFL_PULL_START_YEAR, COMPLETED_NFL_SEASON]`.
  2. Automatically backfills missing Combine workout numbers (`ht`, `wt`, `forty`,
     `vertical`, `broad_jump`, `cone`, `shuttle`) from the open-source Combine + Pro Day
     dataset (`combine_pro_day.parquet`).
  3. Writes raw files (`data/raw/{draft,combine,snaps,stats}.parquet`) AND syncs
     `data/processed/nfl_draft_picks.parquet` and `data/processed/nfl_stats_player.parquet`.
"""

from __future__ import annotations

import nflreadpy as nfl
import polars as pl

from config import (
    COMBINE_PRO_DAY_RAW_PATH,
    COMBINE_PRO_DAY_URL,
    COMBINE_RAW_PATH,
    COMPLETED_NFL_SEASON,
    DRAFT_RAW_PATH,
    NFL_DRAFT_PICKS_PATH,
    NFL_PULL_START_YEAR,
    NFL_STATS_PLAYER_PATH,
    SNAPS_RAW_PATH,
    STATS_RAW_PATH,
)
from download_combine_data import _fetch_url, _merge_pro_day_backfill


def _to_polars(df) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df)


def pull_nfl(include_pro_day: bool = True) -> None:
    stat_seasons = list(range(NFL_PULL_START_YEAR, COMPLETED_NFL_SEASON + 1))
    # Official NFL snap counts are only recorded from 2012 onwards
    snap_seasons = [s for s in stat_seasons if s >= 2012]

    print(f"Pulling nflverse player stats for seasons {stat_seasons[0]}–{stat_seasons[-1]}...")
    print(f"Pulling nflverse snap counts for seasons {snap_seasons[0]}–{snap_seasons[-1]} (2012+)...")

    draft = _to_polars(nfl.load_draft_picks())
    combine = _to_polars(nfl.load_combine())
    snaps = _to_polars(nfl.load_snap_counts(seasons=snap_seasons))
    stats = _to_polars(nfl.load_player_stats(seasons=stat_seasons))

    # Optional Pro Day backfill onto combine table
    if include_pro_day:
        print(f"Fetching Pro Day backfill dataset from {COMBINE_PRO_DAY_URL}...")
        if _fetch_url(COMBINE_PRO_DAY_URL, COMBINE_PRO_DAY_RAW_PATH):
            pd_df = pl.read_parquet(COMBINE_PRO_DAY_RAW_PATH)
            combine = _merge_pro_day_backfill(combine, pd_df)

    outputs = {
        "draft": (draft, DRAFT_RAW_PATH),
        "combine": (combine, COMBINE_RAW_PATH),
        "snaps": (snaps, SNAPS_RAW_PATH),
        "stats": (stats, STATS_RAW_PATH),
    }

    for name, (df, path) in outputs.items():
        print(f"{name:<8} shape={df.shape} -> {path}")
        df.write_parquet(path)

    # Keep processed NFL tables synced so `run_unified_pipeline.py` always reads fresh data
    draft.write_parquet(NFL_DRAFT_PICKS_PATH)
    stats.write_parquet(NFL_STATS_PLAYER_PATH)
    print(f"Synced processed tables: {NFL_DRAFT_PICKS_PATH.name}, {NFL_STATS_PLAYER_PATH.name}")


if __name__ == "__main__":
    pull_nfl()