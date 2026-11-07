"""Shared data locations and the live class window.

College box scores run through LAST_CFB_SEASON (2024).
Drafted WR classes run through LAST_DRAFT_YEAR (2025 NFL draft).
"""
from __future__ import annotations

from pathlib import Path

FIRST_CFB_SEASON = 2010
LAST_CFB_SEASON = 2024
FIRST_DRAFT_YEAR = 2013
LAST_DRAFT_YEAR = 2025
# Last class with three full NFL seasons in nfl_stats_player (stats through 2024).
LAST_COMPLETE_CLASS = 2022


def resolve_data_dir() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here / "data", here.parent / "data"):
        if cand.is_dir():
            return cand.resolve()
    cand = here.parent / "data"
    cand.mkdir(parents=True, exist_ok=True)
    return cand.resolve()


DATA_DIR = resolve_data_dir()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
