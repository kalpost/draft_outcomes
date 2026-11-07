"""
CFBD Schedule & Box Score Downloader (`pull_cfbd.py`)
=====================================================
Pulls regular-season and postseason schedules (`/games`) and player game logs
(`/games/players`) from the CollegeFootballData API across `[CFB_START_YEAR, CFB_END_YEAR]`
(or specific CLI year arguments), caching compressed `.json.gz` files in `CFBD_RAW_DIR`.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from dotenv import load_dotenv
import requests

from config import (
    CFB_END_YEAR,
    CFB_START_YEAR,
    CFBD_API_RESERVE,
    CFBD_BASE_URL,
    CFBD_EXTRA_WEEKS,
    CFBD_RAW_DIR,
)

load_dotenv()
API_KEY = os.environ.get("CFBD_API_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "accept": "application/json",
}


def fetch(endpoint: str, name: str, **params) -> str:
    path = CFBD_RAW_DIR / endpoint.strip("/").replace("/", "_") / f"{name}.json.gz"
    # Don't treat empty/corrupted gz files as valid cache
    if path.exists() and path.stat().st_size > 50:
        return "cached"

    if not API_KEY:
        sys.exit("Error: CFBD_API_KEY environment variable not set.")

    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(CFBD_BASE_URL + endpoint, headers=HEADERS, params=params, timeout=120)
    left = int(r.headers.get("X-CallLimit-Remaining", -1))

    if not r.ok:
        print("ERROR", endpoint, params, r.status_code, r.text[:150])
        return "error"

    data = r.json()
    # Don't cache empty list responses for future/unplayed weeks
    if isinstance(data, list) and len(data) == 0:
        print(f"{endpoint} {name}: 0 rows returned (skipping cache)")
        return "empty"

    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"{endpoint} {name}: {len(data)} rows, {len(r.content)/1e6:.1f} MB, {left} calls left")
    if 0 <= left < CFBD_API_RESERVE:
        sys.exit(f"Stopping: only {left} calls left (reserve={CFBD_API_RESERVE})")
    return "ok"


def pull_year(year: int) -> None:
    print(f"\n--- Pulling {year} Season ---")
    # 1. Regular Season Player Game Logs (Weeks 1 to 15+)
    reg_weeks = list(range(1, 16)) + list(CFBD_EXTRA_WEEKS.get(year, ()))
    for wk in reg_weeks:
        fetch("/games/players", f"{year}_w{wk:02d}", year=year, week=wk, seasonType="regular")

    # 2. Postseason / Bowl / CFP Player Game Logs
    for pwk in range(1, 6):
        fetch("/games/players", f"{year}_post_w{pwk:02d}", year=year, week=pwk, seasonType="postseason")
    fetch("/games/players", f"{year}_postseason_all", year=year, seasonType="postseason")

    # 3. Team-Level Game Schedules & Results
    fetch("/games", f"{year}_regular", year=year, seasonType="regular")
    fetch("/games", f"{year}_post", year=year, seasonType="postseason")


def pull_cfbd(years: list[int] | None = None) -> None:
    target_years = years or list(range(CFB_START_YEAR, CFB_END_YEAR + 1))
    print(f"Pulling CFBD data for years: {target_years[0]}–{target_years[-1]} ({len(target_years)} seasons)")
    for y in target_years:
        pull_year(int(y))
    print("\nDone pulling CFBD data.")


if __name__ == "__main__":
    cli_years = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else None
    pull_cfbd(cli_years)