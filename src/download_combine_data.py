"""
Download and verify nflverse combine & pro day dataset.

Writes:
    data/raw/combine.parquet
"""

import sys
import urllib.request
from pathlib import Path
import polars as pl

COMBINE_URL = "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.parquet"
OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "combine.parquet"

def download_combine():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching combine parquet from {COMBINE_URL}...")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(COMBINE_URL, headers=headers)
    with urllib.request.urlopen(req) as resp, open(OUTPUT_FILE, "wb") as f:
        f.write(resp.read())
        
    print(f"Saved {OUTPUT_FILE.stat().st_size / 1024:.1f} KB to {OUTPUT_FILE}")
    
    # Audit WR coverage for 2012-2022
    df = pl.read_parquet(OUTPUT_FILE)
    print(f"Total combine records loaded: {len(df)}")
    
    # Identify season column
    season_col = "season" if "season" in df.columns else "draft_year"
    pos_col = "pos" if "pos" in df.columns else "position"
    
    wr_df = df.filter(
        (pl.col(season_col) >= 2012) & 
        (pl.col(season_col) <= 2022) & 
        (pl.col(pos_col) == "WR")
    )
    print(f"Found {len(wr_df)} wide receiver workout records for 2012-2022.")

if __name__ == "__main__":
    download_combine()