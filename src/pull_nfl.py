import nflreadpy as nfl
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)

draft = nfl.load_draft_picks()
combine = nfl.load_combine()
snaps = nfl.load_snap_counts(seasons=list(range(2012, 2026)))
stats = nfl.load_player_stats(seasons=list(range(2012, 2026)))

for name, df in {"draft": draft, "combine": combine, "snaps": snaps, "stats": stats}.items():
    print(name, df.shape)
    print(df.columns)
    df.write_parquet(f"data/raw/{name}.parquet")