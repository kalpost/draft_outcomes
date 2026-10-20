import gzip, json, os, sys, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
H = {"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"}
BASE = "https://api.collegefootballdata.com"
OUT = Path("data/raw/cfbd")
RESERVE = 150  # stop if fewer calls than this remain

def fetch(endpoint, name, **params):
    path = OUT / endpoint.strip("/").replace("/", "_") / f"{name}.json.gz"
    if path.exists():
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(BASE + endpoint, headers=H, params=params, timeout=120)
    left = int(r.headers.get("X-CallLimit-Remaining", -1))
    if not r.ok:
        print("ERROR", endpoint, params, r.status_code, r.text[:150])
        return "error"
    with gzip.open(path, "wt") as f:
        f.write(r.text)
    print(f"{endpoint} {name}: {len(r.json())} rows, {len(r.content)/1e6:.1f} MB, {left} calls left")
    if 0 <= left < RESERVE:
        sys.exit(f"Stopping: only {left} calls left")
    return "ok"

EXTRA_WEEKS = {2020: (16, 17)}

def pull_year(year):
    for wk in list(range(1, 16)) + list(EXTRA_WEEKS.get(year, ())):
        fetch("/games/players", f"{year}_w{wk:02d}", year=year, week=wk)
    fetch("/games", f"{year}_regular", year=year, seasonType="regular")
    fetch("/games", f"{year}_post", year=year, seasonType="postseason")
    
if __name__ == "__main__":
    years = sys.argv[1:] or ["2023"]
    print("pulling years:", years)
    for y in years:
        pull_year(int(y))
    print("done")