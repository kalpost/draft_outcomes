import os, json, time, requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
H = {"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"}
BASE = "https://api.collegefootballdata.com"
Path("data/raw").mkdir(parents=True, exist_ok=True)
samples, calls = {}, 0

def get(path, **params):
    global calls
    calls += 1
    t = time.time()
    r = requests.get(BASE + path, headers=H, params=params, timeout=60)
    lim = {k: v for k, v in r.headers.items() if "limit" in k.lower() or "remaining" in k.lower()}
    return r, time.time() - t, lim

def keys(x):
    if isinstance(x, dict): return list(x.keys())
    if isinstance(x, list) and x: return keys(x[0])
    return type(x).__name__

def probe(label, path, **params):
    r, dt, lim = get(path, **params)
    if not r.ok:
        print(f"{label:30s} {r.status_code} {r.text[:100]!r}")
        return None
    data = r.json()
    n = len(data) if isinstance(data, list) else 1
    txt = r.text.lower()
    samples[label] = data[0] if isinstance(data, list) and data else data
    print(f"{label:30s} 200 rows={n} bytes={len(r.content)} t={dt:.1f}s keys={keys(data)}"
          f" | target={'target' in txt} attempt={'attempt' in txt}")
    if lim: print("   limit headers:", lim)
    return data

Y, W = 2023, 5
print("--- game level, no team filter ---")
probe("games (schedule)", "/games", year=Y, week=W)
probe("games/players", "/games/players", year=Y, week=W)
probe("ppa/players/games", "/ppa/players/games", year=Y, week=W)
probe("games/teams", "/games/teams", year=Y, week=W)
probe("stats/game/advanced", "/stats/game/advanced", year=Y, week=W)
probe("plays/stats (one team)", "/plays/stats", year=Y, week=W, team="Ohio State")

print("--- ratings and adjusted ---")
probe("ratings/elo (week 5)", "/ratings/elo", year=Y, week=W)
probe("ratings/sp", "/ratings/sp", year=Y)
probe("ratings/fpi", "/ratings/fpi", year=Y)
probe("ratings/srs", "/ratings/srs", year=Y)
probe("wepa/team/season", "/wepa/team/season", year=Y)
probe("wepa/players/passing", "/wepa/players/passing", year=Y)
probe("wepa/players/receiving", "/wepa/players/receiving", year=Y)  # expect failure

print("--- context ---")
probe("player/usage", "/player/usage", year=Y, team="Ohio State")
probe("recruiting/players", "/recruiting/players", year=2020)
probe("talent", "/talent", year=Y)
probe("player/returning", "/player/returning", year=Y)
probe("player/portal", "/player/portal", year=Y)

print("--- Elo changes by week? (Georgia) ---")
for wk in (1, 2, 3, 4):
    r, _, _ = get("/ratings/elo", year=Y, week=wk, team="Georgia")
    print(wk, [d.get("elo") for d in r.json()] if r.ok else r.status_code)

print("--- SP+ and Elo history for one team ---")
for ep in ("/ratings/sp", "/ratings/elo"):
    r, _, _ = get(ep, team="Georgia")
    ys = sorted({d.get("year") for d in r.json()}) if r.ok else r.status_code
    print(ep, ys)

print("--- earliest year with game-level data (week 5) ---")
for yr in (2012, 2014, 2016, 2018, 2020):
    for ep in ("/games/players", "/ppa/players/games"):
        r, _, _ = get(ep, year=yr, week=W)
        print(yr, ep, len(r.json()) if r.ok else r.status_code)

json.dump(samples, open("data/raw/cfbd_probe_samples.json", "w"), indent=1, default=str)
print("calls used:", calls)