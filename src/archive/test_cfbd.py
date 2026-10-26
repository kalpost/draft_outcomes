import os, requests
from dotenv import load_dotenv

load_dotenv()
headers = {"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"}
base = "https://api.collegefootballdata.com"

tests = {
    "player season stats": ("/stats/player/season", {"year": 2023, "team": "Ohio State"}),
    "player PPA": ("/ppa/players/season", {"year": 2023, "team": "Ohio State"}),
    "advanced team stats": ("/stats/season/advanced", {"year": 2023, "team": "Ohio State"}),
}

for name, (path, params) in tests.items():
    r = requests.get(base + path, headers=headers, params=params)
    print(name, r.status_code, len(r.json()) if r.ok else r.text[:200])
    r = requests.get(base + "/stats/player/season", headers=headers, params={"year": 2023, "team": "Ohio State"})
    data = r.json()
    print(data[0])  # see the exact fields
    print(set(d.get("category") for d in data))  # stat categories

    r = requests.get(base + "/ppa/players/season", headers=headers, params={"year": 2013, "team": "Ohio State"})
    for d in r.json():
        print(d.get("name"), d.get("position"))  # who is covered

    for year in (2013, 2015):
        r = requests.get(base + "/ppa/players/season", headers=headers, params={"year": year, "team": "Ohio State"})
    print(year, len(r.json()) if r.ok else r.status_code)