import gzip, json
games = json.load(gzip.open("data/raw/cfbd/games_players/2023_w05.json.gz", "rt"))
g = games[0]
print("game keys:", list(g.keys()), "| teams:", len(g["teams"]))
t = g["teams"][0]
rec = next(c for c in t["categories"] if c["name"] == "receiving")
print("receiving types:", [x["name"] for x in rec["types"]])
print(json.dumps(rec["types"][0], indent=1)[:900])