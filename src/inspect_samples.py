import json
s = json.load(open("data/raw/cfbd_probe_samples.json"))
t0 = s["games/players"]["teams"][0]
print("team keys:", list(t0.keys()))
for cat in t0["categories"]:
    print(cat["name"], [ty["name"] for ty in cat["types"]])
tt = s["games/teams"]["teams"][0]
print("team stat keys:", list(tt.keys()))
print([x.get("category") for x in tt["stats"]])