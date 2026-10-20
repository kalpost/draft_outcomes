import gzip, json, glob, sys, collections as C

for y in (sys.argv[1:] or ["2023"]):
    reg = json.load(gzip.open(f"data/raw/cfbd/games/{y}_regular.json.gz", "rt"))
    post = json.load(gzip.open(f"data/raw/cfbd/games/{y}_post.json.gz", "rt"))
    sched = {g["id"]: g for g in reg + post}
    box = set()
    for f in glob.glob(f"data/raw/cfbd/games_players/{y}_*.json.gz"):
        box |= {g["id"] for g in json.load(gzip.open(f, "rt"))}
    fbs = {i for i, g in sched.items() if "fbs" in (g["homeClassification"], g["awayClassification"])}
    fcs = {i for i, g in sched.items() if g["homeClassification"] == "fcs" and g["awayClassification"] == "fcs"}
    print(y, "FBS games:", len(fbs), "with box:", len(fbs & box), "missing:", len(fbs - box),
          "| FCS-vs-FCS games:", len(fcs), "with box:", len(fcs & box))
    if fbs - box:
        print("   missing by week:", sorted(C.Counter(sched[i]["week"] for i in fbs - box).items()))