import gzip, json, glob, collections as C

reg = json.load(gzip.open("data/raw/cfbd/games/2023_regular.json.gz", "rt"))
post = json.load(gzip.open("data/raw/cfbd/games/2023_post.json.gz", "rt"))
sched = {g["id"]: g for g in reg + post}

all_box = []
for f in sorted(glob.glob("data/raw/cfbd/games_players/2023_*.json.gz")):
    all_box += [g["id"] for g in json.load(gzip.open(f, "rt"))]
box = set(all_box)

print("box games:", len(all_box), "| unique:", len(box))
print("box ids not in schedule:", sum(1 for i in box if i not in sched))
print("box games by season type:", C.Counter(sched[i]["seasonType"] for i in box if i in sched))
print(C.Counter((sched[i]["homeClassification"], sched[i]["awayClassification"])
                for i in box if i in sched).most_common(8))

fbs = {i for i, g in sched.items() if "fbs" in (g["homeClassification"], g["awayClassification"])}
print("schedule games with an FBS team:", len(fbs), "| with a box score:", len(fbs & box))