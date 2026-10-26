import gzip, json, collections as C

reg = json.load(gzip.open("data/raw/cfbd/games/2023_regular.json.gz", "rt"))
post = json.load(gzip.open("data/raw/cfbd/games/2023_post.json.gz", "rt"))
box = {1:178,2:122,3:120,4:120,5:115,6:102,7:109,8:108,9:113,10:124,11:124,12:125,13:75,14:19,15:4}

sched = C.Counter(g["week"] for g in reg)
print("week: schedule games vs box-score games")
for w in sorted(sched):
    print(w, sched[w], box.get(w))
print("postseason games by week:", sorted(C.Counter(g["week"] for g in post).items()))
print("home classification:", C.Counter(g["homeClassification"] for g in reg))
both = sum(1 for g in reg if g["homePregameElo"] is not None and g["awayPregameElo"] is not None)
print("games with pregame Elo for both teams:", both, "of", len(reg))