import gzip, json, glob, re, collections as C
import polars as pl

def norm(s):
    s = (s or "").lower().replace(".", "").replace("&", "and")
    s = re.sub(r"\bst\b", "state", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

ALIAS = {"mississippi": "ole miss", "central florida": "ucf", "miami fl": "miami",
         "ala birmingham": "uab", "connecticut": "uconn", "hawaii": "hawai i",
         "north carolina state": "nc state", "middle tenn state": "middle tennessee",
         "la monroe": "ul monroe", "boston col": "boston college"}

fbs = set()
for f in glob.glob("data/raw/cfbd/games/*_regular.json.gz"):
    for g in json.load(gzip.open(f, "rt")):
        if g["homeClassification"] == "fbs": fbs.add(norm(g["homeTeam"]))
        if g["awayClassification"] == "fbs": fbs.add(norm(g["awayTeam"]))

wr = pl.read_parquet("data/raw/draft.parquet").filter(
    (pl.col("position") == "WR") & pl.col("season").is_between(2013, 2025))
raw = wr["college"].to_list()
cols = [ALIAS.get(norm(r), norm(r)) for r in raw]
print("matched FBS:", sum(c in fbs for c in cols), "of", len(cols))
miss = C.Counter((c, r) for c, r in zip(cols, raw) if c not in fbs)
for (c, r), n in miss.most_common():
    print(n, repr(r))
print("FBS names containing key words:", sorted(n for n in fbs if any(k in n for k in ("miss", "ucf", "miami", "uab", "conn", "hawai", "nc state", "tennessee", "monroe", "boston"))))