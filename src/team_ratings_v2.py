import numpy as np, polars as pl
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge

split = pl.read_parquet("data/processed/game_split.parquet").select("game_id", "team", "is_test")
tg = pl.read_parquet("data/processed/team_games.parquet")
rg = pl.read_parquet("data/processed/receiver_games.parquet")
gm = pl.read_parquet("data/processed/games.parquet").select(
    pl.col("id").alias("game_id"), "homeTeam", "homeClassification", "awayClassification")

yd = rg.group_by(["game_id", "team"]).agg(pass_yds=pl.col("yds").sum())
d = (tg.filter(pl.col("pass_att") >= 10)
       .join(split, on=["game_id", "team"], how="inner")
       .join(yd, on=["game_id", "team"], how="left")
       .join(gm, on="game_id", how="left")
       .with_columns(pl.col("pass_yds").fill_null(0))
       .with_columns(
           ypa=pl.col("pass_yds") / pl.col("pass_att"),
           off=pl.col("team") + "|" + pl.col("season").cast(pl.String),
           dfn=pl.col("opponent") + "|" + pl.col("season").cast(pl.String),
           home=pl.when(pl.col("neutralSite").fill_null(False)).then(0.0)
                 .when(pl.col("is_home")).then(1.0).otherwise(-1.0),
           own_class=pl.when(pl.col("team") == pl.col("homeTeam"))
                      .then(pl.col("homeClassification")).otherwise(pl.col("awayClassification"))))

# inner validation cutoff inside the training period, for choosing the penalty
ic = (d.filter(~pl.col("is_test")).group_by("season")
        .agg(icut=pl.col("week").quantile(2 / 3, interpolation="nearest")))
d = d.join(ic, on="season", how="left")

def onehot(vals):
    cats = {c: i for i, c in enumerate(sorted(set(vals)))}
    idx = np.array([cats[v] for v in vals])
    return csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), idx)), shape=(len(idx), len(cats))), list(cats)

Xo, off_names = onehot(d["off"].to_list())
Xd, dfn_names = onehot(d["dfn"].to_list())
home = csr_matrix(d["home"].to_numpy().reshape(-1, 1))
XA, XB = hstack([Xo, home]).tocsr(), hstack([Xo, Xd, home]).tocsr()
y = d["ypa"].to_numpy()
w = d["pass_att"].to_numpy().astype(float); w = w / w.mean()
te = d["is_test"].to_numpy(); tr = ~te
week, icut = d["week"].to_numpy(), d["icut"].to_numpy()
iv = tr & (week > icut); it = tr & ~iv

def wmse(p, mask):
    return float(np.sum(w[mask] * (y[mask] - p) ** 2) / np.sum(w[mask]))

best = {}
for name, X in (("offense only", XA), ("offense + defense", XB)):
    sc = {a: wmse(Ridge(alpha=a).fit(X[it], y[it], sample_weight=w[it]).predict(X[iv]), iv)
          for a in (1, 3, 10, 30)}
    best[name] = min(sc, key=sc.get)
    print(name, "validation MSE by alpha:", {a: round(v, 3) for a, v in sc.items()}, "-> alpha", best[name])

mA = Ridge(alpha=best["offense only"]).fit(XA[tr], y[tr], sample_weight=w[tr])
mB = Ridge(alpha=best["offense + defense"]).fit(XB[tr], y[tr], sample_weight=w[tr])
pA, pB = mA.predict(XA[te]), mB.predict(XB[te])
g = np.average(y[tr], weights=w[tr])
print("\ntraining rows:", int(tr.sum()), "| test rows:", int(te.sum()))
print("test MSE | global mean:", round(wmse(g, te), 3),
      "| offense only:", round(wmse(pA, te), 3), "| offense + defense:", round(wmse(pB, te), 3))

# bootstrap the defense gain over team-season clusters
yt, wt = y[te], w[te]
eA, eB = wt * (yt - pA) ** 2, wt * (yt - pB) ** 2
_, cl = np.unique(d.filter(pl.col("is_test"))["off"].to_numpy(), return_inverse=True)
nA, nB = np.bincount(cl, eA), np.bincount(cl, eB)
rng = np.random.default_rng(0)
gains = []
for _ in range(1000):
    s = rng.integers(0, len(nA), len(nA))
    gains.append(100 * (1 - nB[s].sum() / nA[s].sum()))
print("defense gain on later games: %.2f%%  (95%% interval %.2f to %.2f)"
      % (100 * (1 - eB.sum() / eA.sum()), np.percentile(gains, 2.5), np.percentile(gains, 97.5)))

fb = (d.filter(pl.col("is_test"))["own_class"] == "fbs").fill_null(False).to_numpy()
print("FBS teams only: %.2f%%" % (100 * (1 - eB[fb].sum() / eA[fb].sum())))

no, nd = len(off_names), len(dfn_names)
pl.DataFrame({"team_season": off_names, "ypa_effect": mB.coef_[:no]}).write_parquet("data/processed/pass_offense_train.parquet")
pl.DataFrame({"team_season": dfn_names, "ypa_allowed_effect": mB.coef_[no:no + nd]}).write_parquet("data/processed/pass_defense_train.parquet")