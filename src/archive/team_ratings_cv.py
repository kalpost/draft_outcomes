import numpy as np, polars as pl
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge

tg = pl.read_parquet("data/processed/team_games.parquet")
rg = pl.read_parquet("data/processed/receiver_games.parquet")
yd = rg.group_by(["game_id", "team"]).agg(pass_yds=pl.col("yds").sum())
d = (tg.join(yd, on=["game_id", "team"], how="left")
       .with_columns(pl.col("pass_yds").fill_null(0))
       .filter(pl.col("pass_att") >= 10)
       .with_columns(ypa=pl.col("pass_yds") / pl.col("pass_att"),
                     off=pl.col("team") + "|" + pl.col("season").cast(pl.String),
                     dfn=pl.col("opponent") + "|" + pl.col("season").cast(pl.String),
                     home=pl.when(pl.col("neutralSite").fill_null(False)).then(0.0)
                           .when(pl.col("is_home")).then(1.0).otherwise(-1.0)))

def onehot(vals):
    cats = {c: i for i, c in enumerate(sorted(set(vals)))}
    idx = np.array([cats[v] for v in vals])
    return csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), idx)), shape=(len(idx), len(cats)))

Xo, Xd = onehot(d["off"].to_list()), onehot(d["dfn"].to_list())
home = csr_matrix(d["home"].to_numpy().reshape(-1, 1))
XA, XB = hstack([Xo, home]).tocsr(), hstack([Xo, Xd, home]).tocsr()
y = d["ypa"].to_numpy()
w = d["pass_att"].to_numpy().astype(float); w = w / w.mean()
gid = np.array(d["game_id"].to_list())
games = np.unique(gid)

def wmse(p, mask):
    return float(np.sum(w[mask] * (y[mask] - p) ** 2) / np.sum(w[mask]))

gains = []
for seed in range(10):
    rng = np.random.default_rng(seed)
    held = set(rng.choice(games, size=int(0.15 * len(games)), replace=False).tolist())
    te = np.array([g in held for g in gid]); tr = ~te
    a = wmse(Ridge(alpha=3).fit(XA[tr], y[tr], sample_weight=w[tr]).predict(XA[te]), te)
    b = wmse(Ridge(alpha=3).fit(XB[tr], y[tr], sample_weight=w[tr]).predict(XB[te]), te)
    gains.append(100 * (1 - b / a))
    print(seed, round(gains[-1], 2))
print("defense gain %: mean", round(np.mean(gains), 2), "| min", round(min(gains), 2), "| max", round(max(gains), 2))