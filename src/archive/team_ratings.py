import numpy as np, polars as pl
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge

pl.Config.set_tbl_rows(20)
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
    return csr_matrix((np.ones(len(idx)), (np.arange(len(idx)), idx)), shape=(len(idx), len(cats))), list(cats)

Xo, off_names = onehot(d["off"].to_list())
Xd, dfn_names = onehot(d["dfn"].to_list())
home = csr_matrix(d["home"].to_numpy().reshape(-1, 1))
y = d["ypa"].to_numpy()
w = d["pass_att"].to_numpy().astype(float); w = w / w.mean()

rng = np.random.default_rng(0)
gids = d["game_id"].unique().to_list()
held = set(rng.choice(gids, size=int(0.15 * len(gids)), replace=False).tolist())
te = np.array([g in held for g in d["game_id"].to_list()]); tr = ~te

def wmse(p, mask):
    return float(np.sum(w[mask] * (y[mask] - p) ** 2) / np.sum(w[mask]))

print("rows:", len(y), "| held-out rows:", int(te.sum()))
print("global-mean baseline:", round(wmse(np.average(y[tr], weights=w[tr]), te), 3))
designs = {"offense only": hstack([Xo, home]).tocsr(), "offense + defense": hstack([Xo, Xd, home]).tocsr()}
res = {}
for name, X in designs.items():
    for a in (1, 3, 10, 30):
        m = Ridge(alpha=a).fit(X[tr], y[tr], sample_weight=w[tr])
        res[(name, a)] = wmse(m.predict(X[te]), te)
        print(f"{name:18s} alpha={a:<3d} held-out weighted MSE {res[(name, a)]:.3f}")

a_best = min(v for k, v in res.items() if k[0] == "offense only")
b_key = min((k for k in res if k[0] == "offense + defense"), key=res.get)
print("defense adds:", round(100 * (1 - res[b_key] / a_best), 1), "% lower held-out error")

m = Ridge(alpha=b_key[1]).fit(designs["offense + defense"], y, sample_weight=w)
no, nd = len(off_names), len(dfn_names)
ofn = pl.DataFrame({"team_season": off_names, "ypa_effect": m.coef_[:no]})
dfn = pl.DataFrame({"team_season": dfn_names, "ypa_allowed_effect": m.coef_[no:no + nd]})
ofn.write_parquet("data/processed/pass_offense_ratings.parquet")
dfn.write_parquet("data/processed/pass_defense_ratings.parquet")
print("best pass defenses, 2023 (lowest yards allowed):")
print(dfn.filter(pl.col("team_season").str.ends_with("|2023")).sort("ypa_allowed_effect").head(10))
print("best pass offenses, 2023:")
print(ofn.filter(pl.col("team_season").str.ends_with("|2023")).sort("ypa_effect", descending=True).head(10))