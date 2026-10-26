import numpy as np, polars as pl
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge, TweedieRegressor

QUICK = True                          # True: seasons 2019+ only, to test the pipeline. Then set False
POWER = 1.5
ALPHAS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4)    # Tweedie penalties, set in advance
RIDGE_ALPHAS = (3, 10, 30, 100)
K_GRID = (0, 25, 50, 100, 200, 400)   # shrinkage strengths for the own-average baseline
SEASON_SCALE = 0.5                    # season deviations penalized 4x more than career effects
CTRL_SCALE = 10.0                     # controls nearly unpenalized

p = pl.read_parquet("data/processed/panel_v2.parquet")
if QUICK:
    p = p.filter(pl.col("season") >= 2019)
off = pl.read_parquet("data/processed/pass_offense_train.parquet").rename({"team_season": "off_key"})
dfn = pl.read_parquet("data/processed/pass_defense_train.parquet").rename({"team_season": "def_key"})
qb = pl.read_parquet("data/processed/team_game_qb.parquet").select("game_id", "team", "qb_id")
gm = pl.read_parquet("data/processed/games.parquet").select(
    pl.col("id").alias("game_id"), "homeTeam", "homeClassification", "awayClassification")

p = p.with_columns(
    off_key=pl.col("team") + "|" + pl.col("season").cast(pl.String),
    def_key=pl.col("opponent") + "|" + pl.col("season").cast(pl.String),
    home=pl.when(pl.col("neutralSite").fill_null(False)).then(0.0)
          .when(pl.col("is_home")).then(1.0).otherwise(-1.0),
    ps=pl.col("athlete_id") + "|" + pl.col("season").cast(pl.String))
p = (p.join(off, on="off_key", how="left").join(dfn, on="def_key", how="left")
      .join(qb, on=["game_id", "team"], how="left").join(gm, on="game_id", how="left"))
p = p.with_columns(
    qbk=pl.when(pl.col("qb_id").is_not_null()).then(pl.col("qb_id") + "|" + pl.col("season").cast(pl.String)),
    ycl=pl.col("yds").clip(lower_bound=0),
    own_class=pl.when(pl.col("team") == pl.col("homeTeam"))
               .then(pl.col("homeClassification")).otherwise(pl.col("awayClassification")))
p = p.with_columns(pl.col("ypa_effect").fill_null(0.0), pl.col("ypa_allowed_effect").fill_null(0.0))
p = p.with_columns(y=pl.col("ycl") / pl.col("pass_att"))   # negative-yardage games clipped to 0

y = p["y"].to_numpy(); w = p["pass_att"].to_numpy().astype(float)
te = p["is_test"].to_numpy(); tr = ~te
week = p["week"].to_numpy()
fbs = (p["own_class"] == "fbs").fill_null(False).to_numpy()

# inner validation block inside the training period (used only to choose penalties)
ic = (p.select("season").join(
        p.filter(~pl.col("is_test")).group_by("season")
         .agg(ic=pl.col("week").quantile(2 / 3, interpolation="nearest")),
        on="season", how="left")["ic"].to_numpy())
iv = tr & (week > ic); it = tr & ~iv
# receivers qualify for validation using INNER-training catches only, mirroring the real test setup
qi = (p.filter(pl.Series(it)).group_by("ps")
        .agg(r=pl.col("rec").sum()).filter(pl.col("r") >= 5)["ps"])
qmask = p["ps"].is_in(qi).to_numpy()
it, iv = it & qmask, iv & qmask
print("rows:", p.height, "| train:", int(tr.sum()), "| test:", int(te.sum()),
      "| inner-train/val:", int(it.sum()), int(iv.sum()), "| receivers:", p["athlete_id"].n_unique())

def z(a):
    return (a - a[tr].mean()) / a[tr].std()

C = csr_matrix(np.column_stack([z(p["ypa_effect"].to_numpy()), z(p["ypa_allowed_effect"].to_numpy()),
                                p["home"].to_numpy()]) * CTRL_SCALE)

def onehot(vals):
    cats, r, c = {}, [], []
    for i, v in enumerate(vals):
        if v is None:
            continue
        r.append(i); c.append(cats.setdefault(v, len(cats)))
    return csr_matrix((np.ones(len(r)), (r, c)), shape=(len(vals), len(cats)))

Pm = onehot(p["athlete_id"].to_list())
Sm = onehot(p["ps"].to_list()) * SEASON_SCALE
Qm = onehot(p["qbk"].to_list())
X1 = hstack([C, Pm, Sm]).tocsr()
X2 = hstack([C, Pm, Sm, Qm]).tocsr()
print("columns:", X1.shape[1], "(no QB) |", X2.shape[1], "(with QB)")

def dev(yy, mu, power=POWER):
    mu = np.clip(mu, 1e-3, None)
    return 2 * (np.power(yy, 2 - power) / ((1 - power) * (2 - power))
                - yy * np.power(mu, 1 - power) / (1 - power) + np.power(mu, 2 - power) / (2 - power))

def wmean(v, mask):
    return float(np.sum(w[mask] * v) / np.sum(w[mask]))

def own_avg(fit_mask, pred_mask, k):
    d = p.filter(pl.Series(fit_mask)).group_by("ps").agg(s=pl.col("ycl").sum(), a=pl.col("pass_att").sum())
    m0 = d["s"].sum() / d["a"].sum()
    t = p.filter(pl.Series(pred_mask)).select("ps").join(d, on="ps", how="left")
    s = t["s"].fill_null(0).to_numpy(); a = t["a"].fill_null(0).to_numpy()
    return (s + k * m0) / np.maximum(a + k, 1e-9)

best_k = min(K_GRID, key=lambda k: wmean(dev(y[iv], own_avg(it, iv, k)), iv))
print("own-average shrinkage k chosen on validation:", best_k)

def tune_glm(X, name):
    sc = {}
    for a in ALPHAS:
        m = TweedieRegressor(power=POWER, link="log", alpha=a, max_iter=3000).fit(X[it], y[it], sample_weight=w[it])
        sc[a] = wmean(dev(y[iv], m.predict(X[iv])), iv)
        print(f"  {name} alpha={a}: validation deviance {sc[a]:.4f}, iterations {m.n_iter_}")
    best = min(sc, key=sc.get)
    print(name, "-> alpha", best)
    m = TweedieRegressor(power=POWER, link="log", alpha=best, max_iter=3000).fit(X[tr], y[tr], sample_weight=w[tr])
    print(f"  {name} final fit iterations: {m.n_iter_}")
    return m.predict(X[te])

wn = w / w[tr].mean()
def tune_ridge(X, name):
    sc = {}
    for a in RIDGE_ALPHAS:
        m = Ridge(alpha=a).fit(X[it], y[it], sample_weight=wn[it])
        sc[a] = wmean(dev(y[iv], m.predict(X[iv])), iv)
    best = min(sc, key=sc.get)
    print(name, "validation deviance by alpha:", {a: round(v, 4) for a, v in sc.items()}, "-> alpha", best)
    m = Ridge(alpha=best).fit(X[tr], y[tr], sample_weight=wn[tr])
    return np.clip(m.predict(X[te]), 1e-3, None)

m0 = float(np.sum(w[tr] * y[tr]) / np.sum(w[tr]))
preds = {
    "global mean": np.full(int(te.sum()), m0),
    "own average (raw)": own_avg(tr, te, 0),
    f"M0b own average shrunk (k={best_k})": own_avg(tr, te, best_k),
    "M0 Gaussian ridge": tune_ridge(X1, "M0 ridge"),
    "M2 Tweedie": tune_glm(X1, "M2"),
    "M2 Tweedie + QB": tune_glm(X2, "M2+QB"),
}

wt, yt = w[te], y[te]
fbt = fbs[te]
print("\nHELD-OUT RESULTS (lower is better)")
print(f"{'model':38s} {'deviance':>9s} {'MAE':>8s} {'FBS dev':>9s}")
for name, mu in preds.items():
    d_ = dev(yt, mu)
    print(f"{name:38s} {np.sum(wt * d_) / wt.sum():9.4f} {np.sum(wt * np.abs(yt - mu)) / wt.sum():8.4f} "
          f"{np.sum(wt[fbt] * d_[fbt]) / wt[fbt].sum():9.4f}")

_, cid = np.unique(p.filter(pl.col("is_test"))["ps"].to_numpy(), return_inverse=True)
def boot(a, b, label):
    la, lb = wt * dev(yt, a), wt * dev(yt, b)
    na, nb = np.bincount(cid, la), np.bincount(cid, lb)
    rng = np.random.default_rng(0); g = []
    for _ in range(1000):
        s = rng.integers(0, len(na), len(na)); g.append(100 * (1 - nb[s].sum() / na[s].sum()))
    print(f"{label}: {100 * (1 - lb.sum() / la.sum()):.2f}% lower deviance "
          f"(95% interval {np.percentile(g, 2.5):.2f} to {np.percentile(g, 97.5):.2f})")

print("\nIMPROVEMENT, bootstrapped over receiver-seasons")
boot(preds[f"M0b own average shrunk (k={best_k})"], preds["M2 Tweedie"], "M2 vs shrunk own average")
boot(preds["M0 Gaussian ridge"], preds["M2 Tweedie"], "M2 vs Gaussian ridge")
boot(preds["M2 Tweedie"], preds["M2 Tweedie + QB"], "M2+QB vs M2")

out = p.filter(pl.col("is_test")).select("game_id", "team", "athlete_id", "name", "season", "week", "y", "pass_att")
out = out.with_columns([pl.Series(n, mu) for n, mu in preds.items()])
out.write_parquet("data/processed/model_v1_test_preds.parquet")