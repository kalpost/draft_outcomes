import numpy as np, polars as pl
from scipy.sparse import csr_matrix, hstack
from scipy.stats import spearmanr
from sklearn.linear_model import TweedieRegressor

POWER = 1.5
ALPHA = 3e-5          # fixed in advance from the earlier validation. Not retuned here
SEASON_SCALE = 0.5
CTRL_SCALE = 10.0
FIRST_T = 2018        # first season at the new school that we predict
TUNE_UP_TO = 2021     # arrivals up to here tune the baseline's shrinkage. Later arrivals are the held-out test
K_GRID = (25, 100, 400, 1000, 3000)
N_BOOT = 1000
pl.Config.set_tbl_rows(40)

p = pl.read_parquet("data/processed/panel_v2.parquet")
off = pl.read_parquet("data/processed/pass_offense_train.parquet").rename({"team_season": "off_key"})
dfn = pl.read_parquet("data/processed/pass_defense_train.parquet").rename({"team_season": "def_key"})
gm = pl.read_parquet("data/processed/games.parquet").select(
    pl.col("id").alias("game_id"), "homeTeam", "homeClassification", "awayClassification")

p = p.with_columns(
    off_key=pl.col("team") + "|" + pl.col("season").cast(pl.String),
    def_key=pl.col("opponent") + "|" + pl.col("season").cast(pl.String),
    home=pl.when(pl.col("neutralSite").fill_null(False)).then(0.0)
          .when(pl.col("is_home")).then(1.0).otherwise(-1.0),
    ps=pl.col("athlete_id") + "|" + pl.col("season").cast(pl.String),
    ycl=pl.col("yds").clip(lower_bound=0))
p = p.join(off, on="off_key", how="left").join(dfn, on="def_key", how="left").join(gm, on="game_id", how="left")
p = p.with_columns(
    own_class=pl.when(pl.col("team") == pl.col("homeTeam"))
               .then(pl.col("homeClassification")).otherwise(pl.col("awayClassification")),
    ypa_effect=pl.col("ypa_effect").fill_null(0.0),
    ypa_allowed_effect=pl.col("ypa_allowed_effect").fill_null(0.0))

# one row per receiver-team-season, then pair each season with the previous one
rs = (p.group_by(["athlete_id", "season", "team"])
        .agg(name=pl.col("name").first(), yds=pl.col("ycl").sum(), att=pl.col("pass_att").sum(),
             cls=pl.col("own_class").first())
        .sort(["athlete_id", "season"])
        .with_columns(cum_yds=pl.col("yds").cum_sum().over("athlete_id"),
                      cum_att=pl.col("att").cum_sum().over("athlete_id")))
tf = (rs.with_columns(
        prev_season=pl.col("season").shift(1).over("athlete_id"),
        prev_team=pl.col("team").shift(1).over("athlete_id"),
        prev_cls=pl.col("cls").shift(1).over("athlete_id"),
        prev_yds=pl.col("yds").shift(1).over("athlete_id"),
        prev_att=pl.col("att").shift(1).over("athlete_id"),
        car_yds=pl.col("cum_yds").shift(1).over("athlete_id"),
        car_att=pl.col("cum_att").shift(1).over("athlete_id"))
      .filter((pl.col("prev_team").is_not_null())
              & (pl.col("team") != pl.col("prev_team"))
              & ((pl.col("season") - pl.col("prev_season")).is_between(1, 2))
              & (pl.col("season") >= FIRST_T) & (pl.col("season") <= 2024)))
tf = (tf.with_columns(
        new_key=pl.col("team") + "|" + (pl.col("season") - 1).cast(pl.String),
        prev_ps=pl.col("athlete_id") + "|" + pl.col("prev_season").cast(pl.String))
        .join(off.rename({"off_key": "new_key", "ypa_effect": "new_off"}), on="new_key", how="left"))
print("transfers found by season of arrival:")
print(tf.group_by("season").agg(found=pl.len(), no_context=pl.col("new_off").is_null().sum()).sort("season"))
tf = tf.filter(pl.col("new_off").is_not_null())
print("transfers usable:", tf.height)

def onehot(vals, cats):
    r, c = [], []
    for i, v in enumerate(vals):
        j = cats.get(v)
        if j is not None:
            r.append(i); c.append(j)
    return csr_matrix((np.ones(len(r)), (r, c)), shape=(len(vals), len(cats)))

pieces = []
for t in range(FIRST_T, 2025):
    pred = tf.filter(pl.col("season") == t)
    if pred.height == 0:
        continue
    train = p.filter(pl.col("season") < t)
    c_tr = np.column_stack([train["ypa_effect"].to_numpy(), train["ypa_allowed_effect"].to_numpy()])
    mu_c, sd_c = c_tr.mean(0), c_tr.std(0)
    Ctr = csr_matrix(np.column_stack([(c_tr - mu_c) / sd_c, train["home"].to_numpy()]) * CTRL_SCALE)
    pcats = {v: i for i, v in enumerate(train["athlete_id"].unique().to_list())}
    scats = {v: i for i, v in enumerate(train["ps"].unique().to_list())}
    Xtr = hstack([Ctr, onehot(train["athlete_id"].to_list(), pcats),
                  onehot(train["ps"].to_list(), scats) * SEASON_SCALE]).tocsr()
    y = (train["ycl"] / train["pass_att"]).to_numpy()
    w = train["pass_att"].to_numpy().astype(float)
    m = TweedieRegressor(power=POWER, link="log", alpha=ALPHA, max_iter=3000).fit(Xtr, y, sample_weight=w)
    n = pred.height
    P = onehot(pred["athlete_id"].to_list(), pcats)
    S = onehot(pred["prev_ps"].to_list(), scats) * SEASON_SCALE
    zoff = (pred["new_off"].to_numpy() - mu_c[0]) / sd_c[0]
    def mk(z, Pm, Sm):
        C = csr_matrix(np.column_stack([z, np.zeros(n), np.zeros(n)]) * CTRL_SCALE)
        return hstack([C, Pm, Sm]).tocsr()
    zP, zS = csr_matrix((n, P.shape[1])), csr_matrix((n, S.shape[1]))
    league = float(np.sum(w * y) / np.sum(w))
    print(f"arrivals {t}: {n} transfers | training rows {train.height} | iterations {m.n_iter_}")
    pieces.append(pred.select(["athlete_id", "name", "season", "prev_team", "team", "prev_cls", "cls",
                               "prev_yds", "prev_att", "car_yds", "car_att", "yds", "att"])
                  .with_columns(league=pl.lit(league),
                                m2_full=pl.Series(m.predict(mk(zoff, P, S))),
                                m2_player=pl.Series(m.predict(mk(np.zeros(n), P, S))),
                                m2_context=pl.Series(m.predict(mk(zoff, zP, zS)))))
res = pl.concat(pieces).with_columns(y=pl.col("yds") / pl.col("att"))

def dev(yy, mu, power=POWER):
    mu = np.clip(mu, 1e-3, None)
    return 2 * (np.power(yy, 2 - power) / ((1 - power) * (2 - power))
                - yy * np.power(mu, 1 - power) / (1 - power) + np.power(mu, 2 - power) / (2 - power))

def wdev(df, mu):
    w = df["att"].to_numpy().astype(float)
    return float(np.sum(w * dev(df["y"].to_numpy(), mu)) / w.sum())

tune, test = res.filter(pl.col("season") <= TUNE_UP_TO), res.filter(pl.col("season") > TUNE_UP_TO)
print("\ntune transfers:", tune.height, "| held-out transfers:", test.height)
if tune.height < 20:
    print("WARNING: very few tuning transfers. The baseline's shrinkage is poorly chosen")

def pick_k(ycol, acol):
    sc = {k: wdev(tune, ((tune[ycol] + k * tune["league"]) / (tune[acol] + k)).to_numpy()) for k in K_GRID}
    best = min(sc, key=sc.get)
    print(f"shrinkage for {ycol}: validation deviance", {k: round(v, 4) for k, v in sc.items()}, "-> k", best)
    return best

k_last, k_car = pick_k("prev_yds", "prev_att"), pick_k("car_yds", "car_att")
res = res.with_columns(
    own_last=(pl.col("prev_yds") + k_last * pl.col("league")) / (pl.col("prev_att") + k_last),
    own_career=(pl.col("car_yds") + k_car * pl.col("league")) / (pl.col("car_att") + k_car))
tune, test = res.filter(pl.col("season") <= TUNE_UP_TO), res.filter(pl.col("season") > TUNE_UP_TO)

PRED = {"league mean": "league", "own record, last season": "own_last", "own record, career": "own_career",
        "M2 new-team context only": "m2_context", "M2 player rating only": "m2_player", "M2 full": "m2_full"}

def report(df, title):
    y, w = df["y"].to_numpy(), df["att"].to_numpy().astype(float)
    print(f"\n{title}: {df.height} transfers")
    print(f"{'predictor':28s} {'deviance':>9s} {'MAE':>8s} {'Spearman':>9s}")
    for name, col in PRED.items():
        mu = df[col].to_numpy()
        print(f"{name:28s} {wdev(df, mu):9.4f} {np.sum(w * np.abs(y - mu)) / w.sum():8.4f} {spearmanr(mu, y)[0]:9.3f}")

def boot(df, a, b, label):
    y, w = df["y"].to_numpy(), df["att"].to_numpy().astype(float)
    ma, mb = df[a].to_numpy(), df[b].to_numpy()
    la, lb = w * dev(y, ma), w * dev(y, mb)
    rng, n = np.random.default_rng(0), len(y)
    g, sd = [], []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        g.append(100 * (1 - lb[s].sum() / la[s].sum()))
        sd.append(spearmanr(mb[s], y[s])[0] - spearmanr(ma[s], y[s])[0])
    print(f"{label}: deviance {100 * (1 - lb.sum() / la.sum()):+.2f}% "
          f"(95% {np.percentile(g, 2.5):+.2f} to {np.percentile(g, 97.5):+.2f}) | Spearman "
          f"{spearmanr(mb, y)[0] - spearmanr(ma, y)[0]:+.3f} "
          f"(95% {np.percentile(sd, 2.5):+.3f} to {np.percentile(sd, 97.5):+.3f})")

report(test, "HELD-OUT (first season at the new school 2022 or later)")
print("\nBOOTSTRAPPED OVER TRANSFERS. Positive = the second predictor is better")
boot(test, "own_last", "m2_full", "PRIMARY    M2 full vs own record (last season)")
boot(test, "own_last", "m2_player", "secondary  M2 player only vs own record")
boot(test, "m2_player", "m2_full", "secondary  M2 full vs M2 player only")
boot(test, "own_career", "m2_full", "secondary  M2 full vs own career record")

fcs_to_fbs = ((test["prev_cls"] != "fbs").fill_null(True) & (test["cls"] == "fbs")).to_numpy()
fbs_to_fbs = ((test["prev_cls"] == "fbs") & (test["cls"] == "fbs")).fill_null(False).to_numpy()
for lab, mask in (("FBS to FBS", fbs_to_fbs), ("from below FBS to FBS", fcs_to_fbs)):
    sub = test.filter(pl.Series(mask))
    if sub.height >= 30:
        boot(sub, "own_last", "m2_full", f"subgroup {lab} (n={sub.height})")
    else:
        print(f"subgroup {lab}: only {sub.height} transfers, skipped")

report(tune, "EXPLORATORY (first season at the new school 2018 to 2021)")
boot(tune, "own_last", "m2_full", "exploratory M2 full vs own record")
res.write_parquet("data/processed/transfer_preds.parquet")