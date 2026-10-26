import numpy as np, polars as pl
from scipy.stats import spearmanr

POWER = 1.5
TUNE_UP_TO = 2021
N_BOOT = 1000
pl.Config.set_tbl_rows(40)

res = pl.read_parquet("data/processed/transfer_preds.parquet")
tune = res.filter(pl.col("season") <= TUNE_UP_TO)
test = res.filter(pl.col("season") > TUNE_UP_TO)

def dev(yy, mu, power=POWER):
    mu = np.clip(mu, 1e-3, None)
    return 2 * (np.power(yy, 2 - power) / ((1 - power) * (2 - power))
                - yy * np.power(mu, 1 - power) / (1 - power) + np.power(mu, 2 - power) / (2 - power))

def arr(df, c):
    return df[c].to_numpy()

def wdev(df, mu):
    w = arr(df, "att").astype(float)
    return float(np.sum(w * dev(arr(df, "y"), mu)) / w.sum())

PRED = {"league mean": "league", "own record, last season": "own_last", "own record, career": "own_career",
        "M2 context only": "m2_context", "M2 player only": "m2_player", "M2 full": "m2_full"}
y, w = arr(test, "y"), arr(test, "att").astype(float)
lg = arr(test, "league")

print("1. LEVEL: attempt-weighted mean predicted rate vs actual (held-out)")
print(f"{'actual':28s} {np.sum(w * y) / w.sum():.4f}")
for name, col in PRED.items():
    print(f"{name:28s} {np.sum(w * arr(test, col)) / w.sum():.4f}")

print("\n2. SPREAD: prediction / league mean at the 5th, 50th and 95th percentiles")
print(f"{'actual y':28s}", np.round(np.percentile(y / lg, [5, 50, 95]), 2))
for name, col in PRED.items():
    print(f"{name:28s}", np.round(np.percentile(arr(test, col) / lg, [5, 50, 95]), 2))

print("\n3. CALIBRATION of M2 full, ten bins by predicted value: bin, n, mean predicted, mean actual")
mu = arr(test, "m2_full")
edges = np.quantile(mu, np.linspace(0, 1, 11))
bi = np.clip(np.searchsorted(edges, mu, side="right") - 1, 0, 9)
for i in range(10):
    m = bi == i
    print(i, int(m.sum()), round(float(np.sum(w[m] * mu[m]) / w[m].sum()), 4),
          round(float(np.sum(w[m] * y[m]) / w[m].sum()), 4))

print("\n4. TAIL: share of total deviance coming from the worst 5% of rows")
for name, col in (("own record, last season", "own_last"), ("M2 full", "m2_full")):
    c = np.sort(w * dev(y, arr(test, col)))[::-1]
    print(f"{name:28s} {c[:max(1, int(0.05 * len(c)))].sum() / c.sum():.3f}")

print("\n5. POST HOC, NOT CONFIRMATORY: calibrate M2, widen the baseline grid, both tuned on 2018-2021 only")
def cal(df, col, b):
    l = arr(df, "league")
    return l * np.power(np.clip(arr(df, col), 1e-3, None) / l, b)

bs = np.round(np.arange(0, 1.51, 0.1), 1)
best_b = {}
for col in ("m2_full", "m2_player"):
    sc = {b: wdev(tune, cal(tune, col, b)) for b in bs}
    best_b[col] = min(sc, key=sc.get)
    print(f"{col}: best exponent on tune {best_b[col]}",
          "(edge of grid)" if best_b[col] in (bs[0], bs[-1]) else "")

ks = (1000, 3000, 10000, 30000, 100000, 1000000)
def own(df, k):
    return (arr(df, "prev_yds") + k * arr(df, "league")) / (arr(df, "prev_att") + k)

sck = {k: wdev(tune, own(tune, k)) for k in ks}
kb = min(sck, key=sck.get)
print("own record last season: best k on tune", kb, "(edge of grid)" if kb in (ks[0], ks[-1]) else "")

cands = {"own record (widened k)": own(test, kb),
         "M2 full calibrated": cal(test, "m2_full", best_b["m2_full"]),
         "M2 player calibrated": cal(test, "m2_player", best_b["m2_player"])}
print(f"\n{'predictor (held-out)':28s} {'deviance':>9s} {'MAE':>8s} {'Spearman':>9s}")
for n_, m_ in cands.items():
    print(f"{n_:28s} {wdev(test, m_):9.4f} {np.sum(w * np.abs(y - m_)) / w.sum():8.4f} "
          f"{spearmanr(m_, y)[0]:9.3f}")

rng = np.random.default_rng(0)
base = cands["own record (widened k)"]
for n_ in ("M2 full calibrated", "M2 player calibrated"):
    la, lb = w * dev(y, base), w * dev(y, cands[n_])
    g = []
    for _ in range(N_BOOT):
        s = rng.integers(0, len(y), len(y))
        g.append(100 * (1 - lb[s].sum() / la[s].sum()))
    print(f"{n_} vs own record: deviance {100 * (1 - lb.sum() / la.sum()):+.2f}% "
          f"(95% {np.percentile(g, 2.5):+.2f} to {np.percentile(g, 97.5):+.2f})")