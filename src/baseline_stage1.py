import numpy as np, polars as pl, statsmodels.api as sm

rk = pl.read_parquet("data/processed/rookie_outcomes.parquet").filter(pl.col("season") <= 2023)
season = rk["season"].to_numpy()
y = np.log1p(rk["ypg"].to_numpy())
X = sm.add_constant(np.log(rk["pick"].to_numpy().astype(float)))
taus = [0.1, 0.25, 0.5, 0.75, 0.9]

def pinball(y, q, t):
    d = y - q
    return np.mean(np.maximum(t * d, (t - 1) * d))

ys, slot_q, clim_q = [], [], []
for ty in range(2018, 2024):            # rolling origin: train on earlier classes only
    tr, te = season < ty, season == ty
    s = [sm.QuantReg(y[tr], X[tr]).fit(q=t, max_iter=5000).predict(X[te]) for t in taus]
    c = [np.full(te.sum(), np.quantile(y[tr], t)) for t in taus]
    slot_q.append(np.sort(np.column_stack(s), axis=1))
    clim_q.append(np.sort(np.column_stack(c), axis=1))
    ys.append(y[te])

Y, S, C = np.concatenate(ys), np.vstack(slot_q), np.vstack(clim_q)
print("test rows:", len(Y))
for name, Q in (("climatology", C), ("draft slot", S)):
    loss = np.mean([pinball(Y, Q[:, i], t) for i, t in enumerate(taus)])
    cov = np.mean((Y >= Q[:, 0]) & (Y <= Q[:, -1]))
    print(f"{name:12s} mean pinball {loss:.4f} | 10-90 coverage {cov:.2f} | mean width {np.mean(Q[:, -1] - Q[:, 0]):.2f}")