"""
Step 3: fit M0 (Gaussian ridge), M1 (Poisson receptions), M2 (Tweedie yards) with career and
season-deviation player effects.  Train on training games only, score on the held-out last third,
bootstrap the differences over receiver-seasons.

Usage (from project root):
    python src/03_fit_m0_m1_m2.py [data_dir] [panel_file]
    defaults: data/processed  panel_v2_feat.parquet

Model (log link for M1, M2):
    log E[rate] = b0 + b1*log(off_oth_ypa) + b2*def_rating + b3*home + b4*neutral
                  + career_effect[receiver] + season_deviation[receiver-season]
    rate = receptions (M1) or yards (M2) per team pass attempt; rows weighted by team pass attempts.
No quarterback term (dropped by the pre-specified QB rule).

Penalties.  Each group of columns gets its own ridge strength, set by scaling the columns.  Strength is
in "pseudo-attempts": a career effect with kappa=400 is shrunk as if the receiver had 400 extra
attempts of average production.  Career and season-deviation strengths are chosen on a time-ordered
validation block INSIDE the training period (last third of each season's training weeks), never on test.
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.linear_model import PoissonRegressor, Ridge, TweedieRegressor

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")
PANEL = sys.argv[2] if len(sys.argv) > 2 else "panel_v2_feat.parquet"

POWER = 1.5                   # Tweedie power, fixed in advance
FLOOR = 0.01                  # floor on predicted rate so deviance is finite for every model
KAPPA_REF = 200.0              # reference penalty that fixes alpha for the GLMs (see fit_predict)
KAPPA_CTX = 10.0              # context coefficients: nearly unpenalized
KAPPA_CAREER = (10.0, 30.0, 100.0, 300.0, 1000.0)
KAPPA_SEASON = (10.0, 30.0, 100.0, 300.0, 1000.0)
N_BOOT = 1000
CTX = ["lo", "ld", "home", "neutral"]


# ---------------------------------------------------------------- losses (per row)
def tweedie_dev(y, mu, p=POWER):
    mu = np.maximum(mu, FLOOR)
    t1 = np.where(y > 0, np.power(np.maximum(y, 1e-12), 2 - p) / ((1 - p) * (2 - p)), 0.0)
    return 2 * (t1 - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p))


def poisson_dev(y, mu):
    mu = np.maximum(mu, FLOOR)
    t = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0)
    return 2 * (t - (y - mu))


def sq_err(y, mu):
    return (y - mu) ** 2


# ---------------------------------------------------------------- design matrix
def ctx_raw(df):
    home = (df["is_home"].fill_null(False) & ~df["neutralSite"].fill_null(False)).cast(pl.Float64)
    neutral = df["neutralSite"].fill_null(False).cast(pl.Float64)
    return np.column_stack([
        np.log(df["off_oth_ypa"].to_numpy()),
        df["def_rating"].to_numpy(),
        home.to_numpy(), neutral.to_numpy()])


def make_maps(df):
    return ({a: i for i, a in enumerate(df["athlete_id"].unique().to_list())},
            {c: i for i, c in enumerate(df["cl_rec"].unique().to_list())})


def make_X(df, maps, norm, s_ctx, s_c, s_d, players=True, keep=None):
    n = df.height
    keep = list(range(len(CTX))) if keep is None else keep
    cols = [sp.csr_matrix((ctx_raw(df)[:, keep] - norm[0]) / norm[1] * s_ctx)]
    if players:
        for col, mp, s in (("athlete_id", maps[0], s_c), ("cl_rec", maps[1], s_d)):
            codes = np.array([mp.get(v, -1) for v in df[col].to_list()])
            ok = np.nonzero(codes >= 0)[0]
            cols.append(sp.csr_matrix((np.full(ok.size, s), (ok, codes[ok])), shape=(n, len(mp))))
    return sp.hstack(cols).tocsr()


def fit_predict(kind, tr, te, kc, kd, players=True, drop=()):
    """Fit on tr, predict te.  kind in gauss / pois / tweedie."""
    maps = make_maps(tr)
    keep = [i for i, nm in enumerate(CTX) if nm not in drop]
    raw = ctx_raw(tr)[:, keep]
    norm = (raw.mean(0), raw.std(0) + 1e-9)
    w = tr["pass_att"].to_numpy().astype(float)
    sw = w.sum()
    if kind == "gauss":                 # Ridge objective: sum w r^2 + alpha ||b||^2 -> scale = 1/sqrt(kappa)
        sc = lambda k: 1.0 / np.sqrt(k)
        y = tr["y"].to_numpy()
        model = Ridge(alpha=1.0)
    else:                               # GLM objective: (1/2sw) sum w dev + (alpha/2)||b||^2
        # penalty on a coefficient is alpha*sw/s^2 = kappa.  Fix alpha = KAPPA_REF/sw so every column scale
        # s = sqrt(KAPPA_REF/kappa) stays near 1.  (Scales of several hundred overflow exp() on the first
        # optimizer step and abort the fit at iteration 0.)
        alpha = KAPPA_REF / sw
        sc = lambda k: np.sqrt(KAPPA_REF / k)
        if kind == "pois":
            y = tr["rec_rate"].to_numpy()
            model = PoissonRegressor(alpha=alpha, max_iter=3000, tol=1e-6)
        else:
            y = tr["y"].to_numpy()
            model = TweedieRegressor(power=POWER, link="log", alpha=alpha, max_iter=3000, tol=1e-6)
    Xtr = make_X(tr, maps, norm, sc(KAPPA_CTX), sc(kc), sc(kd), players, keep)
    Xte = make_X(te, maps, norm, sc(KAPPA_CTX), sc(kc), sc(kd), players, keep)
    model.fit(Xtr, y, sample_weight=w)
    pred = model.predict(Xte)
    if kind != "gauss":
        bad = (getattr(model, "n_iter_", 2) <= 1) or (not np.all(np.isfinite(pred))) or (np.ptp(pred) == 0)
        if bad:
            raise RuntimeError(f"{kind} fit failed (iterations={getattr(model, 'n_iter_', '?')}, "
                               f"constant or non-finite predictions). Do not trust any score from this run.")
    return pred


def val_loss(kind, y, mu, w):
    f = {"gauss": sq_err, "pois": poisson_dev, "tweedie": tweedie_dev}[kind]
    return float((w * f(y, mu)).sum() / w.sum())


def tune(kind, tr):
    """Pick (kappa_career, kappa_season) on a time-ordered validation block inside training."""
    cut = tr.group_by("season").agg(c=pl.col("week").quantile(2 / 3, interpolation="nearest"))
    tr = tr.join(cut, on="season", how="left").with_columns(is_val=pl.col("week") > pl.col("c"))
    fit_df, val_df = tr.filter(~pl.col("is_val")), tr.filter(pl.col("is_val"))
    yv = val_df["rec_rate" if kind == "pois" else "y"].to_numpy()
    wv = val_df["pass_att"].to_numpy().astype(float)
    best = None
    for kc in KAPPA_CAREER:
        for kd in KAPPA_SEASON:
            t0 = time.time()
            mu = fit_predict(kind, fit_df, val_df, kc, kd)
            L = val_loss(kind, yv, mu, wv)
            print(f"   [{kind}] kappa_career={kc:>6.0f} kappa_season={kd:>6.0f} -> val loss {L:.5f} ({time.time()-t0:.0f}s)")
            if best is None or L < best[0]:
                best = (L, kc, kd)
    print(f"   [{kind}] chosen: kappa_career={best[1]:.0f}, kappa_season={best[2]:.0f}")
    if best[1] in (KAPPA_CAREER[0], KAPPA_CAREER[-1]) or best[2] in (KAPPA_SEASON[0], KAPPA_SEASON[-1]):
        print(f"   [{kind}] WARNING: chosen penalty is at the edge of the grid; widen KAPPA_CAREER / KAPPA_SEASON and re-run")
    return best[1], best[2]


# ---------------------------------------------------------------- bootstrap
def boot_diff(loss_a, loss_b, w, grp, rng):
    """Attempt-weighted mean loss of A minus B, resampling receiver-seasons (grp codes)."""
    n = grp.max() + 1
    sa = np.bincount(grp, weights=w * loss_a, minlength=n)
    sb = np.bincount(grp, weights=w * loss_b, minlength=n)
    sw = np.bincount(grp, weights=w, minlength=n)
    point = (sa.sum() - sb.sum()) / sw.sum()
    cnt = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
    d = (cnt @ sa - cnt @ sb) / (cnt @ sw)
    return point, np.percentile(d, 2.5), np.percentile(d, 97.5)


def verdict(name, a, b, f, y, mu, w, grp, rng):
    pt, lo, hi = boot_diff(f(y, mu[a]), f(y, mu[b]), w, grp, rng)
    tag = "A BETTER" if hi < 0 else ("A WORSE" if lo > 0 else "not distinguishable")
    print(f"   {name:<34} {pt:+.5f}  95% [{lo:+.5f}, {hi:+.5f}]  {tag}")


def main():
    P = pl.read_parquet(DATA / PANEL).with_columns(
        cl_rec=pl.col("athlete_id").cast(pl.String) + "|" + pl.col("team") + "|" + pl.col("season").cast(pl.String),
        rec_rate=pl.col("rec") / pl.col("pass_att"),
        # negative receiving yards are real (loss on a catch) but Tweedie needs y >= 0; clip once so
        # every model and baseline is fit and scored on the same target (the QB test clipped too)
        y=pl.col("yds").clip(lower_bound=0) / pl.col("pass_att"))
    print(f"rows with negative receiving yards, clipped to 0: {int((pl.read_parquet(DATA / PANEL)['yds'] < 0).sum())}")
    tr, te = P.filter(~pl.col("is_test")), P.filter(pl.col("is_test"))
    print(f"train rows {tr.height}, test rows {te.height}, test receiver-seasons {te['cl_rec'].n_unique()}")

    yte, rte = te["y"].to_numpy(), te["rec_rate"].to_numpy()
    w = te["pass_att"].to_numpy().astype(float)
    mu = {}

    # ---- baselines (training information only)
    mean_y = float((tr["y"] * tr["pass_att"]).sum() / tr["pass_att"].sum())
    mean_r = (tr["rec"].sum() / tr["pass_att"].sum())
    own_y = np.maximum(te["rec_tr_rate"].fill_null(mean_y).to_numpy(), 0.0)
    own = (tr.group_by("cl_rec").agg(r=pl.col("rec").sum(), a=pl.col("pass_att").sum())
             .with_columns(own_r=pl.col("r") / pl.col("a")))
    own_r = te.join(own.select("cl_rec", "own_r"), on="cl_rec", how="left")["own_r"].fill_null(mean_r).to_numpy()
    mu["const_y"] = np.full(te.height, mean_y)
    mu["own_y"] = own_y
    mu["const_r"] = np.full(te.height, mean_r)
    mu["own_r"] = own_r

    # ---- context-only (no player effects) and full models
    chosen = {}
    for kind in ("gauss", "tweedie", "pois"):
        print(f"\n== {kind}")
        kc, kd = tune(kind, tr)
        chosen[kind] = (kc, kd)
        t0 = time.time()
        mu[f"{kind}_full"] = fit_predict(kind, tr, te, kc, kd, players=True)
        mu[f"{kind}_ctx"] = fit_predict(kind, tr, te, kc, kd, players=False)
        print(f"   fitted full and context-only in {time.time()-t0:.0f}s")

    # same penalties, opponent pass defense removed: tests whether opponent adjustment matters on this holdout
    kc, kd = chosen["tweedie"]
    mu["tweedie_nodef"] = fit_predict("tweedie", tr, te, kc, kd, drop=("ld",))
    kc, kd = chosen["pois"]
    mu["pois_nodef"] = fit_predict("pois", tr, te, kc, kd, drop=("ld",))

    # stricter baseline: the receiver's career average over ALL his training rows (same data M2's career effect sees)
    car = (tr.group_by("athlete_id").agg(yy=(pl.col("y") * pl.col("pass_att")).sum(), a=pl.col("pass_att").sum())
             .with_columns(own_c=pl.col("yy") / pl.col("a")))
    mu["own_career_y"] = np.maximum(
        te.join(car.select("athlete_id", "own_c"), on="athlete_id", how="left")["own_c"].fill_null(mean_y).to_numpy(), 0.0)

    # ---- scores
    grp = np.unique(te["cl_rec"].to_numpy(), return_inverse=True)[1]
    rng = np.random.default_rng(0)
    mae = lambda y, m: np.abs(y - m)
    dev_t = lambda y, m: tweedie_dev(y, m)
    dev_p = lambda y, m: poisson_dev(y, m)

    print("\n=== Held-out last third: attempt-weighted scores (lower is better) ===")
    print(f"{'model':<14}{'yards MAE':>11}{'yards Tweedie dev':>20}")
    for k in ("const_y", "own_y", "own_career_y", "gauss_ctx", "gauss_full", "tweedie_ctx", "tweedie_nodef", "tweedie_full"):
        print(f"{k:<14}{(w*mae(yte,mu[k])).sum()/w.sum():>11.5f}{(w*dev_t(yte,mu[k])).sum()/w.sum():>20.5f}")
    print(f"\n{'model':<14}{'receptions MAE':>16}{'Poisson dev':>14}")
    for k in ("const_r", "own_r", "pois_ctx", "pois_nodef", "pois_full"):
        print(f"{k:<14}{(w*mae(rte,mu[k])).sum()/w.sum():>16.5f}{(w*dev_p(rte,mu[k])).sum()/w.sum():>14.5f}")

    print("\n=== Differences, A minus B (negative = A better). Bootstrap over receiver-seasons ===")
    print(" Keep-or-drop rule: M2 must beat M0 and the receiver's own training average by more than the bootstrap error")
    print("-- yards, Tweedie deviance")
    verdict("M2 vs M0", "tweedie_full", "gauss_full", dev_t, yte, mu, w, grp, rng)
    verdict("M2 vs own training average", "tweedie_full", "own_y", dev_t, yte, mu, w, grp, rng)
    verdict("M2 vs context-only", "tweedie_full", "tweedie_ctx", dev_t, yte, mu, w, grp, rng)
    verdict("M2 vs career training average", "tweedie_full", "own_career_y", dev_t, yte, mu, w, grp, rng)
    verdict("M2 vs M2 without opp. defense", "tweedie_full", "tweedie_nodef", dev_t, yte, mu, w, grp, rng)
    print("-- yards, MAE of rate")
    verdict("M2 vs M0", "tweedie_full", "gauss_full", mae, yte, mu, w, grp, rng)
    verdict("M2 vs own training average", "tweedie_full", "own_y", mae, yte, mu, w, grp, rng)
    verdict("M2 vs context-only", "tweedie_full", "tweedie_ctx", mae, yte, mu, w, grp, rng)
    verdict("M2 vs career training average", "tweedie_full", "own_career_y", mae, yte, mu, w, grp, rng)
    verdict("M2 vs M2 without opp. defense", "tweedie_full", "tweedie_nodef", mae, yte, mu, w, grp, rng)
    print("-- receptions, Poisson deviance")
    verdict("M1 vs own training average", "pois_full", "own_r", dev_p, rte, mu, w, grp, rng)
    verdict("M1 vs context-only", "pois_full", "pois_ctx", dev_p, rte, mu, w, grp, rng)

    print("\n=== Does opponent adjustment matter? Rule: at least 2% lower held-out error ===")
    for lab, y_, a, b, f in (("M2 yards, Tweedie deviance", yte, "tweedie_full", "tweedie_nodef", dev_t),
                             ("M2 yards, MAE", yte, "tweedie_full", "tweedie_nodef", mae),
                             ("M1 receptions, Poisson deviance", rte, "pois_full", "pois_nodef", dev_p)):
        la, lb = (w * f(y_, mu[a])).sum() / w.sum(), (w * f(y_, mu[b])).sum() / w.sum()
        pct = 100 * (lb - la) / lb
        print(f"   {lab:<34} without {lb:.5f} -> with {la:.5f}: {pct:+.2f}% lower  ->  {'passes 2% rule' if pct >= 2 else 'below 2% rule'}")

    out = te.select("game_id", "team", "athlete_id", "cl_rec", "season", "pass_att", "y", "rec_rate").with_columns(
        **{k: pl.Series(v) for k, v in mu.items()})
    out.write_parquet(DATA / "preds_step3.parquet")
    print("\nwrote preds_step3.parquet (test-row predictions for steps 4-6)")


if __name__ == "__main__":
    main()