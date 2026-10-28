"""
Step 3b: joint-fit opponent adjustment (the design in the model doc).

Step 3 used PRE-COMPUTED context features (opponent defense rating, teammates-only offense).  Here the
team-season offense effect and the opponent-season pass-defense effect are FITTED effects, estimated
together with the receiver career effect and season deviation in one ridge-penalized Tweedie model:

    log E[yards per team attempt] = b0 + home + neutral
                                    + offense[team-season] + defense[opponent-season]
                                    + career[receiver] + season_deviation[receiver-season]

Fitted jointly, each defense is measured against the offenses it actually faced AND each offense against
the defenses it faced, which the one-pass features in step 3 could not do.

Everything is fit on training games only; scored on the held-out last third.  Compares at TWO levels:
  * receiver-game (deviance and MAE of the per-attempt rate): noisy, because single-game receiver yards are
  * team-game (sum of the receivers' predictions vs the group's actual yards): the scale the 2% rule was set on

Usage (from project root):
    python src/04_joint_adjustment.py [data_dir] [panel_feat_file]
Optional: if preds_step3.parquet is in data_dir, the step-3 M2 is included as a comparison.
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.linear_model import TweedieRegressor

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")
PANEL = sys.argv[2] if len(sys.argv) > 2 else "panel_v2_feat.parquet"

POWER = 1.5
FLOOR = 0.01
KAPPA_REF = 200.0
KAPPA_CTX = 10.0
KC, KD = 100.0, 300.0                 # career / season-deviation penalties chosen in step 3 (Tweedie). Held fixed here.
# Penalties for team-level effects, in receiver-row pseudo-attempts.  A team-game appears once per group
# receiver (about 9 rows), so the step-3 K=220 team attempts is roughly 2000 here.
GRID_OFF = (500.0, 2000.0, 8000.0, 32000.0)
GRID_DEF = (500.0, 2000.0, 8000.0, 32000.0)
N_BOOT = 1000
KEYS = {"off": "cl_off", "def": "cl_def", "car": "athlete_id", "sea": "cl_rec"}


def tweedie_dev(y, mu, p=POWER):
    mu = np.maximum(mu, FLOOR)
    t1 = np.where(y > 0, np.power(np.maximum(y, 1e-12), 2 - p) / ((1 - p) * (2 - p)), 0.0)
    return 2 * (t1 - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p))


def ctx_raw(df):
    home = (df["is_home"].fill_null(False) & ~df["neutralSite"].fill_null(False)).cast(pl.Float64)
    return np.column_stack([home.to_numpy(), df["neutralSite"].fill_null(False).cast(pl.Float64).to_numpy()])


def design(df, maps, norm, kap, use):
    sc = lambda k: np.sqrt(KAPPA_REF / k)
    n = df.height
    mats = [sp.csr_matrix((ctx_raw(df) - norm[0]) / norm[1] * sc(KAPPA_CTX))]
    sl, pos = {}, mats[0].shape[1]
    for b in use:
        mp = maps[b]
        codes = np.array([mp.get(v, -1) for v in df[KEYS[b]].to_list()])
        ok = np.nonzero(codes >= 0)[0]
        mats.append(sp.csr_matrix((np.full(ok.size, sc(kap[b])), (ok, codes[ok])), shape=(n, len(mp))))
        sl[b] = (pos, pos + len(mp))
        pos += len(mp)
    return sp.hstack(mats).tocsr(), sl


def fit_predict(tr, te, use, kap):
    maps = {b: {v: i for i, v in enumerate(tr[KEYS[b]].unique().to_list())} for b in use}
    raw = ctx_raw(tr)
    norm = (raw.mean(0), raw.std(0) + 1e-9)
    w = tr["pass_att"].to_numpy().astype(float)
    # GLM objective (1/2sw) sum w dev + (alpha/2)||b||^2; alpha fixed so column scales stay near 1
    model = TweedieRegressor(power=POWER, link="log", alpha=KAPPA_REF / w.sum(), max_iter=3000, tol=1e-6)
    Xtr, sl = design(tr, maps, norm, kap, use)
    Xte, _ = design(te, maps, norm, kap, use)
    model.fit(Xtr, tr["y"].to_numpy(), sample_weight=w)
    pred = model.predict(Xte)
    if getattr(model, "n_iter_", 2) <= 1 or not np.all(np.isfinite(pred)) or np.ptp(pred) == 0:
        raise RuntimeError("Tweedie fit failed (stopped at iteration 0 or constant predictions). Do not trust this run.")
    return pred, model, sl, maps


def tune(tr):
    cut = tr.group_by("season").agg(c=pl.col("week").quantile(2 / 3, interpolation="nearest"))
    tr = tr.join(cut, on="season", how="left").with_columns(is_val=pl.col("week") > pl.col("c"))
    fit_df, val_df = tr.filter(~pl.col("is_val")), tr.filter(pl.col("is_val"))
    yv, wv = val_df["y"].to_numpy(), val_df["pass_att"].to_numpy().astype(float)
    best = None
    for ko in GRID_OFF:
        for kdf in GRID_DEF:
            t0 = time.time()
            mu, *_ = fit_predict(fit_df, val_df, ("off", "def", "car", "sea"),
                                 {"off": ko, "def": kdf, "car": KC, "sea": KD})
            L = float((wv * tweedie_dev(yv, mu)).sum() / wv.sum())
            print(f"   kappa_offense={ko:>6.0f} kappa_defense={kdf:>6.0f} -> val loss {L:.5f} ({time.time()-t0:.0f}s)")
            if best is None or L < best[0]:
                best = (L, ko, kdf)
    print(f"   chosen: kappa_offense={best[1]:.0f}, kappa_defense={best[2]:.0f}")
    if best[1] in (GRID_OFF[0], GRID_OFF[-1]) or best[2] in (GRID_DEF[0], GRID_DEF[-1]):
        print("   WARNING: chosen penalty is at the edge of the grid; widen GRID_OFF / GRID_DEF and re-run")
    return best[1], best[2]


def boot_pct(la, lb, w, grp, rng):
    """Percent by which A's weighted mean loss is below B's, with a cluster bootstrap CI."""
    n = grp.max() + 1
    sa = np.bincount(grp, weights=w * la, minlength=n)
    sb = np.bincount(grp, weights=w * lb, minlength=n)
    sw = np.bincount(grp, weights=w, minlength=n)
    cnt = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
    A, B = cnt @ sa / (cnt @ sw), cnt @ sb / (cnt @ sw)
    pct = 100 * (B - A) / B
    return 100 * (sb.sum() - sa.sum()) / sb.sum(), np.percentile(pct, 2.5), np.percentile(pct, 97.5)


def line(label, la, lb, w, grp, rng):
    pt, lo, hi = boot_pct(la, lb, w, grp, rng)
    flag = "clear" if lo > 0 else ("worse" if hi < 0 else "not distinguishable from 0")
    bar = "  >= 2%" if pt >= 2 else "  < 2%"
    print(f"   {label:<30} {pt:+6.2f}% lower  95% [{lo:+.2f}, {hi:+.2f}]  {flag}{bar}")


def main():
    P = pl.read_parquet(DATA / PANEL).with_columns(
        cl_off=pl.col("team") + "|" + pl.col("season").cast(pl.String),
        cl_def=pl.col("opponent") + "|" + pl.col("season").cast(pl.String),
        cl_rec=pl.col("athlete_id").cast(pl.String) + "|" + pl.col("team") + "|" + pl.col("season").cast(pl.String),
        y=pl.col("yds").clip(lower_bound=0) / pl.col("pass_att"))
    tr, te = P.filter(~pl.col("is_test")), P.filter(pl.col("is_test"))
    print(f"train rows {tr.height}, test rows {te.height}")

    print("\n== tuning team-level penalties on a time-ordered validation block inside training")
    ko, kdf = tune(tr)
    kap = {"off": ko, "def": kdf, "car": KC, "sea": KD}

    print("\n== fitting")
    mu = {}
    mu["J_full"], model, sl, maps = fit_predict(tr, te, ("off", "def", "car", "sea"), kap)
    mu["J_nodef"], *_ = fit_predict(tr, te, ("off", "car", "sea"), kap)
    mu["J_none"], *_ = fit_predict(tr, te, ("car", "sea"), kap)

    # size of the fitted team effects, on the percent-of-expected-yards scale
    sc = lambda k: np.sqrt(KAPPA_REF / k)
    for b, nm in (("off", "offense"), ("def", "opponent defense")):
        a, z = sl[b]
        eff = model.coef_[a:z] * sc(kap[b])
        print(f"   fitted {nm} effect (team-seasons seen: {z-a}): sd {100*np.std(eff):.1f}% | 10th-90th pct "
              f"{100*(np.exp(np.percentile(eff,10))-1):+.1f}% to {100*(np.exp(np.percentile(eff,90))-1):+.1f}%")

    # optional step-3 comparison
    f3 = DATA / "preds_step3.parquet"
    if f3.exists():
        s3 = pl.read_parquet(f3).select("game_id", "team", "athlete_id", pl.col("tweedie_full").alias("m2_step3"))
        te_m = te.select("game_id", "team", "athlete_id").join(s3, on=["game_id", "team", "athlete_id"], how="left")
        if te_m["m2_step3"].null_count() == 0:
            mu["M2_step3"] = te_m["m2_step3"].to_numpy()

    yte = te["y"].to_numpy()
    w = te["pass_att"].to_numpy().astype(float)
    rng = np.random.default_rng(0)
    pairs = [("J_full vs J_nodef  (opp. defense)", "J_full", "J_nodef"),
             ("J_full vs J_none   (all context)", "J_full", "J_none")]
    if "M2_step3" in mu:
        pairs += [("J_full vs M2 step 3 (joint vs 1-pass)", "J_full", "M2_step3"),
                  ("J_nodef vs M2 step 3 (no def, joint off)", "J_nodef", "M2_step3")]

    print("\n=== RECEIVER-GAME level (attempt-weighted; noisy). Percent lower error for A than B ===")
    grp_r = np.unique(te["cl_rec"].to_numpy(), return_inverse=True)[1]
    print(" Tweedie deviance")
    for lab, a, b in pairs:
        line(lab, tweedie_dev(yte, mu[a]), tweedie_dev(yte, mu[b]), w, grp_r, rng)
    print(" MAE of rate")
    for lab, a, b in pairs:
        line(lab, np.abs(yte - mu[a]), np.abs(yte - mu[b]), w, grp_r, rng)

    # ---- team-game level: sum receivers' predicted rates, compare with group's actual yards per attempt
    T = te.with_columns(**{k: pl.Series(v) for k, v in mu.items()}).group_by(["game_id", "team"]).agg(
        att=pl.col("pass_att").first(), y=pl.col("y").sum(), ts=pl.col("cl_off").first(),
        **{k: pl.col(k).sum() for k in mu})
    wt = T["att"].to_numpy().astype(float)
    yt = T["y"].to_numpy()
    grp_t = np.unique(T["ts"].to_numpy(), return_inverse=True)[1]
    print(f"\n=== TEAM-GAME level ({T.height} team-games; the scale the 2% rule was set on) ===")
    print(" MAE of group yards per attempt")
    for lab, a, b in pairs:
        line(lab, np.abs(yt - T[a].to_numpy()), np.abs(yt - T[b].to_numpy()), wt, grp_t, rng)
    print(" MSE of group yards per attempt")
    for lab, a, b in pairs:
        line(lab, (yt - T[a].to_numpy()) ** 2, (yt - T[b].to_numpy()) ** 2, wt, grp_t, rng)

    te.select("game_id", "team", "athlete_id", "season", "pass_att", "y").with_columns(
        **{k: pl.Series(v) for k, v in mu.items()}).write_parquet(DATA / "preds_joint.parquet")
    print("\nwrote preds_joint.parquet")


if __name__ == "__main__":
    main()