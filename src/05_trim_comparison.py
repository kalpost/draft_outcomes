"""
Step 4: trimmed vs untrimmed zero-game runs.

Rule from the model doc: injuries come in runs and ordinary quiet games do not, so trim runs of 3+
zero-yard games at the START or END of a receiver's season.  Fit with and without trimming.  Keep trimming
only if rankings or scores move meaningfully; otherwise drop it for simplicity.

How trimming is applied here (leakage-safe):
  * Only TRAINING rows are ever trimmed.  The test set is identical for both fits (all held-out rows), because
    at prediction time you cannot know a game will be an injury game.
  * The "start" run is the run of zero-yard games before the receiver's first non-zero training game.
  * The "end" run is the run of zero-yard games ending at the training cutoff (the last training game),
    not at the real end of the season, so no test-period outcome decides which training rows are dropped.
  * A zero-yard game is yards <= 0 (negative yards are clipped to 0 as in step 3).

Both fits use the same step-3 context features and the same penalties (career, season deviation), so the only
thing that differs is which training rows are used.

Two variants are tested separately, because they answer different questions:
  start       : drop only the zero run before the receiver's first non-zero game (not yet healthy or in the rotation)
  start+end   : also drop the zero run at the training cutoff.  CAUTION: a receiver with 3+ zero games at the
                cutoff is often still out in the test period, so the untrimmed model correctly predicts more
                zeros.  Scored on all test rows, end trimming is penalised by construction.  The 'available
                receivers' check below removes receiver-seasons that end the season in a 3+ zero run, to show
                how the variants compare when the receiver is playing.  That check conditions on outcomes, so
                it is DESCRIPTIVE ONLY and is not used for the verdict.

DECISION THRESHOLDS (set before any result was seen; edit them BEFORE running if you disagree):
  Scores move meaningfully  : trimmed test deviance is lower by >= SCORE_PCT with a bootstrap CI above 0.
  Rankings move meaningfully: Spearman of receiver-season ratings < SPEARMAN_MIN, or top-decile overlap
                              < OVERLAP_MIN, among receiver-seasons with >= MIN_ATT_RANK training attempts.

Usage (from project root):
    python src/05_trim_comparison.py [data_dir] [panel_feat_file]
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.linear_model import TweedieRegressor

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")
PANEL = sys.argv[2] if len(sys.argv) > 2 else "panel_v2_feat.parquet"

RUN_MIN = 3                  # trim runs of 3+ zero-yard games at the start or end
POWER = 1.5
FLOOR = 0.01
KAPPA_REF = 200.0
KAPPA_CTX = 10.0
KC, KD = 100.0, 300.0        # career / season-deviation penalties chosen in step 3 (Tweedie)
N_BOOT = 1000
SCORE_PCT = 0.5              # percent lower deviance that counts as "moves scores"
SPEARMAN_MIN = 0.98
OVERLAP_MIN = 0.90
MIN_ATT_RANK = 150


def tweedie_dev(y, mu, p=POWER):
    mu = np.maximum(mu, FLOOR)
    t1 = np.where(y > 0, np.power(np.maximum(y, 1e-12), 2 - p) / ((1 - p) * (2 - p)), 0.0)
    return 2 * (t1 - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p))


def ctx_raw(df):
    home = (df["is_home"].fill_null(False) & ~df["neutralSite"].fill_null(False)).cast(pl.Float64)
    return np.column_stack([
        np.log(df["off_oth_ypa"].to_numpy()), df["def_rating"].to_numpy(),
        home.to_numpy(), df["neutralSite"].fill_null(False).cast(pl.Float64).to_numpy()])


def fit_predict(tr, te):
    """Step-3 M2: context + receiver career effect + receiver-season deviation, Tweedie, training rows `tr`."""
    car = {a: i for i, a in enumerate(tr["athlete_id"].unique().to_list())}
    sea = {c: i for i, c in enumerate(tr["cl_rec"].unique().to_list())}
    raw = ctx_raw(tr)
    norm = (raw.mean(0), raw.std(0) + 1e-9)
    w = tr["pass_att"].to_numpy().astype(float)
    sc = lambda k: np.sqrt(KAPPA_REF / k)

    def X(df):
        n = df.height
        mats = [sp.csr_matrix((ctx_raw(df) - norm[0]) / norm[1] * sc(KAPPA_CTX))]
        for col, mp, k in (("athlete_id", car, KC), ("cl_rec", sea, KD)):
            codes = np.array([mp.get(v, -1) for v in df[col].to_list()])
            ok = np.nonzero(codes >= 0)[0]
            mats.append(sp.csr_matrix((np.full(ok.size, sc(k)), (ok, codes[ok])), shape=(n, len(mp))))
        return sp.hstack(mats).tocsr()

    model = TweedieRegressor(power=POWER, link="log", alpha=KAPPA_REF / w.sum(), max_iter=3000, tol=1e-6)
    model.fit(X(tr), tr["y"].to_numpy(), sample_weight=w)
    pred = model.predict(X(te))
    if getattr(model, "n_iter_", 2) <= 1 or not np.all(np.isfinite(pred)) or np.ptp(pred) == 0:
        raise RuntimeError("Tweedie fit failed (iteration 0 or constant predictions). Do not trust this run.")

    nc = ctx_raw(tr.head(1)).shape[1]
    b_car = model.coef_[nc:nc + len(car)] * sc(KC)
    b_sea = model.coef_[nc + len(car):] * sc(KD)
    pairs = tr.select("cl_rec", "athlete_id").unique()
    rating = {r: b_car[car[a]] + b_sea[sea[r]] for r, a in zip(pairs["cl_rec"].to_list(), pairs["athlete_id"].to_list())}
    return pred, rating


def flag_trim(tr):
    """Add trim flags to training rows; returns df with lead/trail/trim columns (all causal, training rows only)."""
    tr = (tr.with_columns(post=(pl.col("seasonType") == "postseason").cast(pl.Int8))
            .sort(["cl_rec", "post", "week", "game_id"]))
    nz = (pl.col("y") > 0).cast(pl.Int32)
    tr = tr.with_columns(
        before_first=nz.cum_sum().over("cl_rec") == 0,                    # zeros before first non-zero game
        after_last=nz.cum_sum(reverse=True).over("cl_rec") == 0)         # zeros after last non-zero game
    tr = tr.with_columns(
        lead_len=pl.col("before_first").sum().over("cl_rec"),
        trail_len=pl.col("after_last").sum().over("cl_rec"))
    return tr.with_columns(
        trim_lead=pl.col("before_first") & (pl.col("lead_len") >= RUN_MIN),
        trim_trail=pl.col("after_last") & (pl.col("trail_len") >= RUN_MIN)
    ).with_columns(trim=pl.col("trim_lead") | pl.col("trim_trail"))


def boot_pct(la, lb, w, grp, rng):
    n = grp.max() + 1
    sa = np.bincount(grp, weights=w * la, minlength=n)
    sb = np.bincount(grp, weights=w * lb, minlength=n)
    sw = np.bincount(grp, weights=w, minlength=n)
    cnt = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
    A, B = cnt @ sa / (cnt @ sw), cnt @ sb / (cnt @ sw)
    pct = 100 * (B - A) / B
    return 100 * (sb.sum() - sa.sum()) / sb.sum(), np.percentile(pct, 2.5), np.percentile(pct, 97.5)


def rank_of(x):
    return np.argsort(np.argsort(x)).astype(float)


def verdict_lines(name, pt, lo, hi, sp_, overlap):
    score_move = pt >= SCORE_PCT and lo > 0
    score_worse = hi < 0
    rank_move = sp_ < SPEARMAN_MIN or overlap < OVERLAP_MIN
    print(f"   [{name}] scores move meaningfully (>= {SCORE_PCT}% lower, CI above 0): {score_move} | clearly worse: {score_worse}"
          f" | rankings move (Spearman < {SPEARMAN_MIN} or overlap < {100*OVERLAP_MIN:.0f}%): {rank_move}")
    if score_move:
        print(f"      -> KEEP [{name}] (scores improve).")
    elif score_worse:
        print(f"      -> DROP [{name}] (scores worse on the held-out rows).")
    elif rank_move:
        print(f"      -> AMBIGUOUS [{name}]: rankings move but scores do not improve. The doc rule says keep; scores say the "
              f"move is not an improvement. Decide deliberately and write down why.")
    else:
        print(f"      -> DROP [{name}] for simplicity (neither scores nor rankings move meaningfully).")


def main():
    P = pl.read_parquet(DATA / PANEL).with_columns(
        cl_rec=pl.col("athlete_id").cast(pl.String) + "|" + pl.col("team") + "|" + pl.col("season").cast(pl.String),
        y=pl.col("yds").clip(lower_bound=0) / pl.col("pass_att"))
    tr, te = P.filter(~pl.col("is_test")), P.filter(pl.col("is_test"))
    tr = flag_trim(tr)

    n_tr = tr.height
    print(f"training rows {n_tr}, test rows {te.height}")
    for nm, c in (("start runs", "trim_lead"), ("end-of-training runs", "trim_trail"), ("either", "trim")):
        n = int(tr[c].sum())
        a = tr.filter(pl.col(c)).select("cl_rec").unique().height
        print(f"   {nm:<22} rows {n:>6} ({100*n/n_tr:.2f}% of training rows) | receiver-seasons {a}")
    print(f"zero-yard share of training rows: {100*(tr['y']==0).mean():.1f}% -> "
          f"{100*(tr.filter(~pl.col('trim'))['y']==0).mean():.1f}% after trimming both")

    # test receiver-seasons that end the season in a run of 3+ zero-yard games (used only for the descriptive check)
    tes = (te.with_columns(post=(pl.col("seasonType") == "postseason").cast(pl.Int8))
             .sort(["cl_rec", "post", "week", "game_id"]))
    tes = tes.with_columns(tail0=((pl.col("y") > 0).cast(pl.Int32).cum_sum(reverse=True).over("cl_rec") == 0))
    out_end = set(tes.group_by("cl_rec").agg(n=pl.col("tail0").sum()).filter(pl.col("n") >= RUN_MIN)["cl_rec"].to_list())
    avail = ~te["cl_rec"].is_in(list(out_end)).to_numpy()

    yte = te["y"].to_numpy()
    w = te["pass_att"].to_numpy().astype(float)
    grp = np.unique(te["cl_rec"].to_numpy(), return_inverse=True)[1]
    rng = np.random.default_rng(0)
    pred_u, rate_u = fit_predict(tr, te)
    dev_u, mae_u = tweedie_dev(yte, pred_u), np.abs(yte - pred_u)
    print(f"\nuntrimmed: deviance {(w*dev_u).sum()/w.sum():.5f}  MAE {(w*mae_u).sum()/w.sum():.5f}")

    att = tr.group_by("cl_rec").agg(a=pl.col("pass_att").sum())
    base_keys = att.filter(pl.col("a") >= MIN_ATT_RANK)["cl_rec"].to_list()
    verdicts = []
    for name, col in (("start", "trim_lead"), ("start+end", "trim")):
        fit_df = tr.filter(~pl.col(col))
        pred_t, rate_t = fit_predict(fit_df, te)
        dev_t, mae_t = tweedie_dev(yte, pred_t), np.abs(yte - pred_t)
        aff = tr.filter(pl.col(col)).select("cl_rec").unique()
        print(f"\n=================== variant: {name}  (drops {tr.height - fit_df.height} training rows) ===================")
        print(f"   trimmed:   deviance {(w*dev_t).sum()/w.sum():.5f}  MAE {(w*mae_t).sum()/w.sum():.5f}")
        pt, lo, hi = boot_pct(dev_t, dev_u, w, grp, rng)
        print(f"   ALL test rows, deviance: trimmed {pt:+.3f}% lower  95% [{lo:+.3f}, {hi:+.3f}]  (bootstrap over receiver-seasons)")
        pm, lm, hm = boot_pct(mae_t, mae_u, w, grp, rng)
        print(f"   ALL test rows, MAE:      trimmed {pm:+.3f}% lower  95% [{lm:+.3f}, {hm:+.3f}]")
        g2 = np.unique(te["cl_rec"].to_numpy()[avail], return_inverse=True)[1]
        pa, la_, ha = boot_pct(dev_t[avail], dev_u[avail], w[avail], g2, rng)
        print(f"   descriptive only, available receivers ({int(avail.sum())} test rows, excludes seasons ending in a "
              f"{RUN_MIN}+ zero run): deviance {pa:+.3f}% lower  95% [{la_:+.3f}, {ha:+.3f}]")

        keys = [k for k in base_keys if k in rate_u and k in rate_t]
        ru = np.array([rate_u[k] for k in keys]); rt = np.array([rate_t[k] for k in keys])
        ru, rt = ru - ru.mean(), rt - rt.mean()                       # centre: only relative position matters
        sp_ = float(np.corrcoef(rank_of(ru), rank_of(rt))[0, 1])
        k10 = max(1, int(0.1 * len(keys)))
        overlap = len(set(np.argsort(-ru)[:k10]) & set(np.argsort(-rt)[:k10])) / k10
        d = np.exp(rt - ru) - 1
        aff_set = set(aff["cl_rec"].to_list())
        isaff = np.array([k in aff_set for k in keys])
        print(f"   rankings ({len(keys)} receiver-seasons with >= {MIN_ATT_RANK} training attempts): "
              f"Spearman {sp_:.4f} | top-decile overlap {100*overlap:.1f}%")
        if isaff.any():
            print(f"   affected receiver-seasons ({int(isaff.sum())}): mean rating change {100*d[isaff].mean():+.2f}% "
                  f"(median {100*np.median(d[isaff]):+.2f}%), share moving > 5%: {100*(np.abs(d[isaff])>0.05).mean():.1f}%")
        verdicts.append((name, pt, lo, hi, sp_, overlap))

    print("\n=== VERDICT (thresholds fixed in the script header; based on ALL test rows) ===")
    for v in verdicts:
        verdict_lines(*v)


if __name__ == "__main__":
    main()