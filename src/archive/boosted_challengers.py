"""
Step 5: boosted challengers to M2 on the held-out college panel.

Models:
    - B1: LightGBM with M2's objective (Tweedie p=1.5) and features.
          Primary challenger; decides the keep-or-drop rule.
    - B3: LightGBM hybrid fit on M2's log-predictions as an offset (init_score).
          No player identifiers or individual production stats; isolates residual context interactions.
    - B2: XGBoost replicate with matching objective (reg:tweedie p=1.5) and features.
          Reported for verification, not chosen between (no bake-off).

Rules enforced here:
    - Trained strictly on training games (is_test == False).
    - Early stopping and hyperparameter selection on the time-ordered validation block
      inside training (last third of training weeks per season), never on test.
    - Attempt-weighted Tweedie deviance and MAE evaluated on the held-out last third.
    - Error bars and keep-or-drop decisions bootstrapped over receiver-seasons (cl_rec).
"""

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.linear_model import TweedieRegressor
import xgboost as xgb

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")
PANEL = sys.argv[2] if len(sys.argv) > 2 else "panel_v2_feat.parquet"

POWER = 1.5
FLOOR = 0.01
KAPPA_REF = 200.0
KAPPA_CTX = 10.0
KC, KD = 100.0, 300.0  # Frozen M2 penalties from Step 3

# Pre-specified tree constraints (shallow trees, heavy shrinkage, large leaves)
LEAF_GRID = (8, 12, 16)
MIN_CHILD_SAMPLES = 200
LEARNING_RATE = 0.03
EARLY_STOPPING_ROUNDS = 50
MAX_TREES = 1000
N_BOOT = 1000

# Feature definitions
CTX_COLS = ["off_oth_ypa", "off_team_ypa", "def_rating", "home", "neutral"]
B1_FEATURES = ["rec_tr_rate", "rec_tr_games"] + CTX_COLS
B3_FEATURES = CTX_COLS  # Hybrid strictly excludes player identifiers and production stats


# ---------------------------------------------------------------- loss & bootstrap
def tweedie_dev(y, mu, p=POWER):
  mu = np.maximum(mu, FLOOR)
  t1 = np.where(
      y > 0, np.power(np.maximum(y, 1e-12), 2 - p) / ((1 - p) * (2 - p)), 0.0
  )
  return 2 * (
      t1 - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p)
  )


def mae_loss(y, mu):
  return np.abs(y - mu)


def boot_diff(loss_a, loss_b, w, grp, rng):
  """Attempt-weighted loss difference (A minus B) clustered by receiver-season."""
  n = grp.max() + 1
  sa = np.bincount(grp, weights=w * loss_a, minlength=n)
  sb = np.bincount(grp, weights=w * loss_b, minlength=n)
  sw = np.bincount(grp, weights=w, minlength=n)
  point = (sa.sum() - sb.sum()) / sw.sum()
  cnt = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
  d = (cnt @ sa - cnt @ sb) / (cnt @ sw)
  return point, np.percentile(d, 2.5), np.percentile(d, 97.5)


def boot_pct(la, lb, w, grp, rng):
  """Percent by which A is lower than B, with 95% bootstrap interval."""
  n = grp.max() + 1
  sa = np.bincount(grp, weights=w * la, minlength=n)
  sb = np.bincount(grp, weights=w * lb, minlength=n)
  sw = np.bincount(grp, weights=w, minlength=n)
  cnt = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(float)
  A, B = cnt @ sa / (cnt @ sw), cnt @ sb / (cnt @ sw)
  pct = 100 * (B - A) / B
  pt = 100 * (sb.sum() - sa.sum()) / sb.sum()
  return pt, np.percentile(pct, 2.5), np.percentile(pct, 97.5)


# ---------------------------------------------------------------- M2 fitter
def ctx_raw_m2(df):
  return np.column_stack([
      np.log(df["off_oth_ypa"].to_numpy()),
      df["def_rating"].to_numpy(),
      df["home"].to_numpy(),
      df["neutral"].to_numpy(),
  ])


def fit_m2(tr, pred_dfs):
  """Fits frozen M2 Tweedie model on tr and returns predictions for each df in pred_dfs."""
  maps = (
      {a: i for i, a in enumerate(tr["athlete_id"].unique().to_list())},
      {c: i for i, c in enumerate(tr["cl_rec"].unique().to_list())},
  )
  raw = ctx_raw_m2(tr)
  norm = (raw.mean(0), raw.std(0) + 1e-9)
  w = tr["pass_att"].to_numpy().astype(float)
  sc = lambda k: np.sqrt(KAPPA_REF / k)

  def make_X(df):
    n = df.height
    cols = [
        sp.csr_matrix((ctx_raw_m2(df) - norm[0]) / norm[1] * sc(KAPPA_CTX))
    ]
    for col, mp, k in (
        ("athlete_id", maps[0], KC),
        ("cl_rec", maps[1], KD),
    ):
      codes = np.array([mp.get(v, -1) for v in df[col].to_list()])
      ok = np.nonzero(codes >= 0)[0]
      cols.append(
          sp.csr_matrix(
              (np.full(ok.size, sc(k)), (ok, codes[ok])), shape=(n, len(mp))
          )
      )
    return sp.hstack(cols).tocsr()

  Xtr = make_X(tr)
  model = TweedieRegressor(
      power=POWER,
      link="log",
      alpha=KAPPA_REF / w.sum(),
      max_iter=3000,
      tol=1e-6,
  )
  model.fit(Xtr, tr["y"].to_numpy(), sample_weight=w)
  return [model.predict(make_X(df)) for df in pred_dfs]


# ---------------------------------------------------------------- B1: LightGBM Challenger
def train_b1(fit_df, val_df, tr_df, te_df):
  print("\n== Tuning B1 (LightGBM Challenger) on inner validation block")
  X_fit = fit_df.select(B1_FEATURES).to_numpy()
  y_fit, w_fit = fit_df["y"].to_numpy(), fit_df["pass_att"].to_numpy()
  X_val = val_df.select(B1_FEATURES).to_numpy()
  y_val, w_val = val_df["y"].to_numpy(), val_df["pass_att"].to_numpy()

  best = None
  for leaves in LEAF_GRID:
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": POWER,
        "learning_rate": LEARNING_RATE,
        "num_leaves": leaves,
        "min_child_samples": MIN_CHILD_SAMPLES,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42,
    }
    dtrain = lgb.Dataset(X_fit, label=y_fit, weight=w_fit)
    dval = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=dtrain)

    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=MAX_TREES,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    val_pred = bst.predict(X_val)
    L = float((w_val * tweedie_dev(y_val, val_pred)).sum() / w_val.sum())
    print(
        f"   B1 leaves={leaves:>2} | best iter={bst.best_iteration:>4} | val"
        f" deviance={L:.5f}"
    )
    if best is None or L < best[0]:
      best = (L, leaves, bst.best_iteration)

  print(f"   B1 chosen: num_leaves={best[1]}, n_rounds={best[2]}")

  # Fit on full training set with chosen rounds
  X_tr = tr_df.select(B1_FEATURES).to_numpy()
  y_tr, w_tr = tr_df["y"].to_numpy(), tr_df["pass_att"].to_numpy()
  X_te = te_df.select(B1_FEATURES).to_numpy()

  dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
  params["num_leaves"] = best[1]
  bst_full = lgb.train(params, dtr, num_boost_round=best[2])
  return bst_full.predict(X_te), bst_full


# ---------------------------------------------------------------- B3: Hybrid LightGBM
def train_b3(fit_df, val_df, tr_df, te_df, m2_fit_sub, m2_val, m2_tr, m2_te):
  print("\n== Tuning B3 (Hybrid LightGBM with M2 Offset) on inner validation")
  X_fit = fit_df.select(B3_FEATURES).to_numpy()
  y_fit, w_fit = fit_df["y"].to_numpy(), fit_df["pass_att"].to_numpy()
  offset_fit = np.log(np.maximum(m2_fit_sub, FLOOR))

  X_val = val_df.select(B3_FEATURES).to_numpy()
  y_val, w_val = val_df["y"].to_numpy(), val_df["pass_att"].to_numpy()
  offset_val = np.log(np.maximum(m2_val, FLOOR))

  best = None
  for leaves in LEAF_GRID:
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": POWER,
        "learning_rate": LEARNING_RATE,
        "num_leaves": leaves,
        "min_child_samples": MIN_CHILD_SAMPLES,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42,
    }
    dtrain = lgb.Dataset(
        X_fit, label=y_fit, weight=w_fit, init_score=offset_fit
    )
    dval = lgb.Dataset(
        X_val,
        label=y_val,
        weight=w_val,
        init_score=offset_val,
        reference=dtrain,
    )

    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=MAX_TREES,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    # raw_score returns margin delta; add offset for Tweedie expectation
    val_pred = np.exp(offset_val + bst.predict(X_val, raw_score=True))
    L = float((w_val * tweedie_dev(y_val, val_pred)).sum() / w_val.sum())
    print(
        f"   B3 leaves={leaves:>2} | best iter={bst.best_iteration:>4} | val"
        f" deviance={L:.5f}"
    )
    if best is None or L < best[0]:
      best = (L, leaves, bst.best_iteration)

  print(f"   B3 chosen: num_leaves={best[1]}, n_rounds={best[2]}")

  # Fit on full training set with full M2 offset
  X_tr = tr_df.select(B3_FEATURES).to_numpy()
  y_tr, w_tr = tr_df["y"].to_numpy(), tr_df["pass_att"].to_numpy()
  offset_tr = np.log(np.maximum(m2_tr, FLOOR))
  X_te = te_df.select(B3_FEATURES).to_numpy()
  offset_te = np.log(np.maximum(m2_te, FLOOR))

  dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, init_score=offset_tr)
  params["num_leaves"] = best[1]
  bst_full = lgb.train(params, dtr, num_boost_round=best[2])

  pred_te = np.exp(offset_te + bst_full.predict(X_te, raw_score=True))
  return pred_te, bst_full


# ---------------------------------------------------------------- B2: XGBoost Replicate
def train_b2(fit_df, val_df, tr_df, te_df):
  print("\n== Tuning B2 (XGBoost Replicate) on inner validation")
  X_fit = fit_df.select(B1_FEATURES).to_numpy()
  y_fit, w_fit = fit_df["y"].to_numpy(), fit_df["pass_att"].to_numpy()
  X_val = val_df.select(B1_FEATURES).to_numpy()
  y_val, w_val = val_df["y"].to_numpy(), val_df["pass_att"].to_numpy()

  best = None
  for depth in (3, 4):
    params = {
        "objective": "reg:tweedie",
        "tweedie_variance_power": POWER,
        "learning_rate": LEARNING_RATE,
        "max_depth": depth,
        "min_child_weight": MIN_CHILD_SAMPLES,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": f"tweedie-nloglik@{POWER}",
    }
    dtrain = xgb.DMatrix(X_fit, label=y_fit, weight=w_fit)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=MAX_TREES,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    val_pred = bst.predict(dval)
    L = float((w_val * tweedie_dev(y_val, val_pred)).sum() / w_val.sum())
    print(
        f"   B2 depth={depth} | best iter={bst.best_iteration:>4} | val"
        f" deviance={L:.5f}"
    )
    if best is None or L < best[0]:
      best = (L, depth, bst.best_iteration)

  print(f"   B2 chosen: max_depth={best[1]}, n_rounds={best[2]}")

  X_tr = tr_df.select(B1_FEATURES).to_numpy()
  y_tr, w_tr = tr_df["y"].to_numpy(), tr_df["pass_att"].to_numpy()
  X_te = te_df.select(B1_FEATURES).to_numpy()

  dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
  dte = xgb.DMatrix(X_te)
  params["max_depth"] = best[1]
  bst_full = xgb.train(params, dtr, num_boost_round=best[2] + 1)
  return bst_full.predict(dte), bst_full


# ---------------------------------------------------------------- main execution
def main():
  P = pl.read_parquet(DATA / PANEL).with_columns(
      cl_rec=pl.col("athlete_id").cast(pl.String)
      + "|"
      + pl.col("team")
      + "|"
      + pl.col("season").cast(pl.String),
      home=(
          pl.col("is_home").fill_null(False)
          & ~pl.col("neutralSite").fill_null(False)
      ).cast(pl.Float64),
      neutral=pl.col("neutralSite").fill_null(False).cast(pl.Float64),
      y=pl.col("yds").clip(lower_bound=0) / pl.col("pass_att"),
  )

  tr, te = P.filter(~pl.col("is_test")), P.filter(pl.col("is_test"))
  print(f"train rows: {tr.height} | test rows: {te.height}")

  # Time-ordered validation block inside training
  cut = tr.group_by("season").agg(
      c=pl.col("week").quantile(2 / 3, interpolation="nearest")
  )
  tr = tr.join(cut, on="season", how="left").with_columns(
      is_val=pl.col("week") > pl.col("c")
  )
  fit_df, val_df = tr.filter(~pl.col("is_val")), tr.filter(pl.col("is_val"))

  # 1. Fit M2 baselines for test evaluation and hybrid offsets
  t0 = time.time()
  print("\n== Fitting structured M2 baseline")
  m2_fit_sub, m2_val = fit_m2(fit_df, [fit_df, val_df])
  m2_tr, m2_te = fit_m2(tr, [tr, te])
  print(f"   M2 fitted in {time.time() - t0:.1f}s")

  # 2. Train B1, B3, B2
  b1_te, bst_b1 = train_b1(fit_df, val_df, tr, te)
  b3_te, bst_b3 = train_b3(
      fit_df, val_df, tr, te, m2_fit_sub, m2_val, m2_tr, m2_te
  )
  b2_te, bst_b2 = train_b2(fit_df, val_df, tr, te)

  # 3. Held-out Evaluation
  yte = te["y"].to_numpy()
  w = te["pass_att"].to_numpy().astype(float)
  grp = np.unique(te["cl_rec"].to_numpy(), return_inverse=True)[1]
  rng = np.random.default_rng(0)

  models = {
      "M2 Tweedie (frozen)": m2_te,
      "B1 LightGBM (primary)": b1_te,
      "B3 Hybrid (M2 + trees)": b3_te,
      "B2 XGBoost (replicate)": b2_te,
  }

  print(
      "\n=== Held-Out Last Third: Attempt-Weighted Scores (Lower is Better)"
      " ==="
  )
  print(f"{'model':<26}{'Tweedie deviance':>18}{'rate MAE':>12}")
  for name, pred in models.items():
    dev = float((w * tweedie_dev(yte, pred)).sum() / w.sum())
    mae = float((w * mae_loss(yte, pred)).sum() / w.sum())
    print(f"{name:<26}{dev:>18.5f}{mae:>12.5f}")

  print(
      "\n=== Differences vs M2 (A minus M2; negative = challenger better) ==="
  )
  print(" Keep-or-drop rule: challenger must beat M2 by more than bootstrap CI")
  for label, key in (
      ("B1 vs M2 (primary decision)", "B1 LightGBM (primary)"),
      ("B3 vs M2 (hybrid challenger)", "B3 Hybrid (M2 + trees)"),
      ("B2 vs M2 (reported replicate)", "B2 XGBoost (replicate)"),
  ):
    pt, lo, hi = boot_diff(
        tweedie_dev(yte, models[key]), tweedie_dev(yte, m2_te), w, grp, rng
    )
    pct, plo, phi = boot_pct(
        tweedie_dev(yte, models[key]), tweedie_dev(yte, m2_te), w, grp, rng
    )
    verdict = (
        "CHALLENGER BEATS M2"
        if hi < 0
        else ("CHALLENGER WORSE" if lo > 0 else "not distinguishable from M2")
    )
    print(
        f"   {label:<30} {pct:+6.2f}% error ({pt:+.5f})  95% [{lo:+.5f},"
        f" {hi:+.5f}]  {verdict}"
    )

  # Check consistency between B1 and B2 replicate
  pt_rep, lo_rep, hi_rep = boot_diff(
      tweedie_dev(yte, b2_te), tweedie_dev(yte, b1_te), w, grp, rng
  )
  print(
      f"\n   Replicate check (B2 vs B1): diff {pt_rep:+.5f}  95% [{lo_rep:+.5f},"
      f" {hi_rep:+.5f}]"
  )

  # Inspect B3 tree importances (did trees find leftover context interactions?)
  print("\n=== B3 Feature Importances (Gain) ===")
  imp = bst_b3.feature_importance(importance_type="gain")
  total_gain = imp.sum() if imp.sum() > 0 else 1.0
  for name, gain in sorted(zip(B3_FEATURES, imp), key=lambda x: -x[1]):
    print(f"   {name:<20} {gain:>10.2f} ({100 * gain / total_gain:>5.1f}%)")

  # Write predictions for Step 6 checks
  out = te.select(
      "game_id", "team", "athlete_id", "cl_rec", "season", "pass_att", "y"
  ).with_columns(
      m2_pred=pl.Series(m2_te),
      b1_pred=pl.Series(b1_te),
      b3_pred=pl.Series(b3_te),
      b2_pred=pl.Series(b2_te),
  )
  out.write_parquet(DATA / "preds_step5.parquet")
  print("\nwrote preds_step5.parquet")


if __name__ == "__main__":
  main()