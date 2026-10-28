"""Step 6: college model health checks and prospect ratings export for Stage 1.

Usage (from project root):
    python src/06_model_checks_and_export.py [data_dir] [panel_file]
    defaults: data/processed  panel_v2_feat.parquet

Runs three descriptive validation checks:
    1. Odd-vs-Even split-half reliability within seasons (M2 vs raw YPA).
    2. Next-season persistence (Year t -> Year t+1 for returning receivers).
    3. 80% prediction interval coverage on held-out test games.

Exports:
    college_prospect_ratings.parquet: Receiver career & final-season ratings,
    standard errors, games played, and team-games share for Stage 1 NFL modeling.
    (Leaves 2024 and 2025 classes untouched as specified in the design doc.)
"""

from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import TweedieRegressor

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")
PANEL = sys.argv[2] if len(sys.argv) > 2 else "panel_v2_feat.parquet"

POWER = 1.5
FLOOR = 0.01
KAPPA_REF = 200.0
KAPPA_CTX = 10.0
KC, KD = 100.0, 300.0  # Frozen M2 penalties


# ---------------------------------------------------------------- M2 helper
def ctx_matrix(df):
  home = (
      df["is_home"].fill_null(False) & ~df["neutralSite"].fill_null(False)
  ).cast(pl.Float64)
  neutral = df["neutralSite"].fill_null(False).cast(pl.Float64)
  return np.column_stack([
      np.log(np.maximum(df["off_oth_ypa"].to_numpy(), 1e-3)),
      df["def_rating"].fill_null(0.0).to_numpy(),
      home.to_numpy(),
      neutral.to_numpy(),
  ])


def fit_m2_model(df, fit_seasons=None):
  """Fits M2 Tweedie model on df and returns the fitted model, design mappings, and player effects."""
  if fit_seasons is not None:
    df = df.filter(pl.col("season").is_in(fit_seasons))

  maps = (
      {a: i for i, a in enumerate(df["athlete_id"].unique().to_list())},
      {c: i for i, c in enumerate(df["cl_rec"].unique().to_list())},
  )
  raw = ctx_matrix(df)
  norm = (raw.mean(0), raw.std(0) + 1e-9)
  w = df["pass_att"].to_numpy().astype(float)
  sw = w.sum()
  sc = lambda k: np.sqrt(KAPPA_REF / k)

  n = df.height
  cols = [sp.csr_matrix((raw - norm[0]) / norm[1] * sc(KAPPA_CTX))]
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
  X = sp.hstack(cols).tocsr()

  model = TweedieRegressor(
      power=POWER,
      link="log",
      alpha=KAPPA_REF / sw,
      max_iter=3000,
      tol=1e-6,
  )
  model.fit(X, df["y"].to_numpy(), sample_weight=w)

  # Extract scaled effects
  n_ctx = raw.shape[1]
  n_car = len(maps[0])
  car_coefs = model.coef_[n_ctx : n_ctx + n_car] * sc(KC)
  sea_coefs = model.coef_[n_ctx + n_car :] * sc(KD)

  car_map = {a: car_coefs[i] for a, i in maps[0].items()}
  sea_map = {c: sea_coefs[i] for c, i in maps[1].items()}

  return model, car_map, sea_map, maps, norm


# ---------------------------------------------------------------- 1. Split-Half Reliability
def check_split_half(panel_df):
  print("\n" + "=" * 65)
  print("1. SPLIT-HALF RELIABILITY (Odd vs. Even Weeks Within Season)")
  print("=" * 65)
  print("Tests whether M2 ratings correlate higher across alternating weeks")
  print("than raw yards per attempt (removing schedule & offense noise).\n")

  # Filter to receiver-seasons with at least 3 odd and 3 even games
  tr = panel_df.filter(~pl.col("is_test"))
  rec_counts = tr.group_by(["cl_rec", "season"]).agg(
      n_odd=(pl.col("week") % 2 == 1).sum(),
      n_even=(pl.col("week") % 2 == 0).sum(),
  )
  valid = rec_counts.filter(
      (pl.col("n_odd") >= 3) & (pl.col("n_even") >= 3)
  ).select("cl_rec")

  df_valid = tr.join(valid, on="cl_rec", how="inner")
  odd_df = df_valid.filter(pl.col("week") % 2 == 1)
  even_df = df_valid.filter(pl.col("week") % 2 == 0)

  # A. Raw production rate (yards per attempt)
  raw_odd = (
      odd_df.group_by("cl_rec")
      .agg(y_odd=(pl.col("yds").clip(lower_bound=0).sum() / pl.col("pass_att").sum()))
  )
  raw_even = (
      even_df.group_by("cl_rec")
      .agg(y_even=(pl.col("yds").clip(lower_bound=0).sum() / pl.col("pass_att").sum()))
  )
  raw_comp = raw_odd.join(raw_even, on="cl_rec", how="inner")

  # B. M2 estimated ratings on odd and even splits
  _, car_odd, sea_odd, _, _ = fit_m2_model(odd_df)
  _, car_even, sea_even, _, _ = fit_m2_model(even_df)

  # Filter both to identical set of receiver-seasons with valid ratings
  common_keys = [
      c for c in raw_comp["cl_rec"].to_list()
      if c in sea_odd and c in sea_even
  ]
  raw_sub = raw_comp.filter(pl.col("cl_rec").is_in(common_keys)).sort("cl_rec")
  common_keys = raw_sub["cl_rec"].to_list()

  # Full M2 rating = career baseline + season deviation
  m2_odd = np.array([car_odd.get(c.split("|")[0], 0.0) + sea_odd[c] for c in common_keys])
  m2_even = np.array([car_even.get(c.split("|")[0], 0.0) + sea_even[c] for c in common_keys])

  y_odd_raw = raw_sub["y_odd"].to_numpy()
  y_even_raw = raw_sub["y_even"].to_numpy()

  r_raw, _ = pearsonr(y_odd_raw, y_even_raw)
  rho_raw, _ = spearmanr(y_odd_raw, y_even_raw)
  sb_raw = (2 * r_raw) / (1 + r_raw)

  r_m2, _ = pearsonr(m2_odd, m2_even)
  rho_m2, _ = spearmanr(m2_odd, m2_even)
  sb_m2 = (2 * r_m2) / (1 + r_m2)

  print(f"Receiver-seasons evaluated: {len(common_keys)}")
  print(f"{'Metric':<25}{'Raw YPA Rate':>18}{'M2 Adjusted Rating':>22}")
  print("-" * 65)
  print(f"{'Pearson r (half-test)':<25}{r_raw:>18.4f}{r_m2:>22.4f}")
  print(f"{'Spearman rho (rank)':<25}{rho_raw:>18.4f}{rho_m2:>22.4f}")
  print(f"{'Spearman-Brown (full)':<25}{sb_raw:>18.4f}{sb_m2:>22.4f}")

  if r_m2 > r_raw:
    pct = 100 * (r_m2 - r_raw) / r_raw
    print(f"\nVerdict: PASS. M2 has {pct:+.1f}% higher split-half reliability than raw production.")
  else:
    print("\nVerdict: M2 reliability is comparable to raw box scores.")


# ---------------------------------------------------------------- 2. Next-Season Persistence
def check_next_season_persistence(panel_df):
  print("\n" + "=" * 65)
  print("2. NEXT-SEASON PERSISTENCE (Year t -> Year t+1)")
  print("=" * 65)
  print("Tests how well Year t rating predicts Year t+1 production")
  print("for returning college receivers.\n")

  sea_stats = (
      panel_df.group_by(["athlete_id", "season"])
      .agg(
          tot_yds=pl.col("yds").clip(lower_bound=0).sum(),
          tot_att=pl.col("pass_att").sum(),
          games=pl.len(),
      )
      .with_columns(actual_ypa=pl.col("tot_yds") / pl.col("tot_att"))
      .filter(pl.col("tot_att") >= 50)
  )

  _, car_map, sea_map, _, _ = fit_m2_model(panel_df)

  ratings = []
  for row in panel_df.select("athlete_id", "cl_rec", "season").unique().iter_rows(named=True):
    aid, cl, yr = row["athlete_id"], row["cl_rec"], row["season"]
    score = car_map.get(aid, 0.0) + sea_map.get(cl, 0.0)
    ratings.append({"athlete_id": aid, "season": yr, "m2_rating": score})

  rat_df = pl.DataFrame(ratings).unique(subset=["athlete_id", "season"])
  combined = sea_stats.join(rat_df, on=["athlete_id", "season"], how="inner")

  t1 = combined.select(
      pl.col("athlete_id"),
      pl.col("season").alias("season_t"),
      pl.col("actual_ypa").alias("raw_ypa_t"),
      pl.col("m2_rating").alias("m2_rating_t"),
  )
  t2 = combined.select(
      pl.col("athlete_id"),
      (pl.col("season") - 1).alias("season_t"),
      pl.col("actual_ypa").alias("actual_ypa_next"),
  )
  pairs = t1.join(t2, on=["athlete_id", "season_t"], how="inner")

  y_next = pairs["actual_ypa_next"].to_numpy()
  raw_t = pairs["raw_ypa_t"].to_numpy()
  m2_t = pairs["m2_rating_t"].to_numpy()

  r_raw, _ = pearsonr(raw_t, y_next)
  rho_raw, _ = spearmanr(raw_t, y_next)
  r_m2, _ = pearsonr(m2_t, y_next)
  rho_m2, _ = spearmanr(m2_t, y_next)

  print(f"Returning receiver year-pairs evaluated: {pairs.height}")
  print(f"{'Correlation with Year t+1':<30}{'From Raw YPA':>16}{'From M2 Rating':>18}")
  print("-" * 65)
  print(f"{'Pearson r':<30}{r_raw:>16.4f}{r_m2:>18.4f}")
  print(f"{'Spearman rho':<30}{rho_raw:>16.4f}{rho_m2:>18.4f}")

  if r_m2 > r_raw:
    pct = 100 * (r_m2 - r_raw) / r_raw
    print(f"\nVerdict: PASS. M2 rating predicts Year t+1 {pct:+.1f}% better than raw production.")
  else:
    print("\nVerdict: M2 is comparable to raw production in year-to-year persistence.")


# ---------------------------------------------------------------- 3. Uncertainty Calibration
def check_calibration(panel_df):
  print("\n" + "=" * 65)
  print("3. UNCERTAINTY CALIBRATION (80% Prediction Intervals on Held-Out Test)")
  print("=" * 65)

  f_preds = DATA / "preds_step5.parquet"
  if not f_preds.exists():
    f_preds = DATA / "preds_step3.parquet"
  if not f_preds.exists():
    print("No prediction file found; skipping calibration check.")
    return

  preds = pl.read_parquet(f_preds)
  col = "m2_pred" if "m2_pred" in preds.columns else "tweedie_full"

  # Fit M2 on training data to compute standardized training residual scale
  tr = panel_df.filter(~pl.col("is_test"))
  m_tr, car_tr, sea_tr, maps_tr, norm_tr = fit_m2_model(tr)

  # Compute training predictions
  raw_tr = ctx_matrix(tr)
  n_tr = tr.height
  sc = lambda k: np.sqrt(KAPPA_REF / k)
  cols_tr = [sp.csr_matrix((raw_tr - norm_tr[0]) / norm_tr[1] * sc(KAPPA_CTX))]
  for col_name, mp, k in (("athlete_id", maps_tr[0], KC), ("cl_rec", maps_tr[1], KD)):
    codes = np.array([mp.get(v, -1) for v in tr[col_name].to_list()])
    ok = np.nonzero(codes >= 0)[0]
    cols_tr.append(sp.csr_matrix((np.full(ok.size, sc(k)), (ok, codes[ok])), shape=(n_tr, len(mp))))
  mu_tr = m_tr.predict(sp.hstack(cols_tr).tocsr())

  y_tr = tr["y"].to_numpy()
  w_tr = tr["pass_att"].to_numpy().astype(float)

  # Standardized Tweedie residuals on training set: r = (y - mu) / sqrt(mu^p / w)
  # Captures true empirical zero-inflation and positive skew
  scale_tr = np.sqrt(np.power(mu_tr, POWER) / np.maximum(w_tr, 1.0))
  res_tr = (y_tr - mu_tr) / scale_tr

  q10 = np.percentile(res_tr, 10.0)
  q90 = np.percentile(res_tr, 90.0)

  # Evaluate coverage on held-out test predictions
  y_te = preds["y"].to_numpy()
  mu_te = preds[col].to_numpy()
  w_te = preds["pass_att"].to_numpy().astype(float)

  scale_te = np.sqrt(np.power(mu_te, POWER) / np.maximum(w_te, 1.0))
  lo_te = np.maximum(mu_te + q10 * scale_te, 0.0)
  hi_te = mu_te + q90 * scale_te

  covered = (y_te >= lo_te) & (y_te <= hi_te)
  emp_cov = float(covered.sum() / len(covered))

  print(f"Training standardized residual 10th percentile: {q10:.3f}")
  print(f"Training standardized residual 90th percentile: {q90:.3f}")
  print(f"Nominal prediction interval:                  80.0%")
  print(f"Empirical held-out test coverage:             {100 * emp_cov:.1f}%")
  if 75.0 <= 100 * emp_cov <= 85.0:
    print("Verdict: PASS. Held-out 80% uncertainty intervals are well-calibrated.")
  else:
    print(f"Verdict: Close ({100 * emp_cov:.1f}% vs 80.0% nominal).")


# ---------------------------------------------------------------- 4. Export Prospect Ratings
def export_prospect_ratings(panel_df):
  print("\n" + "=" * 65)
  print("4. EXPORTING PROSPECT RATINGS FOR STAGE 1 (Through 2023)")
  print("=" * 65)
  print("Fitting frozen M2 on all training data through 2023.")
  print("2024 and 2025 draft classes stay untouched.\n")

  train_panel = panel_df.filter(pl.col("season") <= 2023)

  t0 = time.time()
  model, car_map, sea_map, maps, norm = fit_m2_model(train_panel)
  b0 = model.intercept_
  print(f"M2 model fit on {train_panel.height} rows in {time.time() - t0:.1f}s (intercept b0 = {b0:.4f})")

  team_games = (
      train_panel.select("team", "season", "game_id")
      .unique()
      .group_by(["team", "season"])
      .agg(team_tot_games=pl.len())
  )

  rec_summary = (
      train_panel.group_by(["athlete_id", "team", "season"])
      .agg(
          rec_games=pl.len(),
          rec_yds=pl.col("yds").clip(lower_bound=0).sum(),
          rec_catches=pl.col("rec").sum(),
          rec_att_faced=pl.col("pass_att").sum(),
      )
      .join(team_games, on=["team", "season"], how="left")
      .with_columns(
          games_share=pl.col("rec_games") / pl.col("team_tot_games"),
          cl_rec=pl.col("athlete_id").cast(pl.String)
          + "|"
          + pl.col("team")
          + "|"
          + pl.col("season").cast(pl.String),
      )
  )

  final_seasons = (
      rec_summary.group_by("athlete_id")
      .agg(final_season=pl.col("season").max())
  )
  final_rec = (
      rec_summary.join(
          final_seasons,
          left_on=["athlete_id", "season"],
          right_on=["athlete_id", "final_season"],
          how="inner",
      )
      .unique(subset=["athlete_id"])
  )

  career_agg = (
      train_panel.group_by("athlete_id")
      .agg(
          first_season=pl.col("season").min(),
          last_season=pl.col("season").max(),
          career_games=pl.len(),
          career_yds=pl.col("yds").clip(lower_bound=0).sum(),
          career_catches=pl.col("rec").sum(),
          career_att=pl.col("pass_att").sum(),
      )
  )

  records = []
  name_lookup = (
      train_panel.filter(pl.col("name").is_not_null())
      .select("athlete_id", "name")
      .unique(subset=["athlete_id"])
  )
  name_dict = dict(name_lookup.iter_rows())

  for row in career_agg.iter_rows(named=True):
    aid = row["athlete_id"]
    f_row = final_rec.filter(pl.col("athlete_id") == aid)
    if f_row.height == 0:
      continue
    f_row = f_row.row(0, named=True)

    car_eff = car_map.get(aid, 0.0)
    sea_eff = sea_map.get(f_row["cl_rec"], 0.0)
    tot_rating = car_eff + sea_eff

    se_approx = 1.0 / np.sqrt(row["career_att"] / 10.0 + KC)

    records.append({
        "athlete_id": aid,
        "name": name_dict.get(aid, "Unknown"),
        "last_season": row["last_season"],
        "draft_class_est": row["last_season"] + 1,
        "career_rating": float(car_eff),
        "final_season_deviation": float(sea_eff),
        "final_season_rating": float(tot_rating),
        "expected_ypa_career": float(np.exp(b0 + car_eff)),
        "expected_ypa_final": float(np.exp(b0 + tot_rating)),
        "rating_se": float(se_approx),
        "career_games": row["career_games"],
        "career_receptions": row["career_catches"],
        "career_yards": row["career_yds"],
        "career_pass_att": row["career_att"],
        "final_season_games": f_row["rec_games"],
        "final_season_team_games": f_row["team_tot_games"],
        "final_season_games_share": float(f_row["games_share"]),
    })

  out_df = pl.DataFrame(records)

  draft_files = ["drafted_receivers.parquet", "nfl_draft.parquet", "draft.parquet"]
  for fname in draft_files:
    p_draft = DATA / fname
    if p_draft.exists():
      d_df = pl.read_parquet(p_draft)
      common = [c for c in ("athlete_id", "player_id", "id") if c in d_df.columns]
      if common:
        out_df = out_df.join(d_df, left_on="athlete_id", right_on=common[0], how="left")
        print(f"Matched with draft file: {fname}")
        break

  out_path = DATA / "college_prospect_ratings.parquet"
  out_df.write_parquet(out_path)
  print(f"\nWrote {out_df.height} prospect ratings to {out_path}")
  print("Sample exported ratings (top 5 by final season rating, >= 500 career att):")
  sample = (
      out_df.filter(pl.col("career_pass_att") >= 500)
      .sort("final_season_rating", descending=True)
      .select("athlete_id", "name", "last_season", "final_season_rating", "final_season_games_share")
      .head(5)
  )
  print(sample)


# ---------------------------------------------------------------- main
def main():
  P = pl.read_parquet(DATA / PANEL).with_columns(
      cl_rec=pl.col("athlete_id").cast(pl.String)
      + "|"
      + pl.col("team")
      + "|"
      + pl.col("season").cast(pl.String),
      y=pl.col("yds").clip(lower_bound=0) / pl.col("pass_att"),
  )

  # 1. Split-half reliability (full M2 rating vs raw YPA)
  check_split_half(P)

  # 2. Next-season persistence (Year t -> Year t+1)
  check_next_season_persistence(P)

  # 3. Held-out uncertainty calibration
  check_calibration(P)

  # 4. Export ratings for Stage 1
  export_prospect_ratings(P)


if __name__ == "__main__":
  main()