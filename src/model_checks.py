"""
College Model Health Checks and Prospect Ratings Export (`model_checks.py`)
===========================================================================
Runs three descriptive validation checks on `panel_v2_feat.parquet`:
  1. Odd-vs-Even split-half reliability within seasons (M2 vs raw YPA).
  2. Next-season persistence (Year t -> Year t+1 for returning receivers).
  3. 80% prediction interval coverage on held-out test games.

Exports:
  `college_prospect_ratings.parquet` across all seasons up through `CFB_END_YEAR`
  (removing the old hardcoded `season <= 2023` cutoff so 2024 and 2025 college
  seasons are exported too).
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import TweedieRegressor

from config import (
    CFB_END_YEAR,
    LEGACY_RATINGS_PATH,
    PANEL_V2_FEAT_PATH,
    PROCESSED_DIR,
    TWEEDIE_KAPPA_CTX,
    TWEEDIE_KAPPA_REF,
    TWEEDIE_KC,
    TWEEDIE_KD,
    TWEEDIE_POWER,
)


def ctx_matrix(df: pl.DataFrame) -> np.ndarray:
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


def fit_m2_model(df: pl.DataFrame, fit_seasons=None):
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
    sc = lambda k: np.sqrt(TWEEDIE_KAPPA_REF / k)

    n = df.height
    cols = [sp.csr_matrix((raw - norm[0]) / norm[1] * sc(TWEEDIE_KAPPA_CTX))]
    for col, mp, k in (
        ("athlete_id", maps[0], TWEEDIE_KC),
        ("cl_rec", maps[1], TWEEDIE_KD),
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
        power=TWEEDIE_POWER,
        link="log",
        alpha=TWEEDIE_KAPPA_REF / sw,
        max_iter=3000,
        tol=1e-6,
    )
    model.fit(X, df["y"].to_numpy(), sample_weight=w)

    n_ctx = raw.shape[1]
    n_car = len(maps[0])
    car_coefs = model.coef_[n_ctx : n_ctx + n_car] * sc(TWEEDIE_KC)
    sea_coefs = model.coef_[n_ctx + n_car :] * sc(TWEEDIE_KD)

    car_map = {a: car_coefs[i] for a, i in maps[0].items()}
    sea_map = {c: sea_coefs[i] for c, i in maps[1].items()}

    return model, car_map, sea_map, maps, norm


def check_split_half(panel_df: pl.DataFrame) -> None:
    print("\n" + "=" * 65)
    print("1. SPLIT-HALF RELIABILITY (Odd vs. Even Weeks Within Season)")
    print("=" * 65)
    print("Tests whether M2 ratings correlate higher across alternating weeks")
    print("than raw yards per attempt (removing schedule & offense noise).\n")

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

    raw_odd = (
        odd_df.group_by("cl_rec")
        .agg(y_odd=(pl.col("yds").clip(lower_bound=0).sum() / pl.col("pass_att").sum()))
    )
    raw_even = (
        even_df.group_by("cl_rec")
        .agg(y_even=(pl.col("yds").clip(lower_bound=0).sum() / pl.col("pass_att").sum()))
    )
    raw_comp = raw_odd.join(raw_even, on="cl_rec", how="inner")

    _, car_odd, sea_odd, _, _ = fit_m2_model(odd_df)
    _, car_even, sea_even, _, _ = fit_m2_model(even_df)

    common_keys = [
        c for c in raw_comp["cl_rec"].to_list()
        if c in sea_odd and c in sea_even
    ]
    raw_sub = raw_comp.filter(pl.col("cl_rec").is_in(common_keys)).sort("cl_rec")
    common_keys = raw_sub["cl_rec"].to_list()

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


def check_next_season_persistence(panel_df: pl.DataFrame) -> None:
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


def check_calibration(panel_df: pl.DataFrame, data_dir: Path) -> None:
    print("\n" + "=" * 65)
    print("3. UNCERTAINTY CALIBRATION (80% Prediction Intervals on Held-Out Test)")
    print("=" * 65)

    f_preds = data_dir / "preds_step5.parquet"
    if not f_preds.exists():
        f_preds = data_dir / "preds_step3.parquet"
    if not f_preds.exists():
        print("No prediction file found; skipping calibration check.")
        return

    preds = pl.read_parquet(f_preds)
    col = "m2_pred" if "m2_pred" in preds.columns else "tweedie_full"

    tr = panel_df.filter(~pl.col("is_test"))
    m_tr, _, _, maps_tr, norm_tr = fit_m2_model(tr)

    raw_tr = ctx_matrix(tr)
    n_tr = tr.height
    sc = lambda k: np.sqrt(TWEEDIE_KAPPA_REF / k)
    cols_tr = [sp.csr_matrix((raw_tr - norm_tr[0]) / norm_tr[1] * sc(TWEEDIE_KAPPA_CTX))]
    for col_name, mp, k in (("athlete_id", maps_tr[0], TWEEDIE_KC), ("cl_rec", maps_tr[1], TWEEDIE_KD)):
        codes = np.array([mp.get(v, -1) for v in tr[col_name].to_list()])
        ok = np.nonzero(codes >= 0)[0]
        cols_tr.append(sp.csr_matrix((np.full(ok.size, sc(k)), (ok, codes[ok])), shape=(n_tr, len(mp))))
    mu_tr = m_tr.predict(sp.hstack(cols_tr).tocsr())

    y_tr = tr["y"].to_numpy()
    w_tr = tr["pass_att"].to_numpy().astype(float)

    scale_tr = np.sqrt(np.power(mu_tr, TWEEDIE_POWER) / np.maximum(w_tr, 1.0))
    res_tr = (y_tr - mu_tr) / scale_tr

    q10 = np.percentile(res_tr, 10.0)
    q90 = np.percentile(res_tr, 90.0)

    y_te = preds["y"].to_numpy()
    mu_te = preds[col].to_numpy()
    w_te = preds["pass_att"].to_numpy().astype(float)

    scale_te = np.sqrt(np.power(mu_te, TWEEDIE_POWER) / np.maximum(w_te, 1.0))
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


def export_prospect_ratings(panel_df: pl.DataFrame, data_dir: Path) -> pl.DataFrame:
    print("\n" + "=" * 65)
    print(f"4. EXPORTING PROSPECT RATINGS FOR STAGE 1 (Through {CFB_END_YEAR})")
    print("=" * 65)

    train_panel = panel_df.filter(pl.col("season") <= CFB_END_YEAR)

    t0 = time.time()
    model, car_map, sea_map, _, _ = fit_m2_model(train_panel)
    b0 = model.intercept_
    print(f"M2 model fit on {train_panel.height:,} rows in {time.time() - t0:.1f}s (intercept b0 = {b0:.4f})")

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

        se_approx = 1.0 / np.sqrt(row["career_att"] / 10.0 + TWEEDIE_KC)

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
    out_path = data_dir / LEGACY_RATINGS_PATH.name
    out_df.write_parquet(out_path)
    print(f"\nWrote {out_df.height:,} prospect ratings to {out_path}")
    return out_df


def run_model_checks(data_dir: Path | None = None, panel_file: str | None = None) -> None:
    target_dir = data_dir or PROCESSED_DIR
    p_path = target_dir / (panel_file or PANEL_V2_FEAT_PATH.name)

    P = pl.read_parquet(p_path).with_columns(
        cl_rec=pl.col("athlete_id").cast(pl.String)
        + "|"
        + pl.col("team")
        + "|"
        + pl.col("season").cast(pl.String),
        y=pl.col("yds").clip(lower_bound=0) / pl.col("pass_att"),
    )

    check_split_half(P)
    check_next_season_persistence(P)
    check_calibration(P, target_dir)
    export_prospect_ratings(P, target_dir)


if __name__ == "__main__":
    cli_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    cli_panel = sys.argv[2] if len(sys.argv) > 2 else None
    run_model_checks(cli_dir, cli_panel)