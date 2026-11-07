"""
Stage 1: NFL Translation (Pre-Draft Information -> NFL Career Outcomes).

Downloads NFL draft and player stats from nflverse, links to
college_prospect_ratings.parquet, and evaluates whether college ratings
improve NFL quantile forecasts over draft slot alone.

Target:
    Log of Year 1-3 NFL receiving yards: log(yds_1_to_3 + 10).

Models evaluated across 2018-2023 draft classes (2024-2025 untouched):
    - Baseline: Draft slot alone (linear quantile regression on log(pick)).
    - Primary: Regularized linear quantile regression (Draft slot + College ratings + Availability).
    - Challenger: Monotone boosted quantile regression (XGBoost with monotone draft constraint).

Keep-or-drop rule:
    Boosting or college metrics count only if improvement over draft slot
    has a 95% bootstrap interval strictly above zero.
"""

import sys
import time
from pathlib import Path
import urllib.request
import re

import numpy as np
import polars as pl
from sklearn.linear_model import QuantileRegressor
import xgboost as xgb

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed")
COLLEGE_FILE = DATA / "college_prospect_ratings.parquet"

NFLVERSE_DRAFT_URL = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.parquet"
NFLVERSE_STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.parquet"

QUANTILES = (0.10, 0.50, 0.90)
N_BOOT = 1000


# ---------------------------------------------------------------- 1. Data Ingestion & Download
def download_stream(url, dest_path):
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    )
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def fetch_nflverse_data():
    DATA.mkdir(parents=True, exist_ok=True)
    draft_path = DATA / "nfl_draft_picks.parquet"
    stats_path = DATA / "nfl_stats_player.parquet"

    if not draft_path.exists():
        print(f"Downloading NFL draft picks from nflverse...")
        download_stream(NFLVERSE_DRAFT_URL, draft_path)
        print(f"   Saved {draft_path.name}")
    else:
        print(f"Using cached draft picks: {draft_path.name}")

    if not stats_path.exists():
        print(f"Downloading player stats from nflverse...")
        download_stream(NFLVERSE_STATS_URL, stats_path)
        print(f"   Saved {stats_path.name}")
    else:
        print(f"Using cached player stats: {stats_path.name}")

    return pl.read_parquet(draft_path), pl.read_parquet(stats_path)


# ---------------------------------------------------------------- 2. Name Normalization & Link
def clean_name(s):
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[.'\-,]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def build_linked_panel(draft_df, stats_df, college_df):
    print("\nLinking NFL draft picks to college prospect ratings...")

    pos_col = "position" if "position" in draft_df.columns else "pos"
    name_col = "pfr_player_name" if "pfr_player_name" in draft_df.columns else "player_name"

    wr_draft = draft_df.filter(pl.col(pos_col) == "WR").with_columns(
        name_clean=pl.col(name_col).map_elements(clean_name, return_dtype=pl.String),
        draft_year=pl.col("season").cast(pl.Int64),
        pick=pl.col("pick").cast(pl.Float64)
    )

    # Use player_display_name (full name, e.g. "DeVonta Smith") instead of player_name ("D.Smith")
    stat_name = "player_display_name" if "player_display_name" in stats_df.columns else (
        "player_name" if "player_name" in stats_df.columns else "player"
    )
    yds_col = "receiving_yards" if "receiving_yards" in stats_df.columns else "rec_yds"

    stats_clean = stats_df.with_columns(
        name_clean=pl.col(stat_name).map_elements(clean_name, return_dtype=pl.String),
        stat_season=pl.col("season").cast(pl.Int64)
    )

    if "season_type" in stats_clean.columns:
        stats_clean = stats_clean.filter(pl.col("season_type") == "REG")

    # Match NFL stats: prefer pfr_player_id -> pfr_id if available, fallback to (name_clean, draft_year)
    has_pfr_ids = (
        "pfr_player_id" in wr_draft.columns and
        "pfr_id" in stats_clean.columns and
        wr_draft["pfr_player_id"].drop_nulls().len() > 0
    )

    if has_pfr_ids:
        matched_stats = (
            wr_draft.select("pfr_player_id", "draft_year")
            .join(
                stats_clean.select("pfr_id", "stat_season", yds_col),
                left_on="pfr_player_id", right_on="pfr_id", how="inner"
            )
            .filter(
                (pl.col("stat_season") >= pl.col("draft_year")) &
                (pl.col("stat_season") <= pl.col("draft_year") + 2)
            )
            .group_by("pfr_player_id")
            .agg(nfl_yds_y1_3=pl.col(yds_col).sum())
        )
        wr_panel = wr_draft.join(matched_stats, on="pfr_player_id", how="left")
    else:
        matched_stats = (
            wr_draft.select("name_clean", "draft_year")
            .join(
                stats_clean.select("name_clean", "stat_season", yds_col),
                on="name_clean", how="inner"
            )
            .filter(
                (pl.col("stat_season") >= pl.col("draft_year")) &
                (pl.col("stat_season") <= pl.col("draft_year") + 2)
            )
            .group_by(["name_clean", "draft_year"])
            .agg(nfl_yds_y1_3=pl.col(yds_col).sum())
        )
        wr_panel = wr_draft.join(matched_stats, on=["name_clean", "draft_year"], how="left")

    wr_panel = wr_panel.with_columns(
        nfl_yds_y1_3=pl.col("nfl_yds_y1_3").fill_null(0.0),
        target_y=(pl.col("nfl_yds_y1_3").fill_null(0.0) + 10.0).log()
    )

    col_clean = college_df.with_columns(
        name_clean=pl.col("name").map_elements(clean_name, return_dtype=pl.String),
        last_yr=pl.col("last_season").cast(pl.Int64)
    )

    linked = wr_panel.join(
        col_clean, on="name_clean", how="inner"
    ).filter(
        (pl.col("draft_year") == pl.col("last_yr") + 1) |
        (pl.col("draft_year") == pl.col("last_yr") + 2)
    ).unique(subset=["name_clean", "draft_year"])

    sample_eval = linked.filter((pl.col("draft_year") >= 2018) & (pl.col("draft_year") <= 2023))
    print(f"   Total drafted WRs in file: {wr_draft.height} | Successfully linked to college: {linked.height}")
    print(f"   2018-2023 sample: {sample_eval.height} players | Mean NFL Yds (Y1-3): {sample_eval['nfl_yds_y1_3'].mean():.1f} | Max: {sample_eval['nfl_yds_y1_3'].max():.1f}")
    return linked


# ---------------------------------------------------------------- 3. Metrics & Scoring
def pinball_loss(y, q_pred, tau):
    diff = y - q_pred
    return np.maximum(tau * diff, (tau - 1) * diff)


def score_quantiles(y, q10, q50, q90):
    l10 = pinball_loss(y, q10, 0.10).mean()
    l50 = pinball_loss(y, q50, 0.50).mean()
    l90 = pinball_loss(y, q90, 0.90).mean()
    mean_pb = (l10 + l50 + l90) / 3.0

    cov = ((y >= q10) & (y <= q90)).mean()
    width = (q90 - q10).mean()
    return {
        "pinball": mean_pb,
        "p10": l10,
        "p50": l50,
        "p90": l90,
        "coverage": cov,
        "width": width
    }


# ---------------------------------------------------------------- 4. Models & Evaluation
def run_stage1_cv(panel):
    print("\n" + "=" * 70)
    print("STAGE 1 EVALUATION: 2018-2023 DRAFT CLASSES (Leave-One-Class-Out)")
    print("=" * 70)

    eval_df = panel.filter(
        (pl.col("draft_year") >= 2018) & (pl.col("draft_year") <= 2023)
    ).sort(["draft_year", "pick"])

    classes = sorted(eval_df["draft_year"].unique().to_list())
    print(f"Classes: {classes} | Total prospect sample: {eval_df.height} players")

    features_lin = [
        "log_pick", "final_season_rating", "career_rating", "final_season_games_share"
    ]
    features_xgb = [
        "pick", "final_season_rating", "career_rating", "final_season_games_share", "rating_se"
    ]

    eval_df = eval_df.with_columns(
        log_pick=pl.col("pick").log()
    )

    preds_base = {q: [] for q in QUANTILES}
    preds_lin = {q: [] for q in QUANTILES}
    preds_xgb = {q: [] for q in QUANTILES}
    y_all = []
    draft_years = []

    # Leave-one-class-out validation across 2018-2023
    for holdout_yr in classes:
        tr = eval_df.filter(pl.col("draft_year") != holdout_yr)
        te = eval_df.filter(pl.col("draft_year") == holdout_yr)

        y_tr, y_te = tr["target_y"].to_numpy(), te["target_y"].to_numpy()
        y_all.extend(y_te)
        draft_years.extend(te["draft_year"].to_list())

        # A. Draft-Slot Baseline (Quantile Regression on log_pick)
        X_tr_base = tr.select("log_pick").to_numpy()
        X_te_base = te.select("log_pick").to_numpy()

        for q in QUANTILES:
            qr_b = QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
            qr_b.fit(X_tr_base, y_tr)
            preds_base[q].extend(qr_b.predict(X_te_base))

        # B. Primary Linear Model (Quantile Regression + College Ratings)
        X_tr_lin = tr.select(features_lin).to_numpy()
        X_te_lin = te.select(features_lin).to_numpy()

        for q in QUANTILES:
            qr_l = QuantileRegressor(quantile=q, alpha=0.01, solver="highs")
            qr_l.fit(X_tr_lin, y_tr)
            preds_lin[q].extend(qr_l.predict(X_te_lin))

        # C. Challenger Boosted Model (XGBoost with Monotone Constraint on Pick)
        # Monotone constraint: pick is -1 (higher pick number = later pick = monotonically lower yards)
        X_tr_xgb = tr.select(features_xgb).to_numpy()
        X_te_xgb = te.select(features_xgb).to_numpy()

        dtr = xgb.DMatrix(X_tr_xgb, label=y_tr)
        dte = xgb.DMatrix(X_te_xgb)

        q_preds_te = []
        for q in QUANTILES:
            params = {
                "objective": "reg:quantileerror",
                "quantile_alpha": q,
                "learning_rate": 0.03,
                "max_depth": 3,
                "min_child_weight": 10,
                "monotone_constraints": (-1, 0, 0, 0, 0),
                "random_state": 42
            }
            bst = xgb.train(params, dtr, num_boost_round=100)
            q_preds_te.append(bst.predict(dte))

        # Sort quantiles to prevent quantile crossing (doc requirement)
        q_mat = np.sort(np.column_stack(q_preds_te), axis=1)
        for i, q in enumerate(QUANTILES):
            preds_xgb[q].extend(q_mat[:, i])

    y_all = np.array(y_all)
    for d in (preds_base, preds_lin, preds_xgb):
        for q in QUANTILES:
            d[q] = np.array(d[q])

    s_base = score_quantiles(y_all, preds_base[0.10], preds_base[0.50], preds_base[0.90])
    s_lin = score_quantiles(y_all, preds_lin[0.10], preds_lin[0.50], preds_lin[0.90])
    s_xgb = score_quantiles(y_all, preds_xgb[0.10], preds_xgb[0.50], preds_xgb[0.90])

    print(f"{'Model':<35}{'Pinball':>10}{'10-90 Cov':>12}{'Width':>10}")
    print("-" * 70)
    print(f"{'Draft-Slot Baseline (Gate 1)':<35}{s_base['pinball']:>10.4f}{s_base['coverage']:>12.2f}{s_base['width']:>10.2f}")
    print(f"{'Primary Linear (Draft + College)':<35}{s_lin['pinball']:>10.4f}{s_lin['coverage']:>12.2f}{s_lin['width']:>10.2f}")
    print(f"{'Boosted Monotone Challenger':<35}{s_xgb['pinball']:>10.4f}{s_xgb['coverage']:>12.2f}{s_xgb['width']:>10.2f}")

    # Clustered Bootstrap over Draft Classes / Players
    print("\n" + "=" * 70)
    print("BOOTSTRAP DIFFERENCES (vs. Draft-Slot Baseline; Negative = Better)")
    print("Keep-or-drop rule: 95% bootstrap interval must strictly exclude zero.")
    print("=" * 70)

    rng = np.random.default_rng(0)
    n = len(y_all)

    loss_base = (
        pinball_loss(y_all, preds_base[0.10], 0.10) +
        pinball_loss(y_all, preds_base[0.50], 0.50) +
        pinball_loss(y_all, preds_base[0.90], 0.90)
    ) / 3.0

    for name, preds_m in [
        ("Primary Linear vs Baseline", preds_lin),
        ("Boosted Monotone vs Baseline", preds_xgb),
        ("Boosted vs Primary Linear", preds_xgb)
    ]:
        ref_loss = loss_base if "Baseline" in name else (
            pinball_loss(y_all, preds_lin[0.10], 0.10) +
            pinball_loss(y_all, preds_lin[0.50], 0.50) +
            pinball_loss(y_all, preds_lin[0.90], 0.90)
        ) / 3.0
        m_loss = (
            pinball_loss(y_all, preds_m[0.10], 0.10) +
            pinball_loss(y_all, preds_m[0.50], 0.50) +
            pinball_loss(y_all, preds_m[0.90], 0.90)
        ) / 3.0

        diff = m_loss - ref_loss
        boot_diffs = []
        for _ in range(N_BOOT):
            idx = rng.choice(n, size=n, replace=True)
            boot_diffs.append(diff[idx].mean())

        pt = diff.mean()
        lo = np.percentile(boot_diffs, 2.5)
        hi = np.percentile(boot_diffs, 97.5)
        verdict = "PASSES (Beats Reference)" if hi < 0 else (
            "FAILS (Worse than Reference)" if lo > 0 else "Not Distinguishable"
        )
        print(f"   {name:<30} diff: {pt:+.4f}  95% CI: [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")

    out_df = eval_df.with_columns(
        base_p10=pl.Series(preds_base[0.10]),
        base_p50=pl.Series(preds_base[0.50]),
        base_p90=pl.Series(preds_base[0.90]),
        lin_p10=pl.Series(preds_lin[0.10]),
        lin_p50=pl.Series(preds_lin[0.50]),
        lin_p90=pl.Series(preds_lin[0.90]),
        xgb_p10=pl.Series(preds_xgb[0.10]),
        xgb_p50=pl.Series(preds_xgb[0.50]),
        xgb_p90=pl.Series(preds_xgb[0.90]),
    )
    out_path = DATA / "stage1_nfl_predictions.parquet"
    out_df.write_parquet(out_path)
    print(f"\nwrote {out_df.height} prospect predictions to {out_path}")


# ---------------------------------------------------------------- main
def main():
    if not COLLEGE_FILE.exists():
        print(f"ERROR: {COLLEGE_FILE} not found. Run model_checks.py first to export ratings.")
        sys.exit(1)

    college_df = pl.read_parquet(COLLEGE_FILE)
    draft_df, stats_df = fetch_nflverse_data()

    panel = build_linked_panel(draft_df, stats_df, college_df)
    run_stage1_cv(panel)


if __name__ == "__main__":
    main()