"""
Evaluate Stage 1 Prospect Model Performance on 2012-2022 Cohorts.
Calculates probabilistic Brier Scores, empirical 10-90 coverage, and error metrics.
Thresholds: Boom >= 2,500 yds | Starter >= 1,500 yds | Bust < 500 yds
"""

from pathlib import Path
import numpy as np
import polars as pl

# 1. Flexible path resolution for running from root or src/
candidates = [
    Path("data/processed"),
    Path("../data/processed"),
    Path(".")
]
DATA_DIR = next((d for d in candidates if (d / "stage1_prospect_rankings.parquet").exists()), Path("data/processed"))
RANKINGS_FILE = DATA_DIR / "stage1_prospect_rankings.parquet"
OUTPUT_CSV = DATA_DIR / "stage1_error_analysis.csv"

if not RANKINGS_FILE.exists():
    raise FileNotFoundError(f"Missing {RANKINGS_FILE}. Run stage1_prospect_rankings.py first.")

df = pl.read_parquet(RANKINGS_FILE)

y_actual = df["nfl_actual_y1_3"].to_numpy()
p10 = df["nfl_proj_floor"].to_numpy()
p50 = df["nfl_proj_median"].to_numpy()
p90 = df["nfl_proj_ceiling"].to_numpy()
prob_boom = df["prob_boom"].to_numpy()
prob_bust = df["prob_bust"].to_numpy()

# Support prob_starter with fallback to prob_contributor
if "prob_starter" in df.columns:
    prob_starter = df["prob_starter"].to_numpy()
elif "prob_contributor" in df.columns:
    prob_starter = df["prob_contributor"].to_numpy()
else:
    prob_starter = np.clip(prob_boom * 1.5, 0.10, 0.88)

# 2. Continuous Calibration & Quantile Metrics
errors = np.abs(y_actual - p50)
mae = np.mean(errors)
med_ae = np.median(errors)
rmse = np.sqrt(np.mean((y_actual - p50) ** 2))
coverage = np.mean((y_actual >= p10) & (y_actual <= p90)) * 100
mean_width = np.mean(p90 - p10)

# 3. Probabilistic Brier Scores (Calibrated Thresholds)
# Threshold 1: >= 1,500 yds (Quality Starter)
obs_starter = (y_actual >= 1500).astype(float)
pred_starter = prob_starter
brier_starter = np.mean((pred_starter - obs_starter) ** 2)
clim_starter = np.mean((np.mean(obs_starter) - obs_starter) ** 2)
bss_starter = (1.0 - brier_starter / clim_starter) * 100

# Threshold 2: >= 2,500 yds (Boom / WR1 Alpha)
obs_boom = (y_actual >= 2500).astype(float)
pred_boom = prob_boom
brier_boom = np.mean((pred_boom - obs_boom) ** 2)
clim_boom = np.mean((np.mean(obs_boom) - obs_boom) ** 2)
bss_boom = (1.0 - brier_boom / clim_boom) * 100

# Threshold 3: < 500 yds (Bust / Flop)
obs_bust = (y_actual < 500).astype(float)
pred_bust = prob_bust
brier_bust = np.mean((pred_bust - obs_bust) ** 2)
clim_bust = np.mean((np.mean(obs_bust) - obs_bust) ** 2)
bss_bust = (1.0 - brier_bust / clim_bust) * 100

print("=" * 80)
print(f"STAGE 1 ERROR ANALYSIS & BRIER SCORES (2012-2022: {len(df)} PROSPECTS)")
print("=" * 80)

print("\n1. PROBABILISTIC BRIER SCORES (Lower is Better; Positive BSS = Better Than Climatology)")
print(f"{'Outcome Threshold':<35} {'Base Rate':<15} {'Brier Score':<15} {'Skill vs Clim (BSS)'}")
print("-" * 80)
print(f"{'>= 1,500 yds (Quality Starter)':<35} {np.mean(obs_starter)*100:.1f}%{'':<10} {brier_starter:<15.4f} {bss_starter:+.1f}%")
print(f"{'>= 2,500 yds (Boom / WR1 Alpha)':<35} {np.mean(obs_boom)*100:.1f}%{'':<10} {brier_boom:<15.4f} {bss_boom:+.1f}%")
print(f"{'< 500 yds (Bust / Flop)':<35} {np.mean(obs_bust)*100:.1f}%{'':<10} {brier_bust:<15.4f} {bss_bust:+.1f}%")

print("\n2. CONTINUOUS ERROR & CALIBRATION METRICS")
print("-" * 80)
print(f"Mean Absolute Error (MAE):           {mae:.1f} yards")
print(f"Median Absolute Error:               {med_ae:.1f} yards")
print(f"Root Mean Squared Error (RMSE):      {rmse:.1f} yards")
print(f"10–90 Empirical Coverage:            {coverage:.1f}% (Nominal 80.0%)")
print(f"Mean Interval Width (p90 - p10):     {mean_width:.1f} yards")

# 4. Export audit CSV
audit_df = df.with_columns([
    (pl.col("nfl_actual_y1_3") - pl.col("nfl_proj_median")).alias("error_yds"),
    pl.when((pl.col("nfl_actual_y1_3") >= pl.col("nfl_proj_floor")) & (pl.col("nfl_actual_y1_3") <= pl.col("nfl_proj_ceiling")))
    .then(pl.lit("WITHIN 10-90"))
    .otherwise(pl.lit("BREACH"))
    .alias("coverage_status")
])
audit_df.write_csv(OUTPUT_CSV)
print(f"\nWrote full player-by-player audit ({len(audit_df)} rows) to: {OUTPUT_CSV}")

# 5. Top 10 Closest Hits
name_col = "name_draft" if "name_draft" in df.columns else "name"
print("\n" + "=" * 80)
print("3. TOP 10 CLOSEST PREDICTIONS (Bullseye Model Hits)")
print("=" * 80)
hits = (
    audit_df.with_columns(pl.col("error_yds").abs().alias("abs_err"))
    .sort("abs_err")
    .head(10)
    .select([name_col, "draft_year", "pick", "college_grade", "nfl_proj_median", "nfl_actual_y1_3", "error_yds", "coverage_status"])
)
print(hits)

# 6. Top 8 Outlier Breakouts
print("\n" + "=" * 80)
print("4. TOP 8 OUTLIER BREAKOUTS (Exceeded 90th Percentile Ceiling)")
print("=" * 80)
breakouts = (
    audit_df.filter(pl.col("nfl_actual_y1_3") > pl.col("nfl_proj_ceiling"))
    .with_columns((pl.col("nfl_actual_y1_3") - pl.col("nfl_proj_ceiling")).alias("excess_yds"))
    .sort("excess_yds", descending=True)
    .head(8)
    .select([name_col, "draft_year", "pick", "college_grade", "nfl_proj_ceiling", "nfl_actual_y1_3"])
)
print(breakouts)