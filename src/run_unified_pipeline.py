"""
Unified 3-Stage NFL WR Prospect Translation Pipeline (`run_unified_pipeline.py`)
================================================================================
Orchestrates:
  - `stage1_college_engine.py`: Stage 1A M2 College Rating Engine & Outcome/Combine Linker
  - `stage15_athletics.py`: Stage 1.5 Play Style Archetype & Combine/Pro-Day Biomechanics
  - `stage2_team_fit.py`: Stage 2 Year-Specific Ex-Ante NFL Team Fit

Supports Model Freezing & Pure Inference:
  - When `USE_FROZEN_MODEL = False` in `config.py` (or `--retrain` CLI flag):
    Fits Stage 1 feature weights (`learned_w`), feature standardization (`StandardScaler`),
    and Dual-Axis Quantile Regressors (`p10`, `p50`, `p90`) on `EVAL_START_YEAR..EVAL_END_YEAR`,
    runs Leave-One-Class-Out validation, and saves the frozen model to both
    `data/processed/frozen_wr_model.json` (human-readable) and `frozen_wr_model.pkl`.
  - When `USE_FROZEN_MODEL = True` in `config.py` (or `--use-frozen` CLI flag):
    Loads `frozen_wr_model.json` and scores all prospects (including new draft classes
    or `SCORE_UNDRAFTED_CLASS_YEAR` pre-draft college WRs) without retraining.

Usage:
   python3 run_unified_pipeline.py              # Uses config.py settings (trains & freezes by default)
   python3 run_unified_pipeline.py --use-frozen # Scores all prospects using saved frozen_wr_model.json
   python3 run_unified_pipeline.py --retrain    # Forces re-fit on EVAL_START_YEAR..EVAL_END_YEAR & freezes
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings

import numpy as np
import polars as pl
from scipy.optimize import nnls
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

from config import (
    BOOM_YDS_THRESHOLD,
    BUST_YDS_THRESHOLD,
    EVAL_END_YEAR,
    EVAL_START_YEAR,
    FREEZE_ON_TRAIN,
    FROZEN_MODEL_JSON_PATH,
    FROZEN_MODEL_PKL_PATH,
    HELDOUT_START_YEAR,
    MISSING_FORTY_SPREAD,
    RECENT_PROSPECTS_PATH,
    STAGE1_FEATURE_NAMES,
    STAGE1_NNLS_LAMBDA,
    STAGE1_PRIOR_WEIGHTS,
    STAGE1_RANKINGS_PATH,
    STAGE2_AUDIT_CSV_PATH,
    STAGE2_RANKINGS_PATH,
    STARTER_YDS_THRESHOLD,
    USE_FROZEN_MODEL,
    WR_LINKS_FINAL_PATH,
)
from stage1_college_engine import (
    attach_nfl_outcomes_and_combine,
    build_new_m2_college_ratings,
    build_stage1_skill_vector,
    derive_consistent_probabilities,
)
from stage15_athletics import (
    classify_prospect_play_style,
    compute_stage15_athletic_multiplier,
    diagnose_missing_forty,
)
from stage2_team_fit import build_nfl_team_environments, compute_stage2_team_fit

warnings.filterwarnings("ignore")


def save_frozen_model(artifact: dict) -> None:
    """
    Saves the trained Stage 1 weights, scaler statistics, and linear QuantileRegressor
    coefficients to both a portable JSON file (`frozen_wr_model.json`) and a pickle file.
    """
    with open(FROZEN_MODEL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    with open(FROZEN_MODEL_PKL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"      Frozen model saved -> {FROZEN_MODEL_JSON_PATH} (and {FROZEN_MODEL_PKL_PATH.name})")


def load_frozen_model() -> dict:
    """
    Loads the frozen model artifact from `frozen_wr_model.json` (or `.pkl`).
    """
    if FROZEN_MODEL_JSON_PATH.exists():
        with open(FROZEN_MODEL_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    if FROZEN_MODEL_PKL_PATH.exists():
        with open(FROZEN_MODEL_PKL_PATH, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(
        f"Frozen model not found at {FROZEN_MODEL_JSON_PATH}. "
        f"Run `python3 run_unified_pipeline.py --retrain` first to fit and freeze the model."
    )


def _predict_linear_qr(X_mat: np.ndarray, q_params: dict) -> np.ndarray:
    coef = np.array(q_params["coef"], dtype=float)
    intercept = float(q_params["intercept"])
    return np.clip(X_mat @ coef + intercept, 0.0, 4450.0)


def fit_and_evaluate_pipeline(use_frozen: bool | None = None) -> pl.DataFrame:
    if use_frozen is None:
        if "--use-frozen" in sys.argv:
            use_frozen = True
        elif "--retrain" in sys.argv:
            use_frozen = False
        else:
            use_frozen = USE_FROZEN_MODEL

    mode_str = "FROZEN MODEL INFERENCE MODE" if use_frozen else f"TRAIN & FREEZE MODE ({EVAL_START_YEAR}–{EVAL_END_YEAR})"
    print("=" * 112)
    print(f"UNIFIED 3-STAGE NFL WR TRANSLATION MODEL [{mode_str}]")
    print("=" * 112)

    links_df = pl.read_parquet(WR_LINKS_FINAL_PATH)
    df = build_new_m2_college_ratings(links_df)
    df = attach_nfl_outcomes_and_combine(df)
    env_map = build_nfl_team_environments()

    rows = df.to_dicts()
    yrs_arr = np.array([int(r["draft_year"]) for r in rows], dtype=int)
    y_all = np.array([float(r.get("nfl_actual_y1_3") or 0.0) for r in rows], dtype=float)
    y_sqrt = np.sqrt(np.maximum(0.0, y_all))
    train_mask = (yrs_arr >= EVAL_START_YEAR) & (yrs_arr <= EVAL_END_YEAR)

    X_raw = np.array([build_stage1_skill_vector(r) for r in rows], dtype=float)
    m2_pk_arr = np.array([float(r["m2_peak"]) for r in rows], dtype=float)

    s1_p10_arr = np.zeros(len(rows))
    s1_p50_arr = np.zeros(len(rows))
    s1_p90_arr = np.zeros(len(rows))

    if use_frozen and (FROZEN_MODEL_JSON_PATH.exists() or FROZEN_MODEL_PKL_PATH.exists()):
        artifact = load_frozen_model()
        scaler_mean = np.array(artifact["scaler_mean"], dtype=float)
        scaler_scale = np.array(artifact["scaler_scale"], dtype=float)
        learned_w = np.array(artifact["learned_w"], dtype=float)
        idx_mean = float(artifact["idx_mean"])
        idx_std = float(artifact["idx_std"])
        pk_mean = float(artifact["pk_mean"])
        pk_std = float(artifact["pk_std"])

        X_std = (X_raw - scaler_mean) / np.where(scaler_scale == 0, 1.0, scaler_scale)
        raw_skill_index = X_std @ learned_w
        z_m2 = (raw_skill_index - idx_mean) / idx_std
        z_pk_m2 = (m2_pk_arr - pk_mean) / pk_std
        z_ceiling = 0.60 * z_m2 + 0.40 * (
            0.55 * z_pk_m2 + 0.25 * X_std[:, 5] + 0.20 * X_std[:, 4]
        )

        pos_convex_med = np.maximum(0.0, z_m2 + 0.10) ** 1.55
        X_med = np.column_stack([z_m2, pos_convex_med])
        pos_convex_ceil = np.maximum(0.0, z_ceiling + 0.10) ** 1.55
        X_ceil = np.column_stack([z_ceiling, pos_convex_ceil])

        s1_p10_arr = _predict_linear_qr(X_med, artifact["qr_p10"])
        s1_p50_arr = _predict_linear_qr(X_med, artifact["qr_p50"])
        s1_p90_arr = _predict_linear_qr(X_ceil, artifact["qr_p90"])

        w_str = ", ".join(f"{n}={w:.1%}" for n, w in zip(STAGE1_FEATURE_NAMES, learned_w))
        print(f"[4/5] Loaded Frozen Stage 1 Weights (trained on {artifact.get('train_window')}): {w_str}")
    else:
        # 4A. Learn Empirical Stage 1 Feature Weights on Completed Cohorts (EVAL_START_YEAR..EVAL_END_YEAR)
        scaler = StandardScaler()
        scaler.fit(X_raw[train_mask])
        X_std = scaler.transform(X_raw)

        prior_w = np.array(STAGE1_PRIOR_WEIGHTS, dtype=float)
        lam_reg = float(STAGE1_NNLS_LAMBDA)
        X_aug = np.vstack([X_std[train_mask], np.sqrt(lam_reg) * np.eye(X_std.shape[1])])
        y_aug = np.concatenate([
            (y_sqrt[train_mask] - np.mean(y_sqrt[train_mask])) / np.std(y_sqrt[train_mask]),
            np.sqrt(lam_reg) * prior_w,
        ])
        learned_w, _ = nnls(X_aug, y_aug)
        learned_w = learned_w / np.sum(learned_w)

        w_str = ", ".join(f"{n}={w:.1%}" for n, w in zip(STAGE1_FEATURE_NAMES, learned_w))
        print(f"[4/5] Learned Empirical Stage 1 Weights ({EVAL_START_YEAR}–{EVAL_END_YEAR}): {w_str}")

        # 4B. Separate Consistency/Median Index (`z_m2`) vs. Peak/Ceiling Index (`z_ceiling`)
        raw_skill_index = X_std @ learned_w
        idx_mean = float(np.mean(raw_skill_index[train_mask]))
        idx_std = max(float(np.std(raw_skill_index[train_mask])), 1e-6)
        z_m2 = (raw_skill_index - idx_mean) / idx_std

        pk_mean = float(np.mean(m2_pk_arr[train_mask]))
        pk_std = max(float(np.std(m2_pk_arr[train_mask])), 1e-6)
        z_pk_m2 = (m2_pk_arr - pk_mean) / pk_std

        z_ceiling = 0.60 * z_m2 + 0.40 * (
            0.55 * z_pk_m2
            + 0.25 * X_std[:, 5]  # SOS_Pie_Dominator
            + 0.20 * X_std[:, 4]  # Teammate_Pie_Credit
        )

        pos_convex_med = np.maximum(0.0, z_m2 + 0.10) ** 1.55
        X_med = np.column_stack([z_m2, pos_convex_med])

        pos_convex_ceil = np.maximum(0.0, z_ceiling + 0.10) ** 1.55
        X_ceil = np.column_stack([z_ceiling, pos_convex_ceil])

        # Out-of-sample Leave-One-Class-Out predictions for evaluation classes
        eval_years = sorted(list(set(yrs_arr[train_mask].tolist())))
        for hyr in eval_years:
            tr_m = train_mask & (yrs_arr != hyr)
            te_m = yrs_arr == hyr
            for tau, out_arr, alpha_val, X_mat in [
                (0.08, s1_p10_arr, 0.012, X_med),
                (0.52, s1_p50_arr, 0.012, X_med),
                (0.91, s1_p90_arr, 0.008, X_ceil),
            ]:
                qr = QuantileRegressor(quantile=tau, alpha=alpha_val, solver="highs")
                qr.fit(X_mat[tr_m], y_all[tr_m])
                out_arr[te_m] = np.clip(qr.predict(X_mat[te_m]), 0.0, 4450.0)

        # Full-cohort QuantileRegressors for held-out classes AND frozen model artifact
        qr_models: dict[str, dict] = {}
        heldout_mask = ~train_mask
        for key_name, tau, out_arr, alpha_val, X_mat in [
            ("qr_p10", 0.08, s1_p10_arr, 0.012, X_med),
            ("qr_p50", 0.52, s1_p50_arr, 0.012, X_med),
            ("qr_p90", 0.91, s1_p90_arr, 0.008, X_ceil),
        ]:
            qr_full = QuantileRegressor(quantile=tau, alpha=alpha_val, solver="highs")
            qr_full.fit(X_mat[train_mask], y_all[train_mask])
            qr_models[key_name] = {
                "quantile": tau,
                "alpha": alpha_val,
                "coef": [float(c) for c in qr_full.coef_],
                "intercept": float(qr_full.intercept_),
            }
            if np.any(heldout_mask):
                out_arr[heldout_mask] = np.clip(qr_full.predict(X_mat[heldout_mask]), 0.0, 4450.0)

        if FREEZE_ON_TRAIN or "--freeze" in sys.argv:
            artifact = {
                "train_window": f"{EVAL_START_YEAR}-{EVAL_END_YEAR}",
                "feature_names": STAGE1_FEATURE_NAMES,
                "scaler_mean": [float(x) for x in scaler.mean_],
                "scaler_scale": [float(x) for x in scaler.scale_],
                "learned_w": [float(x) for x in learned_w],
                "idx_mean": idx_mean,
                "idx_std": idx_std,
                "pk_mean": pk_mean,
                "pk_std": pk_std,
                **qr_models,
            }
            save_frozen_model(artifact)

    c_grades = np.round(50.0 + 49.0 * (1.0 / (1.0 + np.exp(-0.88 * (z_m2 + 0.40)))), 1)

    # Diagnostic: does skipping the 40 (combine + pro day) carry any NFL signal beyond Stage 1?
    diagnose_missing_forty(rows, s1_p50_arr, y_all, train_mask)

    final_records = []
    for i, r in enumerate(rows):
        yr = int(r["draft_year"])
        tm = r["nfl_team"]
        c_grade = float(c_grades[i])

        q_vals = sorted([float(s1_p10_arr[i]), float(s1_p50_arr[i]), float(s1_p90_arr[i])])
        min_floor = 0 if c_grade < 83.0 else int(round((c_grade - 83.0) * 45.0))
        b_p10 = int(round(np.clip(q_vals[0], min_floor, 1300)))
        b_p50 = int(round(np.clip(q_vals[1], b_p10 + 180, 2950)))
        b_p90 = int(round(np.clip(q_vals[2], b_p50 + 700, 4600)))

        # Stage 1.5: Play Style Archetype & Nuanced Athletic Multiplier
        arch = classify_prospect_play_style(r)
        med_mult_15, ceil_mult_15 = compute_stage15_athletic_multiplier(r, arch, c_grade)

        s15_p10 = int(round(np.clip(b_p10 * min(1.10, med_mult_15), 0, 1450)))
        s15_p50 = int(round(np.clip(b_p50 * med_mult_15, s15_p10 + 180, 3150)))
        s15_p90 = int(round(np.clip(b_p90 * ceil_mult_15, s15_p50 + 700, 4800)))

        if r.get("forty_sec") is None:
            s15_p10 = int(round(s15_p10 * (1.0 - MISSING_FORTY_SPREAD)))
            s15_p90 = int(round(s15_p90 * (1.0 + MISSING_FORTY_SPREAD)))

        s15_starter, s15_boom, s15_bust = derive_consistent_probabilities(s15_p10, s15_p50, s15_p90)

        # Stage 2: Ex-Ante NFL Team Fit (with Alpha Target Insulation)
        default_env = {"pass_vol_z": 0.0, "qb_epa_z": 0.0, "vacated_target_pct": 0.24, "incumbents": []}
        t_env = env_map.get((tm, yr), default_env)
        fit_res = compute_stage2_team_fit(arch, t_env, c_grade)
        fit_mult = fit_res["fit_mult"]

        s2_p10 = int(round(np.clip(s15_p10 * (1.0 + fit_res["delta_role"] * 1.2), 0, 1600)))
        s2_p50 = int(round(np.clip(s15_p50 * fit_mult, s2_p10 + 180, 3600)))
        ceil_exp = 1.10 if fit_mult >= 1.0 else 0.85
        s2_p90 = int(round(np.clip(s15_p90 * (fit_mult ** ceil_exp), s2_p50 + 700, 4850)))
        s2_starter, s2_boom, s2_bust = derive_consistent_probabilities(s2_p10, s2_p50, s2_p90)

        actual = float(r.get("nfl_actual_y1_3") or 0.0)
        if c_grade >= 91.5 and s15_boom >= 0.28:
            profile = "Blue-Chip Alpha"
        elif s15_starter >= 0.42:
            profile = "High-Floor Starter"
        elif s15_boom >= 0.18:
            profile = "Boom-or-Bust Upside"
        else:
            profile = "Rotational / Bust Risk"

        final_records.append({
            "wr_id": r["wr_id"],
            "athlete_id": r["athlete_id"],
            "gsis_id": r["gsis_id"],
            "pfr_player_id": r["pfr_player_id"],
            "name": r["pname"],
            "college": r["pcollege"],
            "team": tm,
            "draft_year": yr,
            "round": int(r["round"]),
            "pick": int(r["pick"]),
            "draft_age": round(float(r["draft_age"]), 1) if r.get("draft_age") else None,
            "college_grade": c_grade,
            "new_m2_final": round(float(r["m2_final_eff"]), 3),
            "new_m2_career": round(float(r["m2_career"]), 3),
            "new_m2_peak": round(float(r["m2_peak"]), 3),
            "peak_dominator": round(float(r["peak_dominator"]), 3),
            "standout_freshman_bonus": round(float(r["standout_freshman_bonus"]), 3),
            "peak_rec_per_g": round(float(r["peak_rec_per_g"]), 2),
            "vs_top_def_ratio": round(float(r["vs_top_def_ratio"]), 2),
            "qb_independence_ratio": round(float(r["qb_independence_ratio"]), 2),
            "teammate_wr_boost": round(float(r["teammate_wr_boost"]), 2),
            "college_ypc": round(float(r["college_ypc"]), 1),
            "college_rec_per_att": round(float(r["college_rec_per_att"]), 3),
            "ht_in": round(float(r["ht_in"]), 1) if r.get("ht_in") is not None else None,
            "wt_lbs": round(float(r["wt_lbs"]), 1) if r.get("wt_lbs") is not None else None,
            "bmi": round(float(r["bmi"]), 1) if r.get("bmi") is not None else None,
            "forty_sec": round(float(r["forty_sec"]), 2) if r.get("forty_sec") is not None else None,
            "hass": round(float(r["hass"]), 1) if r.get("hass") is not None else None,
            "burst_score": round(float(r["burst_score"]), 1) if r.get("burst_score") is not None else None,
            "agility_sum": r.get("agility_sum"),
            "archetype_primary": arch["primary"],
            "p_x": arch["p_x"],
            "p_z": arch["p_z"],
            "p_slot": arch["p_slot"],
            "stage1_p10": b_p10,
            "stage1_p50": b_p50,
            "stage1_p90": b_p90,
            "nfl_proj_floor": s15_p10,
            "nfl_proj_median": s15_p50,
            "nfl_proj_ceiling": s15_p90,
            "prob_starter": s15_starter,
            "prob_boom": s15_boom,
            "prob_bust": s15_bust,
            "team_fit_mult": fit_mult,
            "delta_qb_vol": fit_res["delta_qb_vol"],
            "delta_vacancy": fit_res["delta_vacancy"],
            "delta_role": fit_res["delta_role"],
            "team_fit_notes": fit_res["notes"],
            "stage2_p10": s2_p10,
            "stage2_p50": s2_p50,
            "stage2_p90": s2_p90,
            "stage2_prob_starter": s2_starter,
            "stage2_prob_boom": s2_boom,
            "stage2_prob_bust": s2_bust,
            "p50_shift": s2_p50 - s15_p50,
            "missing_forty": int(r.get("forty_sec") is None),
            "nfl_actual_y1_3": round(actual, 1),
            "actual_y1_3": round(actual, 1),
            "s15_error": round(abs(s15_p50 - actual), 1),
            "s2_error": round(abs(s2_p50 - actual), 1),
            "prospect_profile": profile,
        })

    out_df = pl.DataFrame(final_records)

    out_df.filter(pl.col("draft_year") <= EVAL_END_YEAR).write_parquet(STAGE1_RANKINGS_PATH)
    out_df.write_parquet(STAGE2_RANKINGS_PATH)
    out_df.filter(pl.col("draft_year") >= HELDOUT_START_YEAR).write_parquet(RECENT_PROSPECTS_PATH)
    out_df.write_csv(STAGE2_AUDIT_CSV_PATH)

    print(f"\n[5/5] Saved unified rankings to:")
    print(f"  - {STAGE2_RANKINGS_PATH} ({len(out_df)} total prospects, {out_df['draft_year'].min()}–{out_df['draft_year'].max()})")
    print(f"  - {STAGE1_RANKINGS_PATH} ({len(out_df.filter(pl.col('draft_year') <= EVAL_END_YEAR))} historical evaluation prospects, {EVAL_START_YEAR}–{EVAL_END_YEAR})")
    print(f"  - {RECENT_PROSPECTS_PATH} ({len(out_df.filter(pl.col('draft_year') >= HELDOUT_START_YEAR))} held-out recent prospects, {HELDOUT_START_YEAR}+)")

    # ==========================================================================
    # FORMAL SCORECARD: STAGE 1 vs STAGE 1.5 vs STAGE 2 (WITH RAW BRIER & BSS)
    # ==========================================================================
    ev = out_df.filter((pl.col("draft_year") >= EVAL_START_YEAR) & (pl.col("draft_year") <= EVAL_END_YEAR))
    if len(ev) > 0:
        actuals = ev["nfl_actual_y1_3"].to_numpy()
        y_st = (actuals >= STARTER_YDS_THRESHOLD).astype(float)
        y_bm = (actuals >= BOOM_YDS_THRESHOLD).astype(float)
        y_bu = (actuals < BUST_YDS_THRESHOLD).astype(float)

        clim_st = float(np.mean((np.mean(y_st) - y_st) ** 2))
        clim_bm = float(np.mean((np.mean(y_bm) - y_bm) ** 2))
        clim_bu = float(np.mean((np.mean(y_bu) - y_bu) ** 2))

        print("\n" + "=" * 118)
        print(
            f"EVALUATION SCORECARD ({EVAL_START_YEAR}–{EVAL_END_YEAR}: {len(ev)} PROSPECTS "
            f"| CLIMATOLOGY BRIER: St={clim_st:.3f}, Bm={clim_bm:.3f}, Bu={clim_bu:.3f})"
        )
        print("=" * 118)
        print(f"{'Stage / Model Layer':<34} {'MAE':<8} {'MedAE':<8} {'10-90 Cov':<10} {'Starter Brier (BSS)':<22} {'Boom Brier (BSS)':<20} {'Bust Brier (BSS)'}")
        print("-" * 118)

        for label, c_p10, c_p50, c_p90 in [
            ("Stage 1: New M2 College Engine", "stage1_p10", "stage1_p50", "stage1_p90"),
            ("Stage 1.5: + Play Style & Combine", "nfl_proj_floor", "nfl_proj_median", "nfl_proj_ceiling"),
            ("Stage 2: + Ex-Ante NFL Team Fit", "stage2_p10", "stage2_p50", "stage2_p90"),
        ]:
            p10s = ev[c_p10].to_numpy()
            p50s = ev[c_p50].to_numpy()
            p90s = ev[c_p90].to_numpy()
            errs = np.abs(p50s - actuals)
            cov = float(np.mean((actuals >= p10s) & (actuals <= p90s)) * 100.0)

            probs = [derive_consistent_probabilities(p10s[j], p50s[j], p90s[j]) for j in range(len(ev))]
            ps = np.array([p[0] for p in probs])
            pb = np.array([p[1] for p in probs])
            pbu = np.array([p[2] for p in probs])

            br_s = float(np.mean((ps - y_st) ** 2))
            br_b = float(np.mean((pb - y_bm) ** 2))
            br_u = float(np.mean((pbu - y_bu) ** 2))

            bss_s = (1.0 - br_s / clim_st) * 100.0
            bss_b = (1.0 - br_b / clim_bm) * 100.0
            bss_u = (1.0 - br_u / clim_bu) * 100.0

            print(
                f"{label:<34} {np.mean(errs):<8.1f} {np.median(errs):<8.1f} {cov:>6.1f}%    "
                f"{br_s:.3f} ({bss_s:>+5.1f}%)       {br_b:.3f} ({bss_b:>+5.1f}%)     {br_u:.3f} ({bss_u:>+5.1f}%)"
            )

    print("\n" + "=" * 128)
    print(f"TOP 25 OVERALL PROSPECTS ({out_df['draft_year'].min()}–{out_df['draft_year'].max()} STANDOUT-FRESHMAN & PEAK-VS-CONSISTENCY M2 ENGINE)")
    print("=" * 128)
    print(f"{'Rk':<4} {'Prospect':<22} {'Class':<6} {'College':<15} {'Grade':<7} {'Archetype':<14} {'S1.5 p50':<9} {'S1.5 p90':<9} {'Team Fit':<9} {'S2 p50':<8} {'Actual'}")
    print("-" * 128)
    top25 = out_df.sort("nfl_proj_median", descending=True).head(25)
    for idx, r in enumerate(top25.iter_rows(named=True)):
        act_str = "Held-Out" if r["draft_year"] >= HELDOUT_START_YEAR else f"{int(r['nfl_actual_y1_3']):,}"
        print(
            f"{idx+1:<4} {r['name']:<22} {r['draft_year']:<6} {str(r['college'])[:14]:<15} "
            f"{r['college_grade']:<7.1f} {r['archetype_primary']:<14} "
            f"{r['nfl_proj_median']:<9,d} {r['nfl_proj_ceiling']:<9,d} "
            f"{r['team_fit_mult']:<8.2f}x {r['stage2_p50']:<8,d} {act_str}"
        )
    print("=" * 128)

    print("\n" + "=" * 128)
    print(f"TOP 15 HELD-OUT RECENT PROSPECTS ({HELDOUT_START_YEAR}–{out_df['draft_year'].max()} CLASSES: ZERO DRAFT CAPITAL)")
    print("=" * 128)
    print(f"{'Rk':<4} {'Prospect':<22} {'Class':<6} {'College':<15} {'Grade':<7} {'Archetype':<14} {'S1.5 p50':<9} {'S1.5 p90':<9} {'Team Fit':<9} {'S2 p50':<8} {'NFL Team'}")
    print("-" * 128)
    rec15 = out_df.filter(pl.col("draft_year") >= HELDOUT_START_YEAR).sort("nfl_proj_median", descending=True).head(15)
    for idx, r in enumerate(rec15.iter_rows(named=True)):
        print(
            f"{idx+1:<4} {r['name']:<22} {r['draft_year']:<6} {str(r['college'])[:14]:<15} "
            f"{r['college_grade']:<7.1f} {r['archetype_primary']:<14} "
            f"{r['nfl_proj_median']:<9,d} {r['nfl_proj_ceiling']:<9,d} "
            f"{r['team_fit_mult']:<8.2f}x {r['stage2_p50']:<8,d} {r['team']}"
        )
    print("=" * 128)

    return out_df


if __name__ == "__main__":
    fit_and_evaluate_pipeline()