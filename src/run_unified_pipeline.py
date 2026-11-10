"""
Unified 3-Stage NFL WR Prospect Translation Pipeline (`run_unified_pipeline.py`)
================================================================================
Orchestrates:
  - `stage1_college_engine.py`: Scheme-Normalized (`true_non_star_oth_ypa`), On-Field Per-Game
    Dominator, & Top-2-Season Peak-Weighted Career M2 Engine
  - `stage15_athletics.py`: Three-Channel Play Style & Biomechanical Layer (Pure Speed + HaSS + Gated Prototype Synergy)
  - `stage2_team_fit.py`: Stage 2 Year-Specific Ex-Ante NFL Team Fit

Key Upgrades:
  1. Uncapped Dynamic Ceiling Curve (`z_ceiling`, `b_p90`, `s15_p90`):
     - Restores upper-tail convexity (`pos_convex_ceil = np.maximum(0.0, z_ceiling + 0.12) ** 1.44`,
       `tau_90 = 0.90, alpha = 0.009`) and raises ceiling caps (`b_p90` up to `4,350`, `s15_p90`
       up to `4,820`, `s2_p90` up to `4,950`) so elite alphas (Justin Jefferson, Malik Nabers,
       Jaxon Smith-Njigba, Julio Jones, A.J. Green, Tetairoa McMillan) and speed/size-speed outliers
       (Matthew Golden, Brian Thomas Jr., Jameson Williams) display their full upper-tail range.
  2. Monotonic Synchronization Between `college_grade` and Stage 1.5 Yardage (`s15_hit`):
     - Computes the final `college_grade` from `z_unified` (blending `z_m2 + 0.25 * ath_ceil_arr`
       with standardized Stage 1.5 starter expectation `z_s15_hit`).

Usage:
   python3 run_unified_pipeline.py --retrain    # Fits on 2013–2022 & freezes to frozen_wr_model.json
   python3 run_unified_pipeline.py --use-frozen # Scores all prospects using saved frozen_wr_model.json
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
    compute_athletic_upside_index,
    compute_stage15_athletic_multiplier,
    diagnose_missing_forty,
)
from stage2_team_fit import build_nfl_team_environments, compute_stage2_team_fit

warnings.filterwarnings("ignore")

# Structural bounds on the 7 Stage 1 weights:
# [M2_Final_Eff, M2_Career, Underclass_Freshman_Bonus, SOS_Competition, Teammate_Pie_Credit, SOS_Pie_Dominator, Consistency_Resilience]
W_MIN_BOUNDS = np.array([0.24, 0.22, 0.08, 0.08, 0.09, 0.08, 0.05], dtype=float)
W_MAX_BOUNDS = np.array([0.34, 0.30, 0.17, 0.16, 0.17, 0.16, 0.12], dtype=float)


def save_frozen_model(artifact: dict) -> None:
    with open(FROZEN_MODEL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    with open(FROZEN_MODEL_PKL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"      Frozen model saved -> {FROZEN_MODEL_JSON_PATH} (and {FROZEN_MODEL_PKL_PATH.name})")


def load_frozen_model() -> dict:
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
    ath_ceil_arr = np.array([compute_athletic_upside_index(r) for r in rows], dtype=float)

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

        z_ceiling = 0.54 * z_m2 + 0.46 * (
            0.42 * z_pk_m2
            + 0.28 * X_std[:, 5]  # SOS_Pie_Dominator
            + 0.18 * X_std[:, 4]  # Teammate / Solo Alpha Credit
            + 0.12 * np.maximum(0.0, ath_ceil_arr)
        )

        pos_convex_med = np.maximum(0.0, z_m2 + 0.10) ** 1.32
        X_med = np.column_stack([z_m2, pos_convex_med])
        pos_convex_ceil = np.maximum(0.0, z_ceiling + 0.12) ** 1.44
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
        learned_w = np.clip(learned_w, W_MIN_BOUNDS, W_MAX_BOUNDS)
        learned_w = learned_w / np.sum(learned_w)

        w_str = ", ".join(f"{n}={w:.1%}" for n, w in zip(STAGE1_FEATURE_NAMES, learned_w))
        print(f"[4/5] Learned Empirical Stage 1 Weights ({EVAL_START_YEAR}–{EVAL_END_YEAR}): {w_str}")

        # 4B. Separate Consistency/Floor Index (`z_m2`) vs. Dynamic Ceiling Index (`z_ceiling`)
        raw_skill_index = X_std @ learned_w
        idx_mean = float(np.mean(raw_skill_index[train_mask]))
        idx_std = max(float(np.std(raw_skill_index[train_mask])), 1e-6)
        z_m2 = (raw_skill_index - idx_mean) / idx_std

        pk_mean = float(np.mean(m2_pk_arr[train_mask]))
        pk_std = max(float(np.std(m2_pk_arr[train_mask])), 1e-6)
        z_pk_m2 = (m2_pk_arr - pk_mean) / pk_std

        z_ceiling = 0.54 * z_m2 + 0.46 * (
            0.42 * z_pk_m2
            + 0.28 * X_std[:, 5]  # SOS_Pie_Dominator
            + 0.18 * X_std[:, 4]  # Teammate / Solo Alpha Credit
            + 0.12 * np.maximum(0.0, ath_ceil_arr)
        )

        pos_convex_med = np.maximum(0.0, z_m2 + 0.10) ** 1.32
        X_med = np.column_stack([z_m2, pos_convex_med])

        pos_convex_ceil = np.maximum(0.0, z_ceiling + 0.12) ** 1.44
        X_ceil = np.column_stack([z_ceiling, pos_convex_ceil])

        # Out-of-sample Leave-One-Class-Out predictions for evaluation classes
        eval_years = sorted(list(set(yrs_arr[train_mask].tolist())))
        for hyr in eval_years:
            tr_m = train_mask & (yrs_arr != hyr)
            te_m = yrs_arr == hyr
            for tau, out_arr, alpha_val, X_mat in [
                (0.08, s1_p10_arr, 0.014, X_med),
                (0.51, s1_p50_arr, 0.013, X_med),
                (0.90, s1_p90_arr, 0.009, X_ceil),
            ]:
                qr = QuantileRegressor(quantile=tau, alpha=alpha_val, solver="highs")
                qr.fit(X_mat[tr_m], y_all[tr_m])
                out_arr[te_m] = np.clip(qr.predict(X_mat[te_m]), 0.0, 4450.0)

        # Full-cohort QuantileRegressors for held-out classes AND frozen model artifact
        qr_models: dict[str, dict] = {}
        heldout_mask = ~train_mask
        for key_name, tau, out_arr, alpha_val, X_mat in [
            ("qr_p10", 0.08, s1_p10_arr, 0.014, X_med),
            ("qr_p50", 0.51, s1_p50_arr, 0.013, X_med),
            ("qr_p90", 0.90, s1_p90_arr, 0.009, X_ceil),
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

    c_grades_s1 = np.round(50.0 + 48.5 * (1.0 / (1.0 + np.exp(-0.85 * (z_m2 + 0.35)))), 1)

    diagnose_missing_forty(rows, s1_p50_arr, y_all, train_mask)

    final_records = []
    for i, r in enumerate(rows):
        yr = int(r["draft_year"])
        tm = r["nfl_team"]
        c_grade_init = float(c_grades_s1[i])

        q_vals = sorted([float(s1_p10_arr[i]), float(s1_p50_arr[i]), float(s1_p90_arr[i])])
        min_floor = 0 if c_grade_init < 83.0 else int(round((c_grade_init - 83.0) * 40.0))
        b_p10 = int(round(np.clip(q_vals[0], min_floor, 1180)))
        b_p50 = int(round(np.clip(q_vals[1], b_p10 + 180, 2980)))
        b_p90 = int(round(np.clip(q_vals[2], b_p50 + 720, 4350)))

        # Stage 1.5: Play Style Archetype & Three-Channel Biomechanical Multiplier
        arch = classify_prospect_play_style(r)
        med_mult_15, ceil_mult_15 = compute_stage15_athletic_multiplier(r, arch, c_grade_init)

        s15_p10 = int(round(np.clip(b_p10 * min(1.08, med_mult_15), 0, 1280)))
        s15_p50 = int(round(np.clip(b_p50 * med_mult_15, s15_p10 + 180, 3280)))
        s15_p90 = int(round(np.clip(b_p90 * ceil_mult_15, s15_p50 + 750, 4820)))

        if r.get("forty_sec") is None:
            s15_p10 = int(round(s15_p10 * (1.0 - MISSING_FORTY_SPREAD)))
            s15_p90 = int(round(s15_p90 * (1.0 + MISSING_FORTY_SPREAD)))

        s15_starter, s15_boom, s15_bust = derive_consistent_probabilities(s15_p10, s15_p50, s15_p90)
        s15_hit = int(round(0.48 * s15_p50 + 0.52 * s15_p90))
        s15_miss = int(round(0.74 * s15_p10 + 0.26 * s15_p50))

        # SYNCHRONIZED FINAL PROSPECT GRADE (`c_grade`)
        z_s15_hit = (float(s15_hit) - 1880.0) / 620.0
        z_unified = 0.52 * (float(z_m2[i]) + 0.26 * float(ath_ceil_arr[i])) + 0.48 * z_s15_hit
        c_grade = round(float(np.clip(50.0 + 48.5 * (1.0 / (1.0 + np.exp(-0.84 * (z_unified + 0.30)))), 50.0, 99.0)), 1)

        # Stage 2: Ex-Ante NFL Team Fit (with Alpha Target Insulation)
        default_env = {"pass_vol_z": 0.0, "qb_epa_z": 0.0, "vacated_target_pct": 0.24, "incumbents": []}
        t_env = env_map.get((tm, yr), default_env)
        fit_res = compute_stage2_team_fit(arch, t_env, c_grade)
        fit_mult = fit_res["fit_mult"]

        s2_p10 = int(round(np.clip(s15_p10 * (1.0 + fit_res["delta_role"] * 1.15), 0, 1400)))
        s2_p50 = int(round(np.clip(s15_p50 * fit_mult, s2_p10 + 180, 3480)))
        ceil_exp = 1.07 if fit_mult >= 1.0 else 0.86
        s2_p90 = int(round(np.clip(s15_p90 * (fit_mult ** ceil_exp), s2_p50 + 750, 4950)))
        s2_starter, s2_boom, s2_bust = derive_consistent_probabilities(s2_p10, s2_p50, s2_p90)
        s2_hit = int(round(0.48 * s2_p50 + 0.52 * s2_p90))
        s2_miss = int(round(0.74 * s2_p10 + 0.26 * s2_p50))

        actual = float(r.get("nfl_actual_y1_3") or 0.0)
        if c_grade >= 91.5 and s15_boom >= 0.26:
            profile = "Blue-Chip Alpha"
        elif s15_starter >= 0.42:
            profile = "High-Floor Starter"
        elif s15_boom >= 0.17:
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
            "pure_speed_score": round(float(r["pure_speed_score"]), 1) if r.get("pure_speed_score") is not None else None,
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
            "nfl_proj_hit": s15_hit,
            "nfl_proj_miss": s15_miss,
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
            "stage2_hit": s2_hit,
            "stage2_miss": s2_miss,
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

    print("\n" + "=" * 134)
    print(f"TOP 25 OVERALL PROSPECTS ({out_df['draft_year'].min()}–{out_df['draft_year'].max()} | SORTED BY COLLEGE GRADE & STARTER 'IF HE HITS' PROJECTION)")
    print("=" * 134)
    print(f"{'Rk':<4} {'Prospect':<22} {'Class':<6} {'College':<15} {'Grade':<7} {'Archetype':<14} {'p10–p90 Range':<15} {'If Hits':<9} {'Team Fit':<9} {'S2 Hits':<9} {'Actual'}")
    print("-" * 134)
    top25 = out_df.sort(["college_grade", "nfl_proj_hit"], descending=True).head(25)
    for idx, r in enumerate(top25.iter_rows(named=True)):
        act_str = "Held-Out" if r["draft_year"] >= HELDOUT_START_YEAR else f"{int(r['nfl_actual_y1_3']):,}"
        rng_str = f"{r['nfl_proj_floor']:,}–{r['nfl_proj_ceiling']:,}"
        print(
            f"{idx+1:<4} {r['name']:<22} {r['draft_year']:<6} {str(r['college'])[:14]:<15} "
            f"{r['college_grade']:<7.1f} {r['archetype_primary']:<14} "
            f"{rng_str:<15} {r['nfl_proj_hit']:<9,d} "
            f"{r['team_fit_mult']:<8.2f}x {r['stage2_hit']:<9,d} {act_str}"
        )
    print("=" * 134)

    print("\n" + "=" * 134)
    print(f"TOP 15 HELD-OUT RECENT PROSPECTS ({HELDOUT_START_YEAR}–{out_df['draft_year'].max()} CLASSES: ZERO DRAFT CAPITAL)")
    print("=" * 134)
    print(f"{'Rk':<4} {'Prospect':<22} {'Class':<6} {'College':<15} {'Grade':<7} {'Archetype':<14} {'p10–p90 Range':<15} {'If Hits':<9} {'Team Fit':<9} {'S2 Hits':<9} {'NFL Team'}")
    print("-" * 134)
    rec15 = out_df.filter(pl.col("draft_year") >= HELDOUT_START_YEAR).sort(["college_grade", "nfl_proj_hit"], descending=True).head(15)
    for idx, r in enumerate(rec15.iter_rows(named=True)):
        rng_str = f"{r['nfl_proj_floor']:,}–{r['nfl_proj_ceiling']:,}"
        print(
            f"{idx+1:<4} {r['name']:<22} {r['draft_year']:<6} {str(r['college'])[:14]:<15} "
            f"{r['college_grade']:<7.1f} {r['archetype_primary']:<14} "
            f"{rng_str:<15} {r['nfl_proj_hit']:<9,d} "
            f"{r['team_fit_mult']:<8.2f}x {r['stage2_hit']:<9,d} {r['team']}"
        )
    print("=" * 134)

    return out_df


if __name__ == "__main__":
    fit_and_evaluate_pipeline()