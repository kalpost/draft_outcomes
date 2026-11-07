"""
Stage 1.5: Play Style Archetype & Biomechanical Adjustment (`stage15_athletics.py`)
===================================================================================
1. Classifies every WR prospect into a continuous play-style vector
   `[P(Boundary X), P(Flanker Z), P(Interior Slot)]`.
2. Computes archetype-conditioned athletic multipliers `(med_mult_15, ceil_mult_15)`
   handling the Contested-Catch Bullying Loophole, Undersized Non-Burner Slot Trap,
   and Sub-185 lb Frame Fragility.
3. Runs the out-of-sample Missing-40 residual diagnostic.
"""

from __future__ import annotations

import numpy as np

from config import MISSING_FORTY_MED_PEN


def classify_prospect_play_style(row: dict) -> dict:
    """
    100% Data-Driven Play Style Archetype Classification [P(X), P(Z), P(Slot)].
    """
    ypc = float(row.get("college_ypc") or 14.0)
    td_rate = float(row.get("college_td_per_rec") or 0.12)
    ht = float(row["ht_in"]) if row.get("ht_in") is not None else 72.5
    wt = float(row["wt_lbs"]) if row.get("wt_lbs") is not None else 198.0
    bmi = float(row["bmi"]) if row.get("bmi") is not None else 26.5
    forty = float(row["forty_sec"]) if row.get("forty_sec") is not None else None
    agility = float(row["agility_sum"]) if row.get("agility_sum") is not None else None
    burst = float(row["burst_score"]) if row.get("burst_score") is not None else 100.0

    # 1. BOUNDARY X (Outside Isolation, Physical Frame, Contested/Red-Zone Alpha)
    x_score = (
        (ht - 72.3) * 0.52
        + (wt - 202.0) * 0.045
        + (bmi - 26.4) * 0.55
        + (td_rate - 0.12) * 7.5
        + (burst - 100.0) * 0.04
        + (ypc - 14.0) * 0.16
    )

    # 2. FLANKER Z (Movement Alpha / Dynamic Route-Runner & Vertical Separator)
    speed_signal = (4.48 - forty) * 7.5 if forty is not None else 0.15
    z_score = (
        0.35
        + (ypc - 13.8) * 0.28
        + speed_signal
        + (burst - 100.0) * 0.03
        - max(0.0, ht - 74.2) * 0.45
        - max(0.0, 70.2 - ht) * 0.45
        - max(0.0, bmi - 27.6) * 0.50
    )

    # 3. INTERIOR SLOT (Underneath Target Depth + Compact Frame + Short-Area Quickness)
    agil_bonus = (11.18 - agility) * 3.2 if agility is not None else 0.0
    slot_score = (
        (13.6 - ypc) * 0.52
        + (71.5 - ht) * 0.55
        + (195.0 - wt) * 0.025
        + agil_bonus
        - (td_rate - 0.12) * 3.0
    )

    scores = np.array([x_score, z_score, slot_score], dtype=float)
    exp_s = np.exp((scores - np.max(scores)) / 1.20)
    probs = exp_s / np.sum(exp_s)

    labels = ["Boundary X", "Flanker Z", "Interior Slot"]
    primary_idx = int(np.argmax(probs))

    return {
        "primary": labels[primary_idx],
        "p_x": round(float(probs[0]), 3),
        "p_z": round(float(probs[1]), 3),
        "p_slot": round(float(probs[2]), 3),
        "vector": np.round(probs, 4),
    }


def compute_stage15_athletic_multiplier(row: dict, arch: dict, c_grade: float) -> tuple[float, float]:
    """
    Computes Archetype-Conditioned Stage 1.5 Multipliers (med_mult, ceil_mult).
    """
    p_x = arch["p_x"]
    p_z = arch["p_z"]
    p_slot = arch["p_slot"]
    p_perim = p_x + p_z

    ht = float(row["ht_in"]) if row.get("ht_in") is not None else 72.0
    wt = float(row["wt_lbs"]) if row.get("wt_lbs") is not None else 196.0
    bmi = float(row["bmi"]) if row.get("bmi") is not None else 26.2
    forty = row.get("forty_sec")
    burst = float(row["burst_score"]) if row.get("burst_score") is not None else 100.0
    agility = row.get("agility_sum")
    peak_rec_g = float(row.get("peak_rec_per_g") or 4.8)
    opp_elo = float(row.get("mean_opp_elo") or 1510.0)
    top_def_ratio = float(row.get("vs_top_def_ratio") or 1.0)

    med_adj = 0.0
    ceil_adj = 0.0

    # 1. Speed Conditioning by Play-Style Archetype
    if forty is not None:
        f = float(forty)
        if f <= 4.38:
            med_adj += 0.065 * p_perim + 0.035 * p_slot
            ceil_adj += 0.11 * p_perim + 0.06 * p_slot
        elif f <= 4.42:
            med_adj += 0.045 * p_perim + 0.025 * p_slot
            ceil_adj += 0.08 * p_perim + 0.045 * p_slot
        elif f <= 4.48:
            med_adj += 0.015 * p_perim + 0.01 * p_slot
            ceil_adj += 0.03 * p_perim + 0.02 * p_slot
        elif f <= 4.51:
            med_adj -= 0.025 * p_perim + 0.015 * p_slot
            ceil_adj -= 0.035 * p_perim + 0.02 * p_slot
        else:
            slow_gap = min(0.28, f - 4.49)
            base_pen = slow_gap * 1.15

            verified_separator = (top_def_ratio >= 1.03 and opp_elo >= 1535.0) or (
                agility is not None and float(agility) <= 11.28
            )
            jumpball_comp = 0.0
            if wt >= 215.0 and ht >= 74.0 and burst >= 103.0 and verified_separator:
                jumpball_comp = min(0.65, ((wt - 210.0) / 22.0) * 0.32 + ((burst - 100.0) / 15.0) * 0.32)

            slot_agil_comp = 0.0
            if agility is not None and float(agility) <= 11.08:
                slot_agil_comp = min(0.70, (11.15 - float(agility)) * 2.2)

            effective_keep = max(jumpball_comp * p_x, slot_agil_comp * p_slot)
            net_pen = base_pen * (1.0 - min(0.72, effective_keep))

            if agility is None and f >= 4.52:
                net_pen += 0.085

            if f >= 4.55 and burst < 95.0:
                net_pen += 0.075

            med_adj -= net_pen * (1.05 * p_perim + 0.65 * p_slot)
            ceil_adj -= net_pen * (1.25 * p_perim + 0.75 * p_slot)
    else:
        if not row.get("is_optout_or_injured_final"):
            med_adj -= MISSING_FORTY_MED_PEN
            ceil_adj -= MISSING_FORTY_MED_PEN * 1.1

    # 2. Undersized + Non-Burner Slot Trap
    if ht <= 70.5:
        short_in = 70.8 - ht
        is_fast_enough = (forty is not None and float(forty) <= 4.43) or (
            agility is not None and float(agility) <= 10.98
        )
        if not is_fast_enough:
            med_adj -= min(0.24, 0.08 + short_in * 0.055)
            ceil_adj -= min(0.26, 0.09 + short_in * 0.060)

    # 3. Severe Weight / Frame Fragility Penalty
    if wt < 185.0:
        wt_deficit = 185.0 - wt
        frag_pen = min(0.28, (wt_deficit / 10.0) ** 1.35 * 0.075)
        if c_grade >= 95.2 and opp_elo >= 1600.0:
            frag_pen *= 0.35
        med_adj -= frag_pen
        ceil_adj -= frag_pen * 0.90
    elif bmi >= 27.0 and ht >= 72.5 and (forty is not None and float(forty) <= 4.50):
        med_adj += 0.03 * p_x
        ceil_adj += 0.045 * p_x

    # 4. Explosive Burst & Catch-Radius Modulation
    if burst >= 105.0:
        b_boost = min(0.055, (burst - 100.0) * 0.0038)
        med_adj += b_boost * (0.8 * p_x + 0.6 * p_z + 0.3 * p_slot)
        ceil_adj += b_boost * 1.30 * (0.9 * p_x + 0.7 * p_z + 0.4 * p_slot)
    elif burst < 94.0:
        med_adj -= 0.045 * p_perim
        ceil_adj -= 0.065 * p_perim

    # 5. Short-Area Agility & Route Separation
    if agility is not None:
        ag = float(agility)
        if ag <= 11.05:
            a_boost = min(0.06, (11.15 - ag) * 0.20)
            med_adj += a_boost * (0.4 * p_x + 0.8 * p_z + 1.1 * p_slot)
            ceil_adj += a_boost * 1.15 * (0.4 * p_x + 0.8 * p_z + 1.1 * p_slot)
        elif ag >= 11.45 and forty is not None and float(forty) <= 4.43:
            med_adj -= 0.045
            ceil_adj -= 0.055

    # 6. Workout-Warrior Dampener for Low-Reception Decoys (< 3.8 rec/g)
    if peak_rec_g < 3.8 and ceil_adj > 0:
        damp = max(0.20, (peak_rec_g - 2.0) / 1.8)
        med_adj *= damp
        ceil_adj *= damp

    med_mult = float(np.clip(1.0 + med_adj, 0.62, 1.20))
    ceil_mult = float(np.clip(1.0 + ceil_adj, 0.62, 1.22))
    return med_mult, ceil_mult


def diagnose_missing_forty(rows, s1_p50_arr, y_all, train_mask, k_shrink: float = 30.0) -> None:
    """
    Tests whether lacking a 40 time (combine AND pro day) predicts NFL output beyond the college model.
    """
    idx = np.where(train_mask)[0]
    if len(idx) == 0:
        return
    resid = np.sqrt(y_all[idx]) - np.sqrt(np.maximum(s1_p50_arr[idx], 0.0))
    miss = np.array([rows[i].get("forty_sec") is None for i in idx])
    inj = np.array([bool(rows[i].get("is_optout_or_injured_final")) for i in idx])
    logpick = np.log(np.array([max(1.0, float(rows[i]["pick"])) for i in idx]))
    rng = np.random.default_rng(7)

    def summarize(label: str, m: np.ndarray) -> None:
        n = int(m.sum())
        if n < 5:
            print(f"  {label:<32} n={n} (too few)")
            return
        r = resid[m]
        boots = [rng.choice(r, n).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {label:<32} n={n:<4} mean resid={r.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")

    print("\n" + "=" * 112)
    print("MISSING-40 DIAGNOSTIC (Stage 1 out-of-sample residuals, sqrt-yards scale; negative = underperformed model)")
    print("=" * 112)
    summarize("Tested (has 40)", ~miss)
    summarize("No 40 - all", miss)
    summarize("No 40 - healthy final season", miss & ~inj)
    summarize("No 40 - injured/opt-out final", miss & inj)

    X = np.column_stack([np.ones(len(idx)), miss.astype(float), logpick, inj.astype(float)])

    def miss_coef(Xm: np.ndarray, y: np.ndarray) -> float:
        return float(np.linalg.lstsq(Xm, y, rcond=None)[0][1])

    if miss.sum() >= 5:
        beta = miss_coef(X, resid)
        boots = []
        for _ in range(1000):
            b = rng.integers(0, len(idx), len(idx))
            boots.append(miss_coef(X[b], resid[b]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  Missing-40 coef | log(pick), injury flag: {beta:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")

    hm = miss & ~inj
    n_h = int(hm.sum())
    if n_h >= 5:
        med_sqrt = float(np.median(np.sqrt(np.maximum(s1_p50_arr[idx][hm], 25.0))))
        shrunk = float(resid[hm].mean()) * n_h / (n_h + k_shrink)
        suggested = max(0.0, -2.0 * shrunk / med_sqrt)
        print(
            f"  Suggested MISSING_FORTY_MED_PEN = {suggested:.3f}  (shrunk toward 0, healthy skips only; "
            f"currently {MISSING_FORTY_MED_PEN:.3f})"
        )
        print("  -> Adopt only if the pick-controlled CI above excludes 0; otherwise leave at 0.0.")
    print("=" * 112)