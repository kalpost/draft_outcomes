"""
Stage 1.5: Play Style Archetype & Four-Channel Biomechanical Layer (`stage15_athletics.py`)
===========================================================================================
Key Upgrades Separating True NFL Route Savants, Burners & Physical Alphas from Stiff Compilers:
  1. Four-Channel Biomechanical Upside (`compute_athletic_upside_index` & `compute_stage15_athletic_multiplier`):
     - Channel A — Pure Track Speed (`forty_sec`, height/weight-neutral):
       * Non-linear ceiling expansion below `4.42s` for `4.29s` burners (Matthew Golden, Jaylen Waddle).
     - Channel B — Size-Adjusted Power Freak (`hass`, Height-Adjusted Speed Score):
       * Rewards `6'2"+, 210+ lb` monsters (`HaSS >= 108`: Calvin Johnson `140.9`, DK Metcalf `134.1`,
         Brian Thomas Jr. `122.9`, Julio Jones `121.5`, A.J. Green).
     - Channel C — Short-Area Route Agility (`agility_sum = cone_sec + shuttle_sec`):
       * Previously missing from `compute_athletic_upside_index`! Now explicitly rewards historic
         short-area separators (`agility_sum <= 11.05`, e.g., **Jaxon Smith-Njigba `10.50`**,
         **Odell Beckham Jr. `10.63`**, **Brandin Cooks `10.78`**, **Julio Jones `10.91`**) while
         penalizing stiff hip-sink prospects (`agility_sum >= 11.38`, e.g., **Treylon Burks `11.68`**).
     - Channel D — Gated Prototype NFL WR1 Catch-Radius + Frame Synergy (`proto_synergy`):
       * Strictly for 3-year underclassmen (`college_seasons <= 3`) with `>= 4.8 rec/g` against
         Power schedules (`opp_elo >= 1510.0`).
  2. Stiff + Low-Burst Separation Penalty & Senior Skip-Testing Penalty:
     - Penalizes perimeter WRs who test both slow (`>= 4.50s`) and stiff (`agility >= 11.38` or
       `burst < 95.0`, e.g., Treylon Burks `4.55s, 33" vert, 11.68 agility`), and penalizes
       healthy 4-year seniors who refuse to run a 40 (`Tyler Johnson`).
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


def compute_athletic_upside_index(row: dict) -> float:
    """
    Computes a signed, standardized Four-Channel Biomechanical Grade & Ceiling Index (`-1.60` to `+2.50`)
    combining:
      - Channel A: Pure Track Speed (`forty_sec`, height/weight-neutral for 4.29s burners like Matthew Golden).
      - Channel B: Size-Adjusted Power Speed (`hass`, for 6'5" 239 lb 4.35s Calvin Johnson, Julio Jones, BTJ).
      - Channel C: Short-Area Route Agility (`agility_sum`, for 10.50s Jaxon Smith-Njigba, OBJ, Cooks, Julio,
        while penalizing 11.68s stiff prospects like Treylon Burks).
      - Channel D: Explosive Burst (`burst_score`) & Gated Prototype Synergy.
    """
    forty = row.get("forty_sec")
    hass = row.get("hass")
    burst = row.get("burst_score")
    agility = row.get("agility_sum")
    ht = float(row["ht_in"]) if row.get("ht_in") is not None else 72.2
    wt = float(row["wt_lbs"]) if row.get("wt_lbs") is not None else 198.0
    c_ypc = float(row.get("college_ypc") or 13.8)
    opp_elo = float(row.get("mean_opp_elo") or 1480.0)
    pk_dom = float(row.get("raw_peak_dominator") or 0.26)
    pk_rec_g = float(row.get("peak_rec_per_g") or 4.5)
    c_seasons = int(row.get("college_seasons") or 4)
    yr = int(row.get("draft_year") or 2024)

    # Channel A: Pure Speed z-signal (centered at 4.48s = 0.0; 4.29s -> +2.05z; 4.59s -> -1.05z)
    pure_speed_z = 0.0
    if forty is not None:
        f = float(forty)
        if f <= 4.48:
            pure_speed_z = (4.48 - f) * 7.5 + max(0.0, 4.36 - f) * 8.2
            if wt < 178.0:
                pure_speed_z *= max(0.28, 1.0 - (178.0 - wt) * 0.052)
            if c_ypc < 11.8:
                pure_speed_z *= max(0.30, 1.0 - (11.8 - c_ypc) * 0.35)
        else:
            slow_gap = min(0.28, f - 4.48)
            pure_speed_z = -slow_gap * 8.2
            if agility is not None and float(agility) <= 10.95:
                # Elite short-area agility (e.g. Jaxon Smith-Njigba 10.50, Jarvis Landry/Cooper Kupp)
                # compensates for 4.51-4.53 long speed!
                pure_speed_z *= 0.25

    # Channel B: Size-Adjusted Speed Score (HaSS) z-signal (centered at 101.0 = 0.0; 140.9 -> +2.75z)
    hass_z = 0.0
    if hass is not None:
        h_val = float(hass)
        if h_val >= 101.0:
            hass_z = (h_val - 101.0) / 12.0 + max(0.0, h_val - 115.0) / 13.5
            # Dampen HaSS if the player is stiff (`agility >= 11.45`) and low-burst (`burst < 95.0`)
            # like Treylon Burks (225 lbs, 4.55s -> 105 HaSS, but 33" vert and 11.68 agility!)
            if (agility is not None and float(agility) >= 11.40) or (burst is not None and float(burst) < 95.5):
                hass_z *= 0.20
        else:
            if forty is None or float(forty) >= 4.47:
                hass_z = (h_val - 101.0) / 18.0

    # Channel C: Short-Area Route Agility (`agility_sum`: 10.50 JSN -> +1.55z; 11.68 Treylon Burks -> -0.95z)
    agil_z = 0.0
    if agility is not None:
        ag_val = float(agility)
        if ag_val <= 11.15:
            agil_z = (11.15 - ag_val) * 2.15 + max(0.0, 10.85 - ag_val) * 1.45
        elif ag_val >= 11.35:
            agil_z = -(ag_val - 11.35) * 2.60

    # Channel D: Explosive Burst & Catch Radius z-signal
    burst_z = ((float(burst) - 100.0) / 7.8) if burst is not None else 0.0

    # Gated Prototype NFL WR1 Frame + Speed + Catch-Radius Synergy:
    proto_synergy = 0.0
    if c_seasons <= 3 and pk_rec_g >= 4.8 and opp_elo >= 1510.0 and pk_dom >= 0.28:
        if ht >= 74.0 and wt >= 206.0 and (forty is None or float(forty) <= 4.53):
            if (agility is None or float(agility) <= 11.32) and (burst is None or float(burst) >= 97.5):
                f_val = float(forty) if forty is not None else 4.49
                proto_synergy += 0.42 + max(0.0, 4.50 - f_val) * 2.8 + max(0.0, (ht - 74.0) * 0.09)
        elif wt >= 198.0 and (forty is not None and float(forty) <= 4.45) and (burst is not None and float(burst) >= 102.0):
            proto_synergy += 0.36

    # Dampen workout-warrior bonus if low college reception volume (< 4.0 rec/g, e.g. Stephen Hill 2.15 rec/g)
    if pk_rec_g < 4.0:
        vol_damp = max(0.22, (pk_rec_g - 1.5) / 2.5)
        pure_speed_z = min(0.0, pure_speed_z) if pure_speed_z < 0 else pure_speed_z * vol_damp
        hass_z = min(0.0, hass_z) if hass_z < 0 else hass_z * vol_damp
        agil_z = min(0.0, agil_z) if agil_z < 0 else agil_z * vol_damp

    # Penalties for Frame Fragility, Stiff+Low-Burst Profiles (Treylon Burks), & 4-Year Seniors Skipping Testing (Tyler Johnson)
    frame_pen = 0.0
    if wt < 188.0:
        frame_pen += min(0.92, (188.0 - wt) * 0.044)
    if ht < 70.2 and c_ypc < 12.5:
        frame_pen += min(0.75, (70.2 - ht) * 0.22 + (12.5 - c_ypc) * 0.25)

    # Stiff + Slow/Low-Burst Penalty (e.g. Treylon Burks: 4.55s 40, 33" vert [burst=92.8], 11.68 agility)
    if forty is not None and float(forty) >= 4.50:
        if (agility is not None and float(agility) >= 11.38) or (burst is not None and float(burst) < 95.5):
            frame_pen += 0.55

    # Healthy 4-Year Senior Skipping All Testing Penalty (e.g. Tyler Johnson 2020)
    if forty is None and agility is None and c_seasons >= 4 and yr <= 2026 and not row.get("is_optout_or_injured_final"):
        frame_pen += 0.65

    best_pos_channel = max(pure_speed_z, hass_z, agil_z * 0.90)
    if best_pos_channel > 0.0:
        synergy = 0.24 * (
            min(max(0.0, pure_speed_z), max(0.0, hass_z))
            + min(max(0.0, max(pure_speed_z, hass_z)), max(0.0, agil_z))
        )
        raw_idx = (
            0.78 * best_pos_channel
            + synergy
            + 0.20 * max(-0.8, burst_z)
            + 0.25 * min(0.0, agil_z)
            + proto_synergy
            - frame_pen
        )
    else:
        raw_idx = (
            0.55 * pure_speed_z
            + 0.25 * hass_z
            + 0.25 * agil_z
            + 0.15 * min(0.0, burst_z)
            + proto_synergy
            - frame_pen
        )

    return float(np.clip(raw_idx, -1.60, 2.50))


def compute_stage15_athletic_multiplier(row: dict, arch: dict, c_grade: float) -> tuple[float, float]:
    """
    Computes Archetype-Conditioned Stage 1.5 Multipliers `(med_mult, ceil_mult)`.
    """
    p_x = arch["p_x"]
    p_z = arch["p_z"]
    p_slot = arch["p_slot"]
    p_perim = p_x + p_z

    ht = float(row["ht_in"]) if row.get("ht_in") is not None else 72.0
    wt = float(row["wt_lbs"]) if row.get("wt_lbs") is not None else 196.0
    forty = row.get("forty_sec")
    hass = float(row["hass"]) if row.get("hass") is not None else None
    burst = float(row["burst_score"]) if row.get("burst_score") is not None else 100.0
    agility = row.get("agility_sum")
    peak_rec_g = float(row.get("peak_rec_per_g") or 4.8)
    peak_dom = float(row.get("raw_peak_dominator") or 0.28)
    opp_elo = float(row.get("mean_opp_elo") or 1510.0)
    top_def_ratio = float(row.get("vs_top_def_ratio") or 1.0)
    c_ypc = float(row.get("college_ypc") or 13.8)
    c_seasons = int(row.get("college_seasons") or 4)
    yr = int(row.get("draft_year") or 2024)

    med_adj = 0.0
    ceil_adj = 0.0

    # --------------------------------------------------------------------------
    # 1A. Continuous Pure Speed Curve (Height/Weight-Neutral Track Speed)
    # --------------------------------------------------------------------------
    if forty is not None:
        f = float(forty)
        if f <= 4.46:
            speed_gain = (4.46 - f) * 0.40
            burner_kick = (max(0.0, 4.37 - f) ** 1.18) * 1.32
            med_adj += (speed_gain * 0.65 + burner_kick * 0.35) * (0.95 * p_perim + 0.68 * p_slot)
            ceil_adj += (speed_gain * 1.25 + burner_kick * 1.58) * (1.02 * p_perim + 0.85 * p_slot)
        elif f <= 4.50:
            med_adj += 0.012
            ceil_adj += 0.030
        elif f <= 4.53:
            med_adj -= 0.008 * p_perim
            ceil_adj -= 0.004 * p_perim
        else:
            slow_gap = min(0.26, f - 4.52)
            base_pen = slow_gap * 1.25

            verified_separator = (top_def_ratio >= 1.02 and opp_elo >= 1520.0) or (
                agility is not None and float(agility) <= 11.25
            )
            jumpball_comp = 0.0
            if wt >= 212.0 and ht >= 74.0 and burst >= 101.0 and verified_separator:
                jumpball_comp = min(0.65, ((wt - 208.0) / 22.0) * 0.32 + ((burst - 98.0) / 15.0) * 0.32)

            slot_agil_comp = 0.0
            if agility is not None and float(agility) <= 11.08:
                slot_agil_comp = min(0.75, (11.15 - float(agility)) * 2.4)

            effective_keep = max(jumpball_comp * p_x, slot_agil_comp * (p_slot + 0.5 * p_z))
            net_pen = base_pen * (1.0 - min(0.72, effective_keep))

            if f >= 4.54 and burst < 96.0:
                net_pen += 0.085

            med_adj -= net_pen * (1.05 * p_perim + 0.70 * p_slot)
            ceil_adj -= net_pen * (1.20 * p_perim + 0.75 * p_slot)
    else:
        if c_seasons >= 4 and yr <= 2026 and not row.get("is_optout_or_injured_final"):
            # Healthy 4-year senior skipping Combine/Pro Day 40 (e.g. Tyler Johnson)
            med_adj -= 0.095
            ceil_adj -= 0.115
        elif not row.get("is_optout_or_injured_final"):
            med_adj -= MISSING_FORTY_MED_PEN
            ceil_adj -= MISSING_FORTY_MED_PEN * 1.1

    # --------------------------------------------------------------------------
    # 1B. Size-Adjusted Power Freak & Prototype Catch-Radius Boost (`hass` + Frame)
    # --------------------------------------------------------------------------
    if hass is not None and hass >= 106.0:
        # Don't reward HaSS if stiff (agility >= 11.40) and low-burst (< 95.5) like Treylon Burks
        stiff_low_burst = (agility is not None and float(agility) >= 11.40) or (burst < 95.5)
        if not stiff_low_burst:
            hass_excess = min(35.0, hass - 105.0)
            hass_med_bon = (hass_excess / 100.0) * 0.36
            hass_ceil_bon = (hass_excess / 100.0) * 0.62
            med_adj += hass_med_bon * (0.95 * p_x + 0.78 * p_z + 0.42 * p_slot)
            ceil_adj += hass_ceil_bon * (1.02 * p_x + 0.88 * p_z + 0.50 * p_slot)

    # Gated Prototype 6'2"+ 206+ lb 3-Year Alpha Synergy (Calvin Johnson, Julio Jones, A.J. Green, Tetairoa McMillan, MHJ)
    if c_seasons <= 3 and peak_rec_g >= 4.8 and ht >= 74.0 and wt >= 206.0 and opp_elo >= 1510.0:
        if (forty is None or float(forty) <= 4.53) and (agility is None or float(agility) <= 11.32) and burst >= 97.5:
            f_val = float(forty) if forty is not None else 4.49
            proto_bon = 0.060 + max(0.0, (ht - 74.0) * 0.015) + max(0.0, (4.50 - f_val) * 0.25)
            med_adj += min(0.11, proto_bon)
            ceil_adj += min(0.16, proto_bon * 1.45)

    # 2. Undersized + Non-Burner OR Low-YPC Gadget Slot Trap
    if ht <= 70.5:
        short_in = 70.8 - ht
        is_fast_enough = (forty is not None and float(forty) <= 4.42) or (
            agility is not None and float(agility) <= 10.98
        )
        if not is_fast_enough:
            med_adj -= min(0.24, 0.08 + short_in * 0.055)
            ceil_adj -= min(0.26, 0.09 + short_in * 0.060)
        elif c_ypc < 11.6:
            gadget_pen = min(0.18, (11.6 - c_ypc) * 0.08 + max(0.0, 69.5 - ht) * 0.03)
            med_adj -= gadget_pen
            ceil_adj -= gadget_pen * 1.15

    # 3. Light-Frame Durability & Press-Coverage Penalty (< 188 lbs)
    if wt < 188.0:
        wt_deficit = 188.0 - wt
        frag_pen = min(0.26, (wt_deficit / 12.0) ** 1.25 * 0.068)
        med_adj -= frag_pen
        ceil_frag_mult = 0.60 if (forty is not None and float(forty) <= 4.35) else 0.88
        ceil_adj -= frag_pen * ceil_frag_mult

    # 4. Explosive Burst & Catch-Radius Modulation
    if burst >= 103.5:
        b_boost = min(0.070, (burst - 100.0) * 0.0040)
        med_adj += b_boost * (0.85 * p_x + 0.65 * p_z + 0.35 * p_slot)
        ceil_adj += b_boost * 1.35 * (0.95 * p_x + 0.78 * p_z + 0.48 * p_slot)
    elif burst < 95.0:
        med_adj -= 0.065 * p_perim + 0.035 * p_slot
        ceil_adj -= 0.085 * p_perim + 0.045 * p_slot

    # 5. Short-Area Agility & Route Separation (Historic Boost for JSN 10.50 / OBJ 10.63; Penalty for Burks 11.68)
    if agility is not None:
        ag = float(agility)
        if ag <= 11.08:
            a_boost = min(0.105, (11.15 - ag) * 0.16)
            med_adj += a_boost * (0.5 * p_x + 0.9 * p_z + 1.15 * p_slot)
            ceil_adj += a_boost * 1.45 * (0.6 * p_x + 1.0 * p_z + 1.25 * p_slot)
        elif ag >= 11.38:
            stiff_pen = min(0.14, (ag - 11.32) * 0.32)
            med_adj -= stiff_pen
            ceil_adj -= stiff_pen * 1.25

    # 6. Workout-Warrior Dampener for Low-Reception Decoys (< 4.0 rec/g, e.g. Stephen Hill 2.15 rec/g)
    if peak_rec_g < 4.0 and (med_adj > 0 or ceil_adj > 0):
        damp = max(0.22, (peak_rec_g - 1.5) / 2.5)
        if med_adj > 0:
            med_adj *= damp
        if ceil_adj > 0:
            ceil_adj *= max(0.38, damp)

    med_mult = float(np.clip(1.0 + med_adj, 0.72, 1.20))
    ceil_mult = float(np.clip(1.0 + ceil_adj, 0.75, 1.30))
    return round(med_mult, 4), round(ceil_mult, 4)


def diagnose_missing_forty(
    rows: list[dict],
    s1_p50_arr: np.ndarray,
    y_all: np.ndarray,
    train_mask: np.ndarray,
) -> float:
    idx = np.where(train_mask)[0]
    s_pred = np.sqrt(np.maximum(0.0, s1_p50_arr[idx]))
    s_act = np.sqrt(np.maximum(0.0, y_all[idx]))
    resid = s_act - s_pred

    miss = np.array([1.0 if rows[i].get("forty_sec") is None else 0.0 for i in idx])
    inj = np.array([1.0 if rows[i].get("is_optout_or_injured_final") else 0.0 for i in idx])
    log_pk = np.array([np.log(max(1.0, float(rows[i].get("pick") or 100))) for i in idx])

    rng = np.random.default_rng(42)

    def _boot_ci(arr: np.ndarray) -> tuple[float, float, float]:
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        bs = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(1000)]
        return float(np.mean(arr)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    m_test, l_test, u_test = _boot_ci(resid[miss == 0])
    m_miss, l_miss, u_miss = _boot_ci(resid[miss == 1])
    m_hlth, l_hlth, u_hlth = _boot_ci(resid[(miss == 1) & (inj == 0)])
    m_inj, l_inj, u_inj = _boot_ci(resid[(miss == 1) & (inj == 1)])

    X = np.column_stack([np.ones(len(idx)), miss, log_pk, inj])
    beta, *_ = np.linalg.lstsq(X, resid, rcond=None)
    b_miss = float(beta[1])
    bs_b = []
    for _ in range(1000):
        bi = rng.choice(len(idx), size=len(idx), replace=True)
        b_i, *_ = np.linalg.lstsq(X[bi], resid[bi], rcond=None)
        bs_b.append(float(b_i[1]))
    b_lo, b_hi = float(np.percentile(bs_b, 2.5)), float(np.percentile(bs_b, 97.5))

    n_hlth = int(np.sum((miss == 1) & (inj == 0)))
    shrink = n_hlth / (n_hlth + 20.0)
    mean_s50 = max(float(np.mean(s_pred)), 20.0)
    sug_pen = max(0.0, round(float(-2.0 * shrink * m_hlth / mean_s50), 3))

    print("\n" + "=" * 112)
    print("MISSING-40 DIAGNOSTIC (Stage 1 out-of-sample residuals, sqrt-yards scale; negative = underperformed model)")
    print("=" * 112)
    print(f"  Tested (has 40)                   n={int(np.sum(miss == 0)):<3d}  mean resid={m_test:+.2f}  95% CI [{l_test:+.2f}, {u_test:+.2f}]")
    print(f"  No 40 - all                       n={int(np.sum(miss == 1)):<3d}  mean resid={m_miss:+.2f}  95% CI [{l_miss:+.2f}, {u_miss:+.2f}]")
    print(f"  No 40 - healthy final season      n={n_hlth:<3d}  mean resid={m_hlth:+.2f}  95% CI [{l_hlth:+.2f}, {u_hlth:+.2f}]")
    print(f"  No 40 - injured/opt-out final     n={int(np.sum((miss == 1) & (inj == 1))):<3d}  mean resid={m_inj:+.2f}  95% CI [{l_inj:+.2f}, {u_inj:+.2f}]")
    print(f"  Missing-40 coef | log(pick), injury flag: {b_miss:+.2f}  95% CI [{b_lo:+.2f}, {b_hi:+.2f}]")
    print(f"  Suggested MISSING_FORTY_MED_PEN = {sug_pen:.3f}  (shrunk toward 0, healthy skips only; currently {MISSING_FORTY_MED_PEN:.3f})")
    print("  -> Adopt only if the pick-controlled CI above excludes 0; otherwise leave at 0.0.")
    print("=" * 112)
    return sug_pen