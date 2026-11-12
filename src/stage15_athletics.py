"""
Stage 1.5: Play Style Archetype & Dynamic Height/Weight-Scaled Biomechanics (`stage15_athletics.py`)
====================================================================================================
Key Upgrades (Dynamic Continuous Scaling Instead of Hardcoded Step-Function Bounds):
  1. Dynamic Height- & Weight-Scaled Agility Expectation (`expected_agil`):
     - Replaces hardcoded agility cutoffs (which previously blocked 6'5", 231 lb Mike Evans for
       running an `11.34s` agility sum against an `11.32s` cliff!) with a continuous size-scaled
       expectation:
         `expected_agil = 11.10 + max(0.0, ht - 71.5) * 0.045 + max(0.0, wt - 198.0) * 0.0055`
     - At `6'5" (77"), 231 lbs`, Mike Evans's `11.34s` is `+0.19s` FASTER than expected for his frame,
       whereas 6'2", 225 lb Treylon Burks's `11.68s` is `-0.38s` SLOWER than expected!

  2. Dynamic Height-Scaled BMI & Press-Coverage Density (`expected_bmi`):
     - Evaluates frame density continuously via `bmi_delta = bmi - (26.0 + max(0.0, ht - 72.5) * 0.12)`:
       * Penalizes tall, rail-thin perimeter WRs with low vertical power (e.g. `6'3", 194 lb`
         Marquess Wilson at `24.2 BMI` and `31"` vertical / `89.2` burst) who get jammed off the line.
       * Rewards high-BMI, high-TD contested-catch boundary technicians (e.g. `6'1", 214 lb`
         DeAndre Hopkins at `28.2 BMI` with `18 TDs` / `20%+` TD rate) with continuous contested-catch
         compensation (`contested_alpha_comp`) without requiring a hardcoded `ht >= 74.0` cliff!

  3. Continuous Prototype Alpha Synergy (`proto_synergy`):
     - Smoothly scales physical prototype credit by height/weight mass, speed/burst/size-adjusted
       agility, and Power-conference reception command (`peak_rec_per_g`), eliminating binary cliffs.
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


def _dynamic_expected_agility(ht: float, wt: float) -> float:
    """
    Expected 20-yard shuttle + 3-cone sum (`agility_sum`) scaled continuously by height and weight:
      - 5'11" (71"), 190 lbs -> 11.10s
      - 6'2"  (74"), 215 lbs -> 11.31s
      - 6'5"  (77"), 231 lbs -> 11.53s (so Mike Evans's 11.34s is +0.19s better than expected!)
    """
    return 11.10 + max(0.0, ht - 71.5) * 0.045 + max(0.0, wt - 198.0) * 0.0055


def _dynamic_expected_forty(ht: float, wt: float) -> float:
    """
    Expected 40-yard dash baseline scaled smoothly by height/weight mass:
      - 5'11", 190 lbs -> 4.47s
      - 6'3",  215 lbs -> 4.50s
      - 6'5",  231 lbs -> 4.53s
    """
    return 4.47 + max(0.0, wt - 198.0) * 0.0014 + max(0.0, ht - 73.0) * 0.0045


def _dynamic_expected_bmi(ht: float) -> float:
    """
    Expected NFL WR BMI scaled by height so 6'3"+ boundary WRs are checked for rail-thin frames
    (e.g. Marquess Wilson at 6'3", 194 lbs = 24.2 BMI vs 26.3 expected).
    """
    return 26.0 + max(0.0, ht - 72.0) * 0.10


def compute_athletic_upside_index(row: dict) -> float:
    """
    Computes a signed, standardized Four-Channel Biomechanical Grade & Ceiling Index (`-1.60` to `+2.50`)
    using continuous, height/weight-dynamic curves rather than hardcoded cliffs.
    """
    forty = row.get("forty_sec")
    hass = row.get("hass")
    burst = row.get("burst_score")
    agility = row.get("agility_sum")
    ht = float(row["ht_in"]) if row.get("ht_in") is not None else 72.2
    wt = float(row["wt_lbs"]) if row.get("wt_lbs") is not None else 198.0
    bmi = float(row["bmi"]) if row.get("bmi") is not None else ((wt * 703.0) / (ht ** 2))
    c_ypc = float(row.get("college_ypc") or 13.8)
    td_rate = float(row.get("college_td_per_rec") or 0.12)
    final_td = float(row.get("final_td") or 7.0)
    opp_elo = float(row.get("mean_opp_elo") or 1480.0)
    pk_dom = float(row.get("raw_peak_dominator") or 0.26)
    pk_rec_g = float(row.get("peak_rec_per_g") or 4.5)
    c_seasons = int(row.get("college_seasons") or 4)
    yr = int(row.get("draft_year") or 2024)

    exp_agil = _dynamic_expected_agility(ht, wt)
    exp_forty = _dynamic_expected_forty(ht, wt)
    exp_bmi = _dynamic_expected_bmi(ht)

    # Channel A: Pure Speed z-signal (height/weight-neutral for 4.29s burners like Matthew Golden)
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
            slow_gap = min(0.28, f - exp_forty)
            if slow_gap > 0:
                pure_speed_z = -slow_gap * 7.8
                # Dynamic Contested-Catch / Red-Zone Technician Compensation (e.g., DeAndre Hopkins:
                # 214 lbs, 28.2 BMI, 18 TDs in final season, 21% TD/rec rate)
                contested_comp = (
                    max(0.0, (bmi - 26.2) * 0.16)
                    + max(0.0, (td_rate - 0.13) * 2.8)
                    + max(0.0, (final_td - 11.0) * 0.035)
                )
                if opp_elo >= 1505.0 and pk_rec_g >= 5.0:
                    pure_speed_z *= max(0.20, 1.0 - min(0.78, contested_comp))
                if agility is not None and float(agility) <= exp_agil - 0.15:
                    pure_speed_z *= 0.25

    # Channel B: Size-Adjusted Speed Score (HaSS) z-signal (Calvin Johnson 140.9, Julio Jones 121.5, Mike Evans 116.5)
    hass_z = 0.0
    if hass is not None:
        h_val = float(hass)
        if h_val >= 101.0:
            hass_z = (h_val - 101.0) / 12.0 + max(0.0, h_val - 115.0) / 13.5
            # Dampen HaSS only if agility is >0.25s slower than size-expected AND burst is <95.5 (Treylon Burks)
            if (agility is not None and float(agility) >= exp_agil + 0.25) or (burst is not None and float(burst) < 95.5):
                hass_z *= 0.22
        else:
            if forty is None or float(forty) >= exp_forty:
                hass_z = (h_val - 101.0) / 18.0

    # Channel C: Size-Scaled Short-Area Route Agility (`exp_agil - agility`)
    agil_z = 0.0
    if agility is not None:
        ag_val = float(agility)
        ag_delta = exp_agil - ag_val  # positive = faster than expected for height/weight!
        if ag_delta >= 0.0:
            agil_z = ag_delta * 2.15 + max(0.0, 10.90 - ag_val) * 1.45
        elif ag_delta <= -0.15:
            agil_z = (ag_delta + 0.15) * 2.50

    # Channel D: Explosive Burst & Catch Radius z-signal
    burst_z = ((float(burst) - 100.0) / 7.8) if burst is not None else 0.0

    # Continuous Prototype NFL WR1 Synergy (no hardcoded 74.0" or 11.32s cliffs!):
    # Smoothly rewards 3-year Power-conference alphas (`c_seasons <= 3`, `pk_rec_g >= 4.8`, `opp_elo >= 1505`)
    # based on their continuous mass above 198 lbs, speed relative to `exp_forty`, and size-adjusted agility!
    proto_synergy = 0.0
    if c_seasons <= 3 and pk_rec_g >= 4.8 and opp_elo >= 1505.0 and pk_dom >= 0.28:
        mass_factor = np.clip((wt - 196.0) / 26.0 + max(0.0, ht - 72.5) / 8.0, 0.0, 1.45)
        f_val = float(forty) if forty is not None else exp_forty
        speed_ok = np.clip((exp_forty + 0.04 - f_val) / 0.12, 0.0, 1.35)
        agil_ok = 1.0 if agility is None else float(np.clip((exp_agil + 0.15 - float(agility)) / 0.30, 0.0, 1.25))
        burst_ok = 1.0 if burst is None else float(np.clip((float(burst) - 94.0) / 8.0, 0.2, 1.30))
        proto_synergy = 0.44 * mass_factor * speed_ok * agil_ok * burst_ok

    if pk_rec_g < 4.0:
        vol_damp = max(0.22, (pk_rec_g - 1.5) / 2.5)
        pure_speed_z = min(0.0, pure_speed_z) if pure_speed_z < 0 else pure_speed_z * vol_damp
        hass_z = min(0.0, hass_z) if hass_z < 0 else hass_z * vol_damp
        agil_z = min(0.0, agil_z) if agil_z < 0 else agil_z * vol_damp

    # Dynamic Frame Fragility & BMI-Relative-to-Height Penalty:
    frame_pen = 0.0
    if wt < 188.0:
        frame_pen += min(0.92, (188.0 - wt) * 0.044)
    # Tall/lanky low-BMI penalty (e.g. Marquess Wilson: 6'3" 194 lbs = 24.2 BMI vs 26.3 expected, plus 89.2 burst!)
    bmi_deficit = max(0.0, exp_bmi - bmi)
    if bmi_deficit > 0.6:
        low_power_mult = 1.45 if (burst is not None and float(burst) < 94.0) else 0.85
        frame_pen += min(0.75, (bmi_deficit - 0.6) * 0.26 * low_power_mult)

    if ht < 70.5 and c_ypc < 12.6:
        frame_pen += min(0.75, (70.5 - ht) * 0.20 + (12.6 - c_ypc) * 0.24)

    # Stiff + Slow/Low-Burst Penalty (scaled to `exp_forty` and `exp_agil` — catches Treylon Burks, spares Mike Evans!)
    if forty is not None and float(forty) >= exp_forty + 0.02:
        if (agility is not None and float(agility) >= exp_agil + 0.22) or (burst is not None and float(burst) < 94.5):
            frame_pen += 0.55

    # Healthy 4-Year Senior Skipping All Testing Penalty (Tyler Johnson 2020)
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
    Computes Archetype-Conditioned Stage 1.5 Multipliers `(med_mult, ceil_mult)`
    using continuous height/weight-dynamic curves.
    """
    p_x = arch["p_x"]
    p_z = arch["p_z"]
    p_slot = arch["p_slot"]
    p_perim = p_x + p_z

    ht = float(row["ht_in"]) if row.get("ht_in") is not None else 72.0
    wt = float(row["wt_lbs"]) if row.get("wt_lbs") is not None else 196.0
    bmi = float(row["bmi"]) if row.get("bmi") is not None else ((wt * 703.0) / (ht ** 2))
    forty = row.get("forty_sec")
    hass = float(row["hass"]) if row.get("hass") is not None else None
    burst = float(row["burst_score"]) if row.get("burst_score") is not None else 100.0
    agility = row.get("agility_sum")
    peak_rec_g = float(row.get("peak_rec_per_g") or 4.8)
    peak_dom = float(row.get("raw_peak_dominator") or 0.28)
    opp_elo = float(row.get("mean_opp_elo") or 1510.0)
    top_def_ratio = float(row.get("vs_top_def_ratio") or 1.0)
    c_ypc = float(row.get("college_ypc") or 13.8)
    td_rate = float(row.get("college_td_per_rec") or 0.12)
    final_td = float(row.get("final_td") or 7.0)
    c_seasons = int(row.get("college_seasons") or 4)
    yr = int(row.get("draft_year") or 2024)

    exp_agil = _dynamic_expected_agility(ht, wt)
    exp_forty = _dynamic_expected_forty(ht, wt)
    exp_bmi = _dynamic_expected_bmi(ht)

    med_adj = 0.0
    ceil_adj = 0.0

    # --------------------------------------------------------------------------
    # 1A. Continuous Pure Speed & Dynamic Size-Expected Speed Curve
    # --------------------------------------------------------------------------
    if forty is not None:
        f = float(forty)
        if f <= 4.46:
            speed_gain = (4.46 - f) * 0.40
            burner_kick = (max(0.0, 4.37 - f) ** 1.18) * 1.32
            med_adj += (speed_gain * 0.65 + burner_kick * 0.35) * (0.95 * p_perim + 0.68 * p_slot)
            ceil_adj += (speed_gain * 1.25 + burner_kick * 1.58) * (1.02 * p_perim + 0.85 * p_slot)
        elif f <= exp_forty + 0.01:
            med_adj += 0.012
            ceil_adj += 0.030
        elif f <= exp_forty + 0.04:
            med_adj -= 0.006 * p_perim
            ceil_adj -= 0.003 * p_perim
        else:
            slow_gap = min(0.26, f - (exp_forty + 0.02))
            base_pen = slow_gap * 1.20

            verified_separator = (top_def_ratio >= 1.00 and opp_elo >= 1505.0) or (
                agility is not None and float(agility) <= exp_agil
            )
            # Dynamic Contested-Catch / Red-Zone Technician Compensation (no hardcoded 74.0" cliff!):
            # Credits dense-frame boundary WRs with elite college TD conversion & volume (e.g. DeAndre Hopkins
            # at 6'1" 214 lbs, 28.2 BMI, 18 TDs, 21% TD/rec rate).
            jumpball_comp = 0.0
            if wt >= 208.0 and bmi >= 26.5 and verified_separator:
                jumpball_comp = min(
                    0.75,
                    max(0.0, (wt - 202.0) / 24.0) * 0.25
                    + max(0.0, (bmi - 26.2) / 3.0) * 0.25
                    + max(0.0, (burst - 96.0) / 14.0) * 0.25
                    + max(0.0, (td_rate - 0.13) * 2.2)
                    + max(0.0, (final_td - 11.0) * 0.025),
                )

            slot_agil_comp = 0.0
            if agility is not None and float(agility) <= exp_agil - 0.05:
                slot_agil_comp = min(0.75, (exp_agil - float(agility)) * 2.2)

            effective_keep = max(jumpball_comp * (p_x + 0.5 * p_z), slot_agil_comp * (p_slot + 0.5 * p_z))
            net_pen = base_pen * (1.0 - min(0.78, effective_keep))

            if f >= exp_forty + 0.05 and burst < 95.0:
                net_pen += 0.085

            med_adj -= net_pen * (1.05 * p_perim + 0.70 * p_slot)
            ceil_adj -= net_pen * (1.20 * p_perim + 0.75 * p_slot)
    else:
        if c_seasons >= 4 and yr <= 2026 and not row.get("is_optout_or_injured_final"):
            med_adj -= 0.095
            ceil_adj -= 0.115
        elif not row.get("is_optout_or_injured_final"):
            med_adj -= MISSING_FORTY_MED_PEN
            ceil_adj -= MISSING_FORTY_MED_PEN * 1.1

    # --------------------------------------------------------------------------
    # 1B. Size-Adjusted Power Freak & Continuous Prototype Synergy (`hass` + Mass)
    # --------------------------------------------------------------------------
    if hass is not None and hass >= 106.0:
        stiff_low_burst = (agility is not None and float(agility) >= exp_agil + 0.22) or (burst < 95.0)
        if not stiff_low_burst:
            hass_excess = min(35.0, hass - 105.0)
            hass_med_bon = (hass_excess / 100.0) * 0.36
            hass_ceil_bon = (hass_excess / 100.0) * 0.62
            med_adj += hass_med_bon * (0.95 * p_x + 0.78 * p_z + 0.42 * p_slot)
            ceil_adj += hass_ceil_bon * (1.02 * p_x + 0.88 * p_z + 0.50 * p_slot)

    # Continuous Prototype 3-Year Alpha Synergy (covers Calvin Johnson, Julio Jones, A.J. Green, Mike Evans,
    # DeAndre Hopkins, Ja'Marr Chase, Justin Jefferson, Tetairoa McMillan, MHJ without hardcoded cliffs!)
    if c_seasons <= 3 and peak_rec_g >= 4.8 and opp_elo >= 1505.0 and peak_dom >= 0.28:
        mass_factor = float(np.clip((wt - 196.0) / 26.0 + max(0.0, ht - 72.5) / 8.0, 0.0, 1.40))
        f_val = float(forty) if forty is not None else exp_forty
        speed_ok = float(np.clip((exp_forty + 0.05 - f_val) / 0.12, 0.15, 1.35))
        agil_ok = 1.0 if agility is None else float(np.clip((exp_agil + 0.15 - float(agility)) / 0.30, 0.0, 1.25))
        burst_ok = float(np.clip((burst - 94.0) / 8.0, 0.25, 1.30))
        proto_bon = 0.065 * mass_factor * speed_ok * agil_ok * burst_ok
        med_adj += min(0.11, proto_bon)
        ceil_adj += min(0.16, proto_bon * 1.45)

    # 2. Compact Slot / Low-YPC Gadget Check
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

    # 3. Dynamic Frame Fragility & Tall/Lanky Low-BMI Penalty (catches 6'3" 194 lb Marquess Wilson!)
    if wt < 188.0:
        wt_deficit = 188.0 - wt
        frag_pen = min(0.26, (wt_deficit / 12.0) ** 1.25 * 0.068)
        med_adj -= frag_pen
        ceil_frag_mult = 0.60 if (forty is not None and float(forty) <= 4.35) else 0.88
        ceil_adj -= frag_pen * ceil_frag_mult

    bmi_deficit = max(0.0, exp_bmi - bmi)
    if bmi_deficit > 0.6:
        low_power_mult = 1.45 if burst < 94.0 else 0.80
        bmi_pen = min(0.18, (bmi_deficit - 0.6) * 0.055 * low_power_mult)
        med_adj -= bmi_pen
        ceil_adj -= bmi_pen * 1.15

    # 4. Explosive Burst & Catch-Radius Modulation
    if burst >= 103.5:
        b_boost = min(0.070, (burst - 100.0) * 0.0040)
        med_adj += b_boost * (0.85 * p_x + 0.65 * p_z + 0.35 * p_slot)
        ceil_adj += b_boost * 1.35 * (0.95 * p_x + 0.78 * p_z + 0.48 * p_slot)
    elif burst < 94.5:
        med_adj -= 0.075 * p_perim + 0.040 * p_slot
        ceil_adj -= 0.095 * p_perim + 0.050 * p_slot

    # 5. Size-Scaled Short-Area Agility (`exp_agil - agility`)
    if agility is not None:
        ag = float(agility)
        ag_delta = exp_agil - ag
        if ag_delta >= 0.05:
            a_boost = min(0.105, ag_delta * 0.16 + max(0.0, 10.90 - ag) * 0.08)
            med_adj += a_boost * (0.5 * p_x + 0.9 * p_z + 1.15 * p_slot)
            ceil_adj += a_boost * 1.45 * (0.6 * p_x + 1.0 * p_z + 1.25 * p_slot)
        elif ag_delta <= -0.18:
            stiff_pen = min(0.14, (-ag_delta - 0.15) * 0.32)
            med_adj -= stiff_pen
            ceil_adj -= stiff_pen * 1.25

    # 6. Workout-Warrior Dampener for Low-Reception Decoys (< 4.0 rec/g)
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