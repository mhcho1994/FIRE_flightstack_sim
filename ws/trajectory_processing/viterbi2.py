# =====================================================================
# PRIMARY + SECONDARY MULTI-AXIS VITERBI SEGMENTATION
# Replaces the previous single-chain physics-aware Viterbi block.
# =====================================================================

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# PRIMARY STATES
# ---------------------------------------------------------------------
PRIMARY_GROUND = 0
PRIMARY_TAKEOFF = 1
PRIMARY_MISSION = 2
PRIMARY_LANDING = 3

PRIMARY_STATE_NAMES = {
    PRIMARY_GROUND: "ground",
    PRIMARY_TAKEOFF: "takeoff",
    PRIMARY_MISSION: "mission",
    PRIMARY_LANDING: "landing",
}


# ---------------------------------------------------------------------
# SECONDARY STATES: LATERAL
# ---------------------------------------------------------------------
LAT_STRAIGHTLIKE = 0
LAT_TURN = 1

LATERAL_STATE_NAMES = {
    LAT_STRAIGHTLIKE: "straightlike",
    LAT_TURN: "turn",
}

# ---------------------------------------------------------------------
# SECONDARY STATES: LONGITUDINAL
# ---------------------------------------------------------------------
LON_CONST = 0
LON_ACC = 1
LON_DEC = 2

LONGITUDINAL_STATE_NAMES = {
    LON_CONST: "const",
    LON_ACC: "acc",
    LON_DEC: "dec",
}

# ---------------------------------------------------------------------
# SECONDARY STATES: VERTICAL
# ---------------------------------------------------------------------
VER_LEVEL = 0
VER_CLIMB = 1
VER_DESCEND = 2

VERTICAL_STATE_NAMES = {
    VER_LEVEL: "level",
    VER_CLIMB: "climb",
    VER_DESCEND: "descend",
}


# =====================================================================
# HELPERS
# =====================================================================
def _safe_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _soft_pos_indicator(x, center, width):
    return _safe_sigmoid((x - center) / max(width, 1e-6))


def _soft_neg_indicator(x, center, width):
    return _safe_sigmoid((-x - center) / max(width, 1e-6))


def _normalize01(x):
    x = np.asarray(x, dtype=float)
    lo = np.percentile(x, 5)
    hi = np.percentile(x, 95)
    if hi <= lo + 1e-9:
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def _state_runs_from_sequence(seq):
    """
    Convert discrete state sequence to list of (start, end, state),
    using half-open interval [start, end).
    """
    if len(seq) == 0:
        return []

    runs = []
    s = 0
    cur = seq[0]

    for i in range(1, len(seq)):
        if seq[i] != cur:
            runs.append((s, i, cur))
            s = i
            cur = seq[i]

    runs.append((s, len(seq), cur))
    return runs


def _stats_1d(x, prefix):
    x = np.asarray(x)
    if len(x) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_rms": np.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_p10": float(np.percentile(x, 10)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(x)))),
    }


# =====================================================================
# GENERIC VITERBI
# =====================================================================
def _viterbi_decode(emission_costs, transition_costs, start_state=0, end_state=None):
    """
    Generic minimum-cost Viterbi decoding.
    """
    T, S = emission_costs.shape
    dp = np.full((T, S), np.inf, dtype=float)
    back = np.full((T, S), -1, dtype=int)

    dp[0, start_state] = emission_costs[0, start_state]

    for t in range(1, T):
        prev_cost = dp[t - 1][:, None] + transition_costs
        best_prev = np.argmin(prev_cost, axis=0)
        dp[t] = prev_cost[best_prev, np.arange(S)] + emission_costs[t]
        back[t] = best_prev

    if end_state is not None and np.isfinite(dp[-1, end_state]):
        last_state = end_state
    else:
        last_state = int(np.argmin(dp[-1]))

    total_cost = float(dp[-1, last_state])

    seq = np.zeros(T, dtype=int)
    seq[-1] = last_state
    for t in range(T - 1, 0, -1):
        seq[t - 1] = back[t, seq[t]]

    return seq, total_cost


def _enforce_min_durations(seq, dt, min_duration_map):
    """
    Merge short runs into neighboring runs.
    """
    out = seq.copy()
    changed = True

    while changed:
        changed = False
        runs = _state_runs_from_sequence(out)

        for idx, (s, e, st) in enumerate(runs):
            dur = (e - s) * dt
            min_dur = min_duration_map.get(st, 0.0)

            if dur >= min_dur:
                continue

            left_st = runs[idx - 1][2] if idx > 0 else None
            right_st = runs[idx + 1][2] if idx < len(runs) - 1 else None

            if left_st is None and right_st is None:
                continue
            elif left_st is None:
                out[s:e] = right_st
            elif right_st is None:
                out[s:e] = left_st
            else:
                left_len = runs[idx - 1][1] - runs[idx - 1][0]
                right_len = runs[idx + 1][1] - runs[idx + 1][0]
                out[s:e] = left_st if left_len >= right_len else right_st

            changed = True
            break

    return out


# =====================================================================
# PRIMARY VITERBI
# =====================================================================
def build_primary_transition_cost_matrix():
    """
    Mission grammar:
        GROUND -> TAKEOFF -> MISSION -> LANDING -> GROUND
    """
    BIG = 1e6
    S = len(PRIMARY_STATE_NAMES)

    A = np.full((S, S), BIG, dtype=float)

    # self
    for s in range(S):
        A[s, s] = 0.03

    # main flow
    A[PRIMARY_GROUND, PRIMARY_TAKEOFF] = 0.10
    A[PRIMARY_TAKEOFF, PRIMARY_MISSION] = 0.10
    A[PRIMARY_MISSION, PRIMARY_LANDING] = 0.12
    A[PRIMARY_LANDING, PRIMARY_GROUND] = 0.08

    # a little flexibility
    A[PRIMARY_GROUND, PRIMARY_MISSION] = 0.45
    A[PRIMARY_MISSION, PRIMARY_GROUND] = 0.60

    return A


def compute_primary_emission_costs(
    t,
    features,
    min_moving_speed=0.8,
    min_turn_speed=1.0,
    ground_altitude_quantile=10,
):
    """
    Primary states:
      - ground
      - takeoff
      - mission
      - landing
    """
    T = len(t)

    altitude = features[:, 0]
    v_alt = features[:, 2]
    speed_xy = features[:, 3]
    curvature = features[:, 8]
    yaw_rate = features[:, 9]

    dt = float(np.median(np.diff(t))) if len(t) >= 2 else 0.02

    speed_xy_smooth = _safe_savgol(speed_xy, window_length=21, polyorder=2)
    a_long = np.gradient(speed_xy_smooth, dt)

    abs_yaw = np.abs(yaw_rate)
    abs_curv = np.abs(curvature)

    ground_alt = np.percentile(altitude, ground_altitude_quantile)
    alt_above_ground = altitude - ground_alt

    tau = np.linspace(0.0, 1.0, T)
    early = 1.0 - tau
    late = tau

    moving = _soft_pos_indicator(speed_xy, center=min_moving_speed, width=0.25)
    valid_heading = _soft_pos_indicator(speed_xy, center=min_turn_speed, width=0.25)
    near_ground = 1.0 - _soft_pos_indicator(alt_above_ground, center=0.8, width=0.4)
    above_ground = _soft_pos_indicator(alt_above_ground, center=1.0, width=0.5)

    climbing = _soft_pos_indicator(v_alt, center=0.18, width=0.08)
    descending = _soft_neg_indicator(v_alt, center=0.18, width=0.08)

    turning = np.maximum(
        _soft_pos_indicator(abs_yaw, center=0.12, width=0.05),
        _soft_pos_indicator(abs_curv, center=0.02, width=0.01),
    )

    mission_like = np.maximum(moving, turning) * above_ground

    emission_costs = np.zeros((T, 4), dtype=float)

    # GROUND
    score_ground = (
        0.38 * near_ground +
        0.28 * (1.0 - moving) +
        0.17 * (1.0 - climbing) +
        0.17 * (1.0 - descending)
    )
    emission_costs[:, PRIMARY_GROUND] = 1.0 - score_ground

    # TAKEOFF
    score_takeoff = (
        0.28 * early +
        0.25 * climbing +
        0.20 * above_ground +
        0.15 * moving +
        0.12 * (1.0 - descending)
    )
    emission_costs[:, PRIMARY_TAKEOFF] = 1.0 - score_takeoff

    # MISSION
    score_mission = (
        0.42 * mission_like +
        0.25 * above_ground +
        0.18 * (1.0 - near_ground) +
        0.15 * (1.0 - np.maximum(climbing * early, descending * late))
    )
    emission_costs[:, PRIMARY_MISSION] = 1.0 - score_mission

    # LANDING
    score_landing = (
        0.30 * late +
        0.24 * descending +
        0.20 * moving +
        0.14 * near_ground +
        0.12 * (1.0 - climbing)
    )
    emission_costs[:, PRIMARY_LANDING] = 1.0 - score_landing

    # penalties
    emission_costs[:, PRIMARY_GROUND] += 0.8 * moving * above_ground
    emission_costs[:, PRIMARY_TAKEOFF] += 0.6 * late
    emission_costs[:, PRIMARY_LANDING] += 0.6 * early
    emission_costs[:, PRIMARY_MISSION] += 0.5 * near_ground * (1.0 - moving)

    aux = {
        "a_long": a_long,
        "moving_soft": moving,
        "valid_heading_soft": valid_heading,
        "turning_soft": turning,
        "climbing_soft": climbing,
        "descending_soft": descending,
        "near_ground_soft": near_ground,
        "above_ground_soft": above_ground,
        "alt_above_ground": alt_above_ground,
    }

    return emission_costs, aux


def decode_primary_sequence(t, features, dt=0.02, apply_duration_smoothing=True):
    emission_costs, aux = compute_primary_emission_costs(t, features)
    transition_costs = build_primary_transition_cost_matrix()

    seq, total_cost = _viterbi_decode(
        emission_costs=emission_costs,
        transition_costs=transition_costs,
        start_state=PRIMARY_GROUND,
        end_state=PRIMARY_GROUND,
    )

    if apply_duration_smoothing:
        seq = _enforce_min_durations(
            seq,
            dt=dt,
            min_duration_map={
                PRIMARY_GROUND: 0.5,
                PRIMARY_TAKEOFF: 0.8,
                PRIMARY_MISSION: 1.0,
                PRIMARY_LANDING: 0.8,
            },
        )

    debug = {
        "primary_emission_costs": emission_costs,
        "primary_transition_costs": transition_costs,
        "primary_total_cost": total_cost,
        **aux,
    }
    return seq, debug


# =====================================================================
# SECONDARY AXIS VITERBI
# =====================================================================
def build_lateral_transition_cost_matrix():
    BIG = 1e6
    S = len(LATERAL_STATE_NAMES)
    A = np.full((S, S), BIG, dtype=float)

    for s in range(S):
        A[s, s] = 0.03

    A[LAT_STRAIGHTLIKE, LAT_TURN] = 0.12
    A[LAT_TURN, LAT_STRAIGHTLIKE] = 0.12
    return A


def build_longitudinal_transition_cost_matrix():
    BIG = 1e6
    S = len(LONGITUDINAL_STATE_NAMES)
    A = np.full((S, S), BIG, dtype=float)

    for s in range(S):
        A[s, s] = 0.03

    A[LON_CONST, LON_ACC] = 0.10
    A[LON_CONST, LON_DEC] = 0.10
    A[LON_ACC, LON_CONST] = 0.08
    A[LON_DEC, LON_CONST] = 0.08
    A[LON_ACC, LON_DEC] = 0.20
    A[LON_DEC, LON_ACC] = 0.20
    return A


def build_vertical_transition_cost_matrix():
    BIG = 1e6
    S = len(VERTICAL_STATE_NAMES)
    A = np.full((S, S), BIG, dtype=float)

    for s in range(S):
        A[s, s] = 0.03

    A[VER_LEVEL, VER_CLIMB] = 0.10
    A[VER_LEVEL, VER_DESCEND] = 0.10
    A[VER_CLIMB, VER_LEVEL] = 0.08
    A[VER_DESCEND, VER_LEVEL] = 0.08
    A[VER_CLIMB, VER_DESCEND] = 0.22
    A[VER_DESCEND, VER_CLIMB] = 0.22
    return A


def compute_lateral_emission_costs(t, features, primary_seq=None):
    """
    Lateral axis:
      - straightlike
      - turn
    """
    speed_xy = features[:, 3]
    curvature = features[:, 8]
    yaw_rate = features[:, 9]

    abs_yaw = np.abs(yaw_rate)
    abs_curv = np.abs(curvature)

    valid_heading = _soft_pos_indicator(speed_xy, center=1.0, width=0.25)
    turning = np.maximum(
        _soft_pos_indicator(abs_yaw, center=0.12, width=0.05),
        _soft_pos_indicator(abs_curv, center=0.02, width=0.01),
    )
    nonturning = 1.0 - turning

    costs = np.zeros((len(t), 2), dtype=float)

    score_straightlike = 0.55 * nonturning + 0.45 * valid_heading
    score_turn = 0.60 * turning + 0.40 * valid_heading

    costs[:, LAT_STRAIGHTLIKE] = 1.0 - score_straightlike
    costs[:, LAT_TURN] = 1.0 - score_turn

    # penalty: turn at very low speed is less plausible
    costs[:, LAT_TURN] += 0.6 * (1.0 - valid_heading)

    if primary_seq is not None:
        is_ground = primary_seq == PRIMARY_GROUND
        costs[is_ground, LAT_TURN] += 1.5

    aux = {
        "lateral_turning_soft": turning,
        "lateral_valid_heading_soft": valid_heading,
    }
    return costs, aux


def compute_longitudinal_emission_costs(t, features, primary_seq=None):
    """
    Longitudinal axis:
      - const
      - acc
      - dec
    """
    speed_xy = features[:, 3]
    dt = float(np.median(np.diff(t))) if len(t) >= 2 else 0.02
    speed_xy_smooth = _safe_savgol(speed_xy, window_length=21, polyorder=2)
    a_long = np.gradient(speed_xy_smooth, dt)

    pos_a_long = _soft_pos_indicator(a_long, center=0.25, width=0.10)
    neg_a_long = _soft_neg_indicator(a_long, center=0.25, width=0.10)
    near_zero_a_long = 1.0 - np.maximum(pos_a_long, neg_a_long)

    moving = _soft_pos_indicator(speed_xy, center=0.7, width=0.25)

    costs = np.zeros((len(t), 3), dtype=float)

    score_const = 0.65 * near_zero_a_long + 0.35 * moving
    score_acc = 0.70 * pos_a_long + 0.30 * moving
    score_dec = 0.70 * neg_a_long + 0.30 * moving

    costs[:, LON_CONST] = 1.0 - score_const
    costs[:, LON_ACC] = 1.0 - score_acc
    costs[:, LON_DEC] = 1.0 - score_dec

    if primary_seq is not None:
        is_ground = primary_seq == PRIMARY_GROUND
        costs[is_ground, LON_ACC] += 1.0
        costs[is_ground, LON_DEC] += 1.0

    aux = {
        "a_long": a_long,
        "lon_pos_a_long_soft": pos_a_long,
        "lon_neg_a_long_soft": neg_a_long,
        "lon_near_zero_a_long_soft": near_zero_a_long,
    }
    return costs, aux


def compute_vertical_emission_costs(t, features, primary_seq=None):
    """
    Vertical axis:
      - level
      - climb
      - descend
    """
    v_alt = features[:, 2]

    climbing = _soft_pos_indicator(v_alt, center=0.18, width=0.08)
    descending = _soft_neg_indicator(v_alt, center=0.18, width=0.08)
    level = 1.0 - np.maximum(climbing, descending)

    costs = np.zeros((len(t), 3), dtype=float)

    costs[:, VER_LEVEL] = 1.0 - level
    costs[:, VER_CLIMB] = 1.0 - climbing
    costs[:, VER_DESCEND] = 1.0 - descending

    if primary_seq is not None:
        is_ground = primary_seq == PRIMARY_GROUND
        is_takeoff = primary_seq == PRIMARY_TAKEOFF
        is_landing = primary_seq == PRIMARY_LANDING

        costs[is_ground, VER_CLIMB] += 1.0
        costs[is_ground, VER_DESCEND] += 1.0

        # bias takeoff toward climb, landing toward descend
        costs[is_takeoff, VER_CLIMB] -= 0.15
        costs[is_landing, VER_DESCEND] -= 0.15

    aux = {
        "vertical_climbing_soft": climbing,
        "vertical_descending_soft": descending,
        "vertical_level_soft": level,
    }
    return costs, aux


def decode_secondary_axes(t, features, primary_seq, dt=0.02, apply_duration_smoothing=True):
    # lateral
    lat_emission, lat_aux = compute_lateral_emission_costs(t, features, primary_seq=primary_seq)
    lat_transition = build_lateral_transition_cost_matrix()
    lateral_seq, lat_cost = _viterbi_decode(
        lat_emission,
        lat_transition,
        start_state=LAT_STRAIGHTLIKE,
        end_state=LAT_STRAIGHTLIKE,
    )
    if apply_duration_smoothing:
        lateral_seq = _enforce_min_durations(
            lateral_seq,
            dt=dt,
            min_duration_map={
                LAT_STRAIGHTLIKE: 0.4,
                LAT_TURN: 0.5,
            },
        )

    # longitudinal
    lon_emission, lon_aux = compute_longitudinal_emission_costs(t, features, primary_seq=primary_seq)
    lon_transition = build_longitudinal_transition_cost_matrix()
    longitudinal_seq, lon_cost = _viterbi_decode(
        lon_emission,
        lon_transition,
        start_state=LON_CONST,
        end_state=LON_CONST,
    )
    if apply_duration_smoothing:
        longitudinal_seq = _enforce_min_durations(
            longitudinal_seq,
            dt=dt,
            min_duration_map={
                LON_CONST: 0.3,
                LON_ACC: 0.25,
                LON_DEC: 0.25,
            },
        )

    # vertical
    ver_emission, ver_aux = compute_vertical_emission_costs(t, features, primary_seq=primary_seq)
    ver_transition = build_vertical_transition_cost_matrix()
    vertical_seq, ver_cost = _viterbi_decode(
        ver_emission,
        ver_transition,
        start_state=VER_LEVEL,
        end_state=VER_LEVEL,
    )
    if apply_duration_smoothing:
        vertical_seq = _enforce_min_durations(
            vertical_seq,
            dt=dt,
            min_duration_map={
                VER_LEVEL: 0.3,
                VER_CLIMB: 0.35,
                VER_DESCEND: 0.35,
            },
        )

    debug = {
        "lateral_emission_costs": lat_emission,
        "lateral_transition_costs": lat_transition,
        "lateral_total_cost": lat_cost,
        "longitudinal_emission_costs": lon_emission,
        "longitudinal_transition_costs": lon_transition,
        "longitudinal_total_cost": lon_cost,
        "vertical_emission_costs": ver_emission,
        "vertical_transition_costs": ver_transition,
        "vertical_total_cost": ver_cost,
        **lat_aux,
        **lon_aux,
        **ver_aux,
    }

    return lateral_seq, longitudinal_seq, vertical_seq, debug


# =====================================================================
# DERIVED HOVERLIKE FLAG
# =====================================================================
def compute_hoverlike_flag(
    t,
    features,
    lateral_seq,
    longitudinal_seq,
    vertical_seq,
    a_long=None,
    hover_speed_thresh=0.5,
    hover_vz_thresh=0.15,
    hover_yaw_thresh=0.10,
    hover_a_long_thresh=0.20,
):
    """
    Hoverlike is a derived low-motion configuration, not an axis state.
    """
    speed_xy = features[:, 3]
    v_alt = features[:, 2]
    yaw_rate = features[:, 9]

    if a_long is None:
        dt = float(np.median(np.diff(t))) if len(t) >= 2 else 0.02
        speed_xy_smooth = _safe_savgol(speed_xy, window_length=21, polyorder=2)
        a_long = np.gradient(speed_xy_smooth, dt)

    is_hoverlike = (
        (speed_xy <= hover_speed_thresh) &
        (np.abs(v_alt) <= hover_vz_thresh) &
        (np.abs(yaw_rate) <= hover_yaw_thresh) &
        (np.abs(a_long) <= hover_a_long_thresh) &
        (lateral_seq == LAT_STRAIGHTLIKE) &
        (longitudinal_seq == LON_CONST) &
        (vertical_seq == VER_LEVEL)
    )

    return is_hoverlike


# =====================================================================
# MASTER SEGMENTATION WRAPPER
# =====================================================================
def segment_primary_and_secondary_viterbi(
    t,
    features,
    dt=0.02,
    apply_duration_smoothing=True,
):
    """
    Full segmentation result containing:
      - primary sequence
      - secondary axis sequences
      - hoverlike flag
      - primary segments
      - debug info
    """
    if features is None or len(features) < 2 or len(t) != len(features):
        return {
            "primary_seq": None,
            "lateral_seq": None,
            "longitudinal_seq": None,
            "vertical_seq": None,
            "is_hoverlike": None,
            "primary_segments": [],
            "debug": {},
        }

    primary_seq, primary_debug = decode_primary_sequence(
        t=t,
        features=features,
        dt=dt,
        apply_duration_smoothing=apply_duration_smoothing,
    )

    lateral_seq, longitudinal_seq, vertical_seq, secondary_debug = decode_secondary_axes(
        t=t,
        features=features,
        primary_seq=primary_seq,
        dt=dt,
        apply_duration_smoothing=apply_duration_smoothing,
    )

    a_long = secondary_debug["a_long"]
    is_hoverlike = compute_hoverlike_flag(
        t=t,
        features=features,
        lateral_seq=lateral_seq,
        longitudinal_seq=longitudinal_seq,
        vertical_seq=vertical_seq,
        a_long=a_long,
    )

    primary_segments = primary_sequence_to_segments(
        t=t,
        features=features,
        primary_seq=primary_seq,
    )

    return {
        "primary_seq": primary_seq,
        "lateral_seq": lateral_seq,
        "longitudinal_seq": longitudinal_seq,
        "vertical_seq": vertical_seq,
        "is_hoverlike": is_hoverlike,
        "primary_segments": primary_segments,
        "debug": {
            **primary_debug,
            **secondary_debug,
        },
    }


# =====================================================================
# PRIMARY SEGMENTS
# =====================================================================
def primary_sequence_to_segments(t, features, primary_seq):
    """
    Convert primary phase sequence to primary segment dictionaries.
    """
    primary_segments = []
    runs = _state_runs_from_sequence(primary_seq)

    for s, e, st in runs:
        label = PRIMARY_STATE_NAMES[st]
        seg = _make_segment(label, t, features, s, e)
        if seg is not None:
            seg["primary_state"] = int(st)
            primary_segments.append(seg)

    return primary_segments


# =====================================================================
# SEGMENT-LEVEL DATAFRAME SCHEMA
# =====================================================================
def build_primary_segment_feature_row(
    segment,
    segmentation_result,
    run_id,
    autopilot,
    source_file=None,
):
    """
    Build one DataFrame row from one primary segment.
    """
    t_seg = np.asarray(segment["time"])
    feat_seg = np.asarray(segment["features"])
    s_idx, e_idx = segment["index_span"]

    if len(t_seg) < 2 or len(feat_seg) < 2:
        return None

    primary_seq = segmentation_result["primary_seq"]
    lateral_seq = segmentation_result["lateral_seq"]
    longitudinal_seq = segmentation_result["longitudinal_seq"]
    vertical_seq = segmentation_result["vertical_seq"]
    is_hoverlike = segmentation_result["is_hoverlike"]

    altitude = feat_seg[:, 0]
    heading = feat_seg[:, 1]
    v_alt = feat_seg[:, 2]
    speed_xy = feat_seg[:, 3]
    a_alt = feat_seg[:, 4]
    acc_norm_xy = feat_seg[:, 5]
    j_alt = feat_seg[:, 6]
    jerk_norm_xy = feat_seg[:, 7]
    curvature = feat_seg[:, 8]
    yaw_rate = feat_seg[:, 9]

    dt_local = float(np.median(np.diff(t_seg))) if len(t_seg) >= 2 else np.nan
    duration = float(t_seg[-1] - t_seg[0])
    path_length_xy = float(np.trapz(speed_xy, t_seg))

    speed_xy_smooth = _safe_savgol(speed_xy, window_length=11, polyorder=2)
    a_long = np.gradient(speed_xy_smooth, dt_local) if len(speed_xy_smooth) >= 2 else np.zeros_like(speed_xy_smooth)
    a_lat_approx = speed_xy * np.abs(yaw_rate)

    d_heading = np.diff(heading)
    heading_net_change = float(heading[-1] - heading[0]) if len(heading) >= 2 else 0.0
    heading_abs_change_sum = float(np.sum(np.abs(d_heading))) if len(d_heading) > 0 else 0.0

    lat_seg = lateral_seq[s_idx:e_idx]
    lon_seg = longitudinal_seq[s_idx:e_idx]
    ver_seg = vertical_seq[s_idx:e_idx]
    hover_seg = is_hoverlike[s_idx:e_idx]

    row = {
        "run_id": run_id,
        "autopilot": autopilot,
        "source_file": source_file,
        "primary_label": segment["label"],
        "primary_state": segment["primary_state"],
        "t_start": float(t_seg[0]),
        "t_end": float(t_seg[-1]),
        "duration": duration,
        "n_samples": int(len(t_seg)),
        "dt_mean": dt_local,
        "path_length_xy": path_length_xy,
        "heading_net_change": heading_net_change,
        "heading_abs_change_sum": heading_abs_change_sum,
    }

    row.update(_stats_1d(altitude, "altitude"))
    row.update(_stats_1d(v_alt, "v_alt"))
    row.update(_stats_1d(speed_xy, "speed_xy"))
    row.update(_stats_1d(a_alt, "a_alt"))
    row.update(_stats_1d(acc_norm_xy, "acc_xy_norm"))
    row.update(_stats_1d(j_alt, "j_alt"))
    row.update(_stats_1d(jerk_norm_xy, "jerk_xy_norm"))
    row.update(_stats_1d(curvature, "curvature"))
    row.update(_stats_1d(yaw_rate, "yaw_rate"))
    row.update(_stats_1d(a_long, "a_long"))
    row.update(_stats_1d(a_lat_approx, "a_lat_approx"))

    row.update({
        "integral_abs_yaw_rate": float(np.trapz(np.abs(yaw_rate), t_seg)),
        "integral_abs_curvature": float(np.trapz(np.abs(curvature), t_seg)),
        "integral_abs_a_long": float(np.trapz(np.abs(a_long), t_seg)),
        "integral_abs_v_alt": float(np.trapz(np.abs(v_alt), t_seg)),
        "speed_delta": float(speed_xy[-1] - speed_xy[0]),
        "altitude_delta": float(altitude[-1] - altitude[0]),
    })

    # secondary fractions
    row.update({
        "lateral_turn_fraction": float(np.mean(lat_seg == LAT_TURN)),
        "lateral_straightlike_fraction": float(np.mean(lat_seg == LAT_STRAIGHTLIKE)),
        "longitudinal_const_fraction": float(np.mean(lon_seg == LON_CONST)),
        "longitudinal_acc_fraction": float(np.mean(lon_seg == LON_ACC)),
        "longitudinal_dec_fraction": float(np.mean(lon_seg == LON_DEC)),
        "vertical_level_fraction": float(np.mean(ver_seg == VER_LEVEL)),
        "vertical_climb_fraction": float(np.mean(ver_seg == VER_CLIMB)),
        "vertical_descend_fraction": float(np.mean(ver_seg == VER_DESCEND)),
        "hoverlike_fraction": float(np.mean(hover_seg)),
        "hoverlike_count": int(np.sum(hover_seg)),
    })

    # dominant modes
    row["lateral_mode_dominant"] = LATERAL_STATE_NAMES[int(np.bincount(lat_seg).argmax())]
    row["longitudinal_mode_dominant"] = LONGITUDINAL_STATE_NAMES[int(np.bincount(lon_seg).argmax())]
    row["vertical_mode_dominant"] = VERTICAL_STATE_NAMES[int(np.bincount(ver_seg).argmax())]

    row["has_turn"] = int(row["lateral_turn_fraction"] > 0.2)
    row["has_hoverlike"] = int(row["hoverlike_fraction"] > 0.2)
    row["has_acc"] = int(row["longitudinal_acc_fraction"] > 0.2)
    row["has_dec"] = int(row["longitudinal_dec_fraction"] > 0.2)
    row["has_climb"] = int(row["vertical_climb_fraction"] > 0.2)
    row["has_descend"] = int(row["vertical_descend_fraction"] > 0.2)

    return row


def build_primary_segment_dataframe(
    segmentation_result,
    run_id,
    autopilot,
    source_file=None,
):
    """
    One row per primary segment.
    """
    rows = []
    for seg in segmentation_result["primary_segments"]:
        row = build_primary_segment_feature_row(
            segment=seg,
            segmentation_result=segmentation_result,
            run_id=run_id,
            autopilot=autopilot,
            source_file=source_file,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["run_id", "t_start"]).reset_index(drop=True)
    return df


# =====================================================================
# OPTIONAL LEGACY-COMPATIBLE SPAN EXPORT
# =====================================================================
def export_legacy_segment_views(segmentation_result):
    """
    Optional helper for compatibility with older plotting code.

    Returns
    -------
    segments : dict
    spans : dict
        Keys include:
          ground, takeoff, mission, landing
    """
    segments = {
        "ground": [],
        "takeoff": None,
        "mission": [],
        "landing": None,
    }
    spans = {
        "ground": [],
        "takeoff": None,
        "mission": [],
        "landing": None,
    }

    for seg in segmentation_result["primary_segments"]:
        label = seg["label"]
        if label == "ground":
            segments["ground"].append(seg)
            spans["ground"].append(seg["span"])
        elif label == "takeoff":
            if segments["takeoff"] is None:
                segments["takeoff"] = seg
                spans["takeoff"] = seg["span"]
            else:
                segments["ground"].append(seg)
                spans["ground"].append(seg["span"])
        elif label == "mission":
            segments["mission"].append(seg)
            spans["mission"].append(seg["span"])
        elif label == "landing":
            if segments["landing"] is None:
                segments["landing"] = seg
                spans["landing"] = seg["span"]
            else:
                segments["ground"].append(seg)
                spans["ground"].append(seg["span"])

    return segments, spans


# =====================================================================
# EXAMPLE PROCESS WRAPPERS
# These replace the older extract_flight_segments_viterbi() usage.
# =====================================================================
def process_px4_flight_data_v3(ulog_path):
    try:
        ulog = ULog(ulog_path)
        loc_data = ulog.get_dataset("vehicle_local_position").data

        t_loc = loc_data["timestamp"] / 1e6
        x, y, z = loc_data["x"], loc_data["y"], loc_data["z"]
        vx, vy, vz = loc_data["vx"], loc_data["vy"], loc_data["vz"]

        extracted = extract_kinematic_features(
            t_loc, x, y, z, vx, vy, vz, target_hz=TARGET_HZ
        )
        if extracted is None:
            return None

        t_full, feat_full = extracted
        segmentation_result = segment_primary_and_secondary_viterbi(
            t=t_full,
            features=feat_full,
            dt=DT,
            apply_duration_smoothing=True,
        )

        return {
            "x": x,
            "y": y,
            "t": t_full,
            "features": feat_full,
            "segmentation": segmentation_result,
        }

    except Exception as e:
        print(f"[PX4 Extract Error] {ulog_path}: {e}")
        return None


def process_ardu_flight_data_v3(bin_path):
    try:
        mlog = mavutil.mavlink_connection(bin_path)
        t_loc, x, y, z, vx, vy, vz = [], [], [], [], [], [], []

        while True:
            msg = mlog.recv_match(type=["XKF1", "NKF1"], blocking=False)
            if not msg:
                break
            t_loc.append(msg.TimeUS / 1e6)
            x.append(msg.PN)
            y.append(msg.PE)
            z.append(msg.PD)
            vx.append(msg.VN)
            vy.append(msg.VE)
            vz.append(msg.VD)

        if len(x) < 50:
            return None

        extracted = extract_kinematic_features(
            np.array(t_loc),
            np.array(x),
            np.array(y),
            np.array(z),
            np.array(vx),
            np.array(vy),
            np.array(vz),
            target_hz=TARGET_HZ,
        )
        if extracted is None:
            return None

        t_full, feat_full = extracted
        segmentation_result = segment_primary_and_secondary_viterbi(
            t=t_full,
            features=feat_full,
            dt=DT,
            apply_duration_smoothing=True,
        )

        return {
            "x": x,
            "y": y,
            "t": t_full,
            "features": feat_full,
            "segmentation": segmentation_result,
        }

    except Exception as e:
        print(f"[ArduPilot Extract Error] {bin_path}: {e}")
        return None


# =====================================================================
# DATASET COLLECTION
# =====================================================================
def collect_primary_segment_dataset_v2(base_data_dir, max_runs=100):
    """
    Build one DataFrame row per primary segment across PX4 and ArduPilot runs.
    """
    all_dfs = []

    for i in range(max_runs):
        run_folder = f"run_{i:03d}"
        run_dir = base_data_dir / run_folder
        if not run_dir.exists():
            continue

        px4_dir = run_dir / "px4_logs" / "raw"
        ardu_dir = run_dir / "ardu_logs" / "raw" / "logs"

        # PX4
        if px4_dir.exists():
            for file in os.listdir(px4_dir):
                if file.lower().endswith(".ulg"):
                    result = process_px4_flight_data_v3(str(px4_dir / file))
                    if result is not None:
                        df_px4 = build_primary_segment_dataframe(
                            segmentation_result=result["segmentation"],
                            run_id=run_folder,
                            autopilot="px4",
                            source_file=file,
                        )
                        if not df_px4.empty:
                            all_dfs.append(df_px4)
                    break

        # ArduPilot
        if ardu_dir.exists():
            for file in os.listdir(ardu_dir):
                if file.lower().endswith(".bin"):
                    result = process_ardu_flight_data_v3(str(ardu_dir / file))
                    if result is not None:
                        df_ardu = build_primary_segment_dataframe(
                            segmentation_result=result["segmentation"],
                            run_id=run_folder,
                            autopilot="ardupilot",
                            source_file=file,
                        )
                        if not df_ardu.empty:
                            all_dfs.append(df_ardu)
                    break

    if not all_dfs:
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all["autopilot_label"] = df_all["autopilot"].map({"px4": 0, "ardupilot": 1})
    return df_all