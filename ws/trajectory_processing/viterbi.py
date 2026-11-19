# =====================================================================
# PHYSICS-AWARE VITERBI SEGMENTATION
# =====================================================================

STATE_GROUND = 0
STATE_TAKEOFF = 1
STATE_STRAIGHT_CONST = 2
STATE_STRAIGHT_ACCEL = 3
STATE_STRAIGHT_DECEL = 4
STATE_TURN = 5
STATE_LANDING = 6

STATE_NAMES = {
    STATE_GROUND: "ground",
    STATE_TAKEOFF: "takeoff",
    STATE_STRAIGHT_CONST: "straight_const",
    STATE_STRAIGHT_ACCEL: "straight_accel",
    STATE_STRAIGHT_DECEL: "straight_decel",
    STATE_TURN: "turn",
    STATE_LANDING: "landing",
}

PRIMARY_STATE_TO_GROUP = {
    STATE_GROUND: "ground",
    STATE_TAKEOFF: "takeoff",
    STATE_STRAIGHT_CONST: "straight",
    STATE_STRAIGHT_ACCEL: "straight",
    STATE_STRAIGHT_DECEL: "straight",
    STATE_TURN: "turn",
    STATE_LANDING: "landing",
}

# ======================================================================================

def _robust_zscore(x):
    """
    Robust z-score using median and MAD.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / (1.4826 * mad)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _soft_abs_score(x, center, width):
    """
    Low cost when |x| is near or below center, increases smoothly afterward.
    """
    return _sigmoid((np.abs(x) - center) / max(width, 1e-6))


def _soft_pos_score(x, center, width):
    """
    Higher value when x is sufficiently positive.
    """
    return _sigmoid((x - center) / max(width, 1e-6))


def _soft_neg_score(x, center, width):
    """
    Higher value when x is sufficiently negative.
    """
    return _sigmoid((-x - center) / max(width, 1e-6))


def _normalize01(x):
    """
    Normalize array approximately into [0,1] using percentiles.
    """
    x = np.asarray(x, dtype=float)
    lo = np.percentile(x, 5)
    hi = np.percentile(x, 95)
    if hi <= lo + 1e-9:
        return np.zeros_like(x)
    out = (x - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)

# ======================================================================================

def build_transition_cost_matrix():
    """
    Build transition cost matrix for mission-aware segmentation.

    Lower cost = more likely transition.
    Very large cost = effectively forbidden.
    """
    n_states = len(STATE_NAMES)
    BIG = 1e6

    A = np.full((n_states, n_states), BIG, dtype=float)

    # self-transitions: always cheap
    for s in range(n_states):
        A[s, s] = 0.05

    # ground <-> takeoff / landing
    A[STATE_GROUND, STATE_TAKEOFF] = 0.2
    A[STATE_LANDING, STATE_GROUND] = 0.2

    # takeoff can go to straight or turn
    A[STATE_TAKEOFF, STATE_STRAIGHT_CONST] = 0.2
    A[STATE_TAKEOFF, STATE_STRAIGHT_ACCEL] = 0.15
    A[STATE_TAKEOFF, STATE_STRAIGHT_DECEL] = 0.5
    A[STATE_TAKEOFF, STATE_TURN] = 0.4

    # straight modes can interchange
    A[STATE_STRAIGHT_CONST, STATE_STRAIGHT_ACCEL] = 0.12
    A[STATE_STRAIGHT_CONST, STATE_STRAIGHT_DECEL] = 0.12
    A[STATE_STRAIGHT_ACCEL, STATE_STRAIGHT_CONST] = 0.10
    A[STATE_STRAIGHT_DECEL, STATE_STRAIGHT_CONST] = 0.10
    A[STATE_STRAIGHT_ACCEL, STATE_STRAIGHT_DECEL] = 0.22
    A[STATE_STRAIGHT_DECEL, STATE_STRAIGHT_ACCEL] = 0.22

    # straight <-> turn
    for s in [STATE_STRAIGHT_CONST, STATE_STRAIGHT_ACCEL, STATE_STRAIGHT_DECEL]:
        A[s, STATE_TURN] = 0.18
        A[STATE_TURN, s] = 0.18

    # landing can come from straight or turn
    for s in [STATE_STRAIGHT_CONST, STATE_STRAIGHT_ACCEL, STATE_STRAIGHT_DECEL]:
        A[s, STATE_LANDING] = 0.28
    A[STATE_TURN, STATE_LANDING] = 0.35

    # allow takeoff to remain or drift slightly
    A[STATE_GROUND, STATE_GROUND] = 0.02
    A[STATE_TAKEOFF, STATE_TAKEOFF] = 0.04
    A[STATE_LANDING, STATE_LANDING] = 0.04

    return A


# ======================================================================================


def compute_state_emission_costs(
    t,
    features,
    ground_altitude_quantile=10,
    min_valid_speed_for_heading=1.0,
):
    """
    Compute emission cost for each state at each time.

    Returns
    -------
    costs : ndarray, shape (T, n_states)
        Lower is better.
    aux : dict
        Auxiliary arrays useful for debugging / plotting.
    """
    T = len(t)
    n_states = len(STATE_NAMES)

    altitude = features[:, 0]
    heading = features[:, 1]
    v_alt = features[:, 2]
    speed_xy = features[:, 3]
    a_alt = features[:, 4]
    acc_norm_xy = features[:, 5]
    j_alt = features[:, 6]
    jerk_norm_xy = features[:, 7]
    curvature = features[:, 8]
    yaw_rate = features[:, 9]

    dt = np.median(np.diff(t)) if len(t) >= 2 else DT

    speed_xy_smooth = _safe_savgol(speed_xy, window_length=21, polyorder=2)
    a_long = np.gradient(speed_xy_smooth, dt)

    abs_yaw = np.abs(yaw_rate)
    abs_curv = np.abs(curvature)
    abs_v_alt = np.abs(v_alt)
    abs_a_long = np.abs(a_long)

    # time priors
    tau = np.linspace(0.0, 1.0, T)

    # reference levels
    ground_alt_ref = np.percentile(altitude, ground_altitude_quantile)
    alt_above_ground = altitude - ground_alt_ref

    # normalized evidences
    speed_n = _normalize01(speed_xy)
    abs_yaw_n = _normalize01(abs_yaw)
    curv_n = _normalize01(abs_curv)
    abs_v_alt_n = _normalize01(abs_v_alt)
    abs_a_long_n = _normalize01(abs_a_long)
    jerk_n = _normalize01(jerk_norm_xy)

    # soft indicators
    moving = _soft_pos_score(speed_xy, center=0.8, width=0.3)
    valid_heading = _soft_pos_score(speed_xy, center=min_valid_speed_for_heading, width=0.3)
    turning = np.maximum(
        _soft_pos_score(abs_yaw, center=0.12, width=0.05),
        _soft_pos_score(abs_curv, center=0.02, width=0.01),
    )
    nonturning = 1.0 - turning

    climbing = _soft_pos_score(v_alt, center=0.15, width=0.08)
    descending = _soft_neg_score(v_alt, center=0.15, width=0.08)

    pos_a_long = _soft_pos_score(a_long, center=0.25, width=0.12)
    neg_a_long = _soft_neg_score(a_long, center=0.25, width=0.12)
    near_zero_a_long = 1.0 - np.maximum(pos_a_long, neg_a_long)

    near_ground = 1.0 - _soft_pos_score(alt_above_ground, center=0.8, width=0.4)
    above_ground = _soft_pos_score(alt_above_ground, center=1.0, width=0.5)

    early = 1.0 - tau
    late = tau

    costs = np.zeros((T, n_states), dtype=float)

    # --------------------------------------------------------------
    # GROUND
    # --------------------------------------------------------------
    score_ground = (
        0.35 * near_ground +
        0.30 * (1.0 - moving) +
        0.20 * (1.0 - climbing) +
        0.15 * (1.0 - descending)
    )
    costs[:, STATE_GROUND] = 1.0 - score_ground

    # --------------------------------------------------------------
    # TAKEOFF
    # --------------------------------------------------------------
    score_takeoff = (
        0.25 * above_ground +
        0.25 * climbing +
        0.20 * moving +
        0.15 * early +
        0.15 * nonturning
    )
    costs[:, STATE_TAKEOFF] = 1.0 - score_takeoff

    # --------------------------------------------------------------
    # STRAIGHT CONST
    # --------------------------------------------------------------
    score_st_const = (
        0.30 * moving +
        0.25 * valid_heading +
        0.20 * nonturning +
        0.15 * near_zero_a_long +
        0.10 * above_ground
    )
    costs[:, STATE_STRAIGHT_CONST] = 1.0 - score_st_const

    # --------------------------------------------------------------
    # STRAIGHT ACCEL
    # --------------------------------------------------------------
    score_st_accel = (
        0.28 * moving +
        0.22 * valid_heading +
        0.18 * nonturning +
        0.22 * pos_a_long +
        0.10 * above_ground
    )
    costs[:, STATE_STRAIGHT_ACCEL] = 1.0 - score_st_accel

    # --------------------------------------------------------------
    # STRAIGHT DECEL
    # --------------------------------------------------------------
    score_st_decel = (
        0.28 * moving +
        0.22 * valid_heading +
        0.18 * nonturning +
        0.22 * neg_a_long +
        0.10 * above_ground
    )
    costs[:, STATE_STRAIGHT_DECEL] = 1.0 - score_st_decel

    # --------------------------------------------------------------
    # TURN
    # --------------------------------------------------------------
    score_turn = (
        0.30 * moving +
        0.25 * valid_heading +
        0.30 * turning +
        0.15 * above_ground
    )
    costs[:, STATE_TURN] = 1.0 - score_turn

    # --------------------------------------------------------------
    # LANDING
    # --------------------------------------------------------------
    score_landing = (
        0.28 * descending +
        0.20 * moving +
        0.18 * late +
        0.16 * near_ground +
        0.18 * nonturning
    )
    costs[:, STATE_LANDING] = 1.0 - score_landing

    # penalties to suppress clearly nonphysical assignments
    # ground in middle of high-speed cruise
    costs[:, STATE_GROUND] += 0.8 * moving * above_ground
    # takeoff late in mission
    costs[:, STATE_TAKEOFF] += 0.5 * late
    # landing very early
    costs[:, STATE_LANDING] += 0.5 * early
    # turn at very low speed
    costs[:, STATE_TURN] += 0.6 * (1.0 - valid_heading)
    # straight when turn evidence is strong
    strong_turn_pen = 0.7 * turning
    costs[:, STATE_STRAIGHT_CONST] += strong_turn_pen
    costs[:, STATE_STRAIGHT_ACCEL] += strong_turn_pen
    costs[:, STATE_STRAIGHT_DECEL] += strong_turn_pen

    aux = {
        "a_long": a_long,
        "moving": moving,
        "turning": turning,
        "climbing": climbing,
        "descending": descending,
        "near_ground": near_ground,
        "above_ground": above_ground,
    }

    return costs, aux






# ======================================================================================

def decode_state_sequence_viterbi(
    emission_costs,
    transition_costs,
    start_state=STATE_GROUND,
    end_state=STATE_GROUND,
):
    """
    Viterbi decoding for minimum-cost state sequence.

    Parameters
    ----------
    emission_costs : ndarray, shape (T, S)
    transition_costs : ndarray, shape (S, S)

    Returns
    -------
    states : ndarray, shape (T,)
        Best state index per time.
    total_cost : float
        Final path cost.
    """
    T, S = emission_costs.shape

    dp = np.full((T, S), np.inf, dtype=float)
    back = np.full((T, S), -1, dtype=int)

    # initialization
    dp[0, start_state] = emission_costs[0, start_state]

    for t in range(1, T):
        prev_cost = dp[t - 1][:, None] + transition_costs
        best_prev = np.argmin(prev_cost, axis=0)
        dp[t] = prev_cost[best_prev, np.arange(S)] + emission_costs[t]
        back[t] = best_prev

    last_state = end_state if np.isfinite(dp[-1, end_state]) else int(np.argmin(dp[-1]))
    total_cost = float(dp[-1, last_state])

    states = np.zeros(T, dtype=int)
    states[-1] = last_state
    for t in range(T - 1, 0, -1):
        states[t - 1] = back[t, states[t]]

    return states, total_cost


# ======================================================================================

def enforce_min_state_durations(state_seq, dt=0.02):
    """
    Merge runs that are shorter than state-specific minimum durations.

    Strategy:
    - If a run is too short, merge it into the neighboring state with lower
      local transition penalty preference.
    """
    min_duration_sec = {
        STATE_GROUND: 0.6,
        STATE_TAKEOFF: 0.8,
        STATE_STRAIGHT_CONST: 0.5,
        STATE_STRAIGHT_ACCEL: 0.35,
        STATE_STRAIGHT_DECEL: 0.35,
        STATE_TURN: 0.5,
        STATE_LANDING: 0.8,
    }

    seq = state_seq.copy()
    changed = True

    while changed:
        changed = False
        runs = _mask_runs_from_state_sequence(seq)

        for idx, (s, e, state) in enumerate(runs):
            dur = (e - s) * dt
            min_dur = min_duration_sec.get(state, 0.0)

            if dur >= min_dur:
                continue

            left_state = runs[idx - 1][2] if idx > 0 else None
            right_state = runs[idx + 1][2] if idx < len(runs) - 1 else None

            if left_state is None and right_state is None:
                continue
            elif left_state is None:
                seq[s:e] = right_state
            elif right_state is None:
                seq[s:e] = left_state
            else:
                # prefer neighbor with longer run
                left_len = runs[idx - 1][1] - runs[idx - 1][0]
                right_len = runs[idx + 1][1] - runs[idx + 1][0]
                seq[s:e] = left_state if left_len >= right_len else right_state

            changed = True
            break

    return seq


def _mask_runs_from_state_sequence(state_seq):
    """
    Convert discrete state sequence to runs:
    [(start, end, state), ...] with [start, end).
    """
    if len(state_seq) == 0:
        return []

    runs = []
    s = 0
    cur = state_seq[0]

    for i in range(1, len(state_seq)):
        if state_seq[i] != cur:
            runs.append((s, i, cur))
            s = i
            cur = state_seq[i]

    runs.append((s, len(state_seq), cur))
    return runs

# ======================================================================================

def state_sequence_to_segments(t, features, state_seq):
    """
    Convert decoded state sequence into segment/spans structure compatible
    with the existing pipeline.
    """
    segments = {
        "ground": [],
        "takeoff": None,
        "landing": None,
        "straight": [],
        "turn": [],
        "straight_const": [],
        "straight_accel": [],
        "straight_decel": [],
    }

    spans = {
        "ground": [],
        "takeoff": None,
        "landing": None,
        "straight": [],
        "turn": [],
        "straight_const": [],
        "straight_accel": [],
        "straight_decel": [],
    }

    runs = _mask_runs_from_state_sequence(state_seq)

    for s, e, state in runs:
        label = STATE_NAMES[state]
        seg = _make_segment(label, t, features, s, e)
        if seg is None:
            continue

        if state == STATE_TAKEOFF:
            if segments["takeoff"] is None:
                segments["takeoff"] = seg
                spans["takeoff"] = seg["span"]
            else:
                segments["ground"].append(seg)
                spans["ground"].append(seg["span"])

        elif state == STATE_LANDING:
            if segments["landing"] is None:
                segments["landing"] = seg
                spans["landing"] = seg["span"]
            else:
                segments["ground"].append(seg)
                spans["ground"].append(seg["span"])

        elif state == STATE_TURN:
            segments["turn"].append(seg)
            spans["turn"].append(seg["span"])

        elif state == STATE_STRAIGHT_CONST:
            segments["straight_const"].append(seg)
            spans["straight_const"].append(seg["span"])
            seg_st = seg.copy()
            seg_st["label"] = "straight"
            segments["straight"].append(seg_st)
            spans["straight"].append(seg["span"])

        elif state == STATE_STRAIGHT_ACCEL:
            segments["straight_accel"].append(seg)
            spans["straight_accel"].append(seg["span"])
            seg_st = seg.copy()
            seg_st["label"] = "straight"
            segments["straight"].append(seg_st)
            spans["straight"].append(seg["span"])

        elif state == STATE_STRAIGHT_DECEL:
            segments["straight_decel"].append(seg)
            spans["straight_decel"].append(seg["span"])
            seg_st = seg.copy()
            seg_st["label"] = "straight"
            segments["straight"].append(seg_st)
            spans["straight"].append(seg["span"])

        elif state == STATE_GROUND:
            segments["ground"].append(seg)
            spans["ground"].append(seg["span"])

    return segments, spans

# ======================================================================================


def extract_flight_segments_viterbi(
    t,
    features,
    dt=0.02,
    apply_duration_smoothing=True,
):
    """
    Physics-aware mission segmentation using feature-based emissions and
    mission-flow-constrained Viterbi decoding.

    Returns
    -------
    segments : dict
    spans : dict
    debug : dict
        Includes emission costs, decoded states, and auxiliary signals.
    """
    if features is None or len(features) < 2 or len(t) != len(features):
        segments, spans = _empty_segment_result()
        return segments, spans, {}

    emission_costs, aux = compute_state_emission_costs(t, features)
    transition_costs = build_transition_cost_matrix()

    state_seq, total_cost = decode_state_sequence_viterbi(
        emission_costs=emission_costs,
        transition_costs=transition_costs,
        start_state=STATE_GROUND,
        end_state=STATE_GROUND,
    )

    if apply_duration_smoothing:
        state_seq = enforce_min_state_durations(state_seq, dt=dt)

    segments, spans = state_sequence_to_segments(t, features, state_seq)

    debug = {
        "emission_costs": emission_costs,
        "transition_costs": transition_costs,
        "state_seq": state_seq,
        "total_cost": total_cost,
        **aux,
    }

    return segments, spans, debug

# ======================================================================================
