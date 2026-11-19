# =====================================================================
# HELPERS
# =====================================================================
def _mask_to_ranges(mask: np.ndarray):
    """
    Convert a boolean mask into half-open ranges [start, end).
    """
    if len(mask) == 0:
        return []

    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)

    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask)]

    return list(zip(starts, ends))


def _remove_short_true_runs(mask: np.ndarray, min_len: int):
    """
    Remove True runs shorter than min_len.
    """
    out = mask.copy()
    for s, e in _mask_to_ranges(mask):
        if (e - s) < min_len:
            out[s:e] = False
    return out


def _fill_short_false_gaps(mask: np.ndarray, max_gap: int):
    """
    Fill False gaps shorter than or equal to max_gap,
    excluding boundary gaps.
    """
    out = mask.copy()
    inv = ~mask
    for s, e in _mask_to_ranges(inv):
        if s == 0 or e == len(mask):
            continue
        if (e - s) <= max_gap:
            out[s:e] = True
    return out


def _make_segment(label, tt, ff, s_idx, e_idx):
    """
    Create a segment dictionary for [s_idx, e_idx).
    """
    if e_idx <= s_idx:
        return None

    safe_e = min(e_idx, len(tt))
    if safe_e <= s_idx:
        return None

    return {
        "label": label,
        "time": tt[s_idx:safe_e],
        "features": ff[s_idx:safe_e],
        "span": (tt[s_idx], tt[safe_e - 1]),
        "index_span": (s_idx, safe_e),
    }


def _empty_segment_result():
    spans = {
        "takeoff": None,
        "landing": None,
        "straight": [],
        "turn": [],
        "unknown": [],
        "straight_const": [],
        "straight_accel": [],
        "straight_decel": [],
    }
    segments = {
        "takeoff": None,
        "landing": None,
        "straight": [],
        "turn": [],
        "unknown": [],
        "straight_const": [],
        "straight_accel": [],
        "straight_decel": [],
    }
    return segments, spans


# =====================================================================
# SEGMENTATION V1
# =====================================================================
def extract_flight_segments(t, 
                            features, 
                            dt=0.02, 
                            min_turn_duration=0.5, 
                            yaw_rate_threshold=0.2, 
                            straight_duration=1.0, 
                            heading_margin=0.2):
    spans = {'takeoff': None, 'landing': None, 'straight': [], 'turn': []}
    segments = {'takeoff': None, 'landing': None, 'straight': [], 'turn': []}

    if features is None or len(features) < 2:
        return segments, spans
        
    N = len(features)
    alt = features[:, 0]
    vz_norm = np.abs(features[:, 2])
    az_norm = np.abs(features[:, 4])
    jerk_norm = np.abs(features[:, 7])  
    
    alt_95 = np.percentile(alt, 95)
    target_alt = alt_95 - 0.1
    
    # [Step 1] Landing/Take-off
    flight_start_idx = 0
    state = 0
    for i in range(N):
        if state == 0 and alt[i] >= target_alt: state = 1
        elif state == 1 and vz_norm[i] <= 0.1: state = 2
        elif state == 2 and az_norm[i] <= 0.1:
            flight_start_idx = i
            break
                
    flight_end_idx = N
    state = 0
    for i in range(N - 1, flight_start_idx, -1):
        if state == 0 and alt[i] >= target_alt: state = 1
        elif state == 1 and vz_norm[i] <= 0.1: state = 2
        elif state == 2 and az_norm[i] <= 0.1:
            flight_end_idx = i
            break

    if flight_start_idx > 0:
        spans['takeoff'] = (t[0], t[flight_start_idx])
        segments['takeoff'] = {
            'label': 'takeoff',
            'time': t[0:flight_start_idx],
            'features': features[0:flight_start_idx],
            'span': (t[0], t[flight_start_idx])
        }
    
    if flight_end_idx < N - 1:
        spans['landing'] = (t[flight_end_idx], t[-1])
        segments['landing'] = {
            'label': 'landing',
            'time': t[flight_end_idx:],
            'features': features[flight_end_idx:],
            'span': (t[flight_end_idx], t[-1])
        }
                
    flight_t = t[flight_start_idx:flight_end_idx]
    flight_features = features[flight_start_idx:flight_end_idx]
    
    if len(flight_features) == 0:
        return segments, spans
        
    # [Step 2] Straight 
    is_straight = np.zeros(len(flight_features), dtype=bool)
    window_size = int(straight_duration / dt) 
    
    if len(flight_features) >= window_size:
        for i in range(len(flight_features) - window_size + 1):
            window_heading = flight_features[i : i + window_size, 1]
            window_yaw_rate_mag = np.abs(flight_features[i : i + window_size, 9])
            
            if np.max(window_yaw_rate_mag) <= yaw_rate_threshold:
                heading_diff = np.max(window_heading) - np.min(window_heading)
                if heading_diff <= heading_margin:
                    is_straight[i : i + window_size] = True

    edges = np.diff(is_straight.astype(int))
    st_starts = np.where(edges == 1)[0] + 1
    st_ends = np.where(edges == -1)[0] + 1
    if is_straight[0]: st_starts = np.insert(st_starts, 0, 0)
    if is_straight[-1]: st_ends = np.append(st_ends, len(is_straight))
    
    for start, end in zip(st_starts, st_ends):
        safe_end = min(end, len(flight_t) - 1)
        spans['straight'].append((flight_t[start], flight_t[safe_end]))
        segments['straight'].append({
            'label': 'straight',
            'time': flight_t[start:safe_end],
            'features': flight_features[start:safe_end],
            'span': (flight_t[start], flight_t[safe_end])
        })

    straight_indices = np.where(is_straight)[0]
    if len(straight_indices) == 0:
        return segments, spans 
        
    first_straight_idx = straight_indices[0]

    # [Step 3] Turn 
    is_turning = ~is_straight 
    is_turning[:first_straight_idx] = False
    
    edges = np.diff(is_turning.astype(int))
    turn_starts = np.where(edges == 1)[0] + 1
    turn_ends = np.where(edges == -1)[0] + 1
    
    if is_turning[0]: turn_starts = np.insert(turn_starts, 0, 0)
    if is_turning[-1]: turn_ends = np.append(turn_ends, len(is_turning))
        
    min_length = int(min_turn_duration / dt) 
    
    for start, end in zip(turn_starts, turn_ends):
        if (end - start) < min_length:
            continue
            
        safe_end = min(end, len(flight_t) - 1)
        spans['turn'].append((flight_t[start], flight_t[safe_end]))
        segments['turn'].append({
            'label': 'turn',
            'time': flight_t[start:safe_end],
            'features': flight_features[start:safe_end],
            'span': (flight_t[start], flight_t[safe_end])
        })
        
    return segments, spans


# =====================================================================
# SEGMENTATION V2
# =====================================================================
def extract_flight_segments_v2(
    t,
    features,
    dt=0.02,
    altitude_margin=0.1,
    vz_small=0.12,
    az_small=0.20,
    min_moving_speed=0.8,
    min_valid_speed_for_heading=1.0,
    yaw_rate_on=0.18,
    yaw_rate_off=0.08,
    curvature_on=0.03,
    curvature_off=0.015,
    min_turn_duration=0.6,
    min_straight_duration=1.0,
    straight_yaw_rate_max=0.08,
    straight_curvature_max=0.015,
    long_acc_threshold=0.4,
    min_accel_duration=0.4,
    max_merge_gap=0.25,
    min_unknown_duration=0.3,
):
    """
    Improved flight segmentation with straight subtypes.

    Primary classes
    ---------------
    - takeoff
    - landing
    - straight
    - turn
    - unknown

    Straight subtypes
    -----------------
    - straight_const
    - straight_accel
    - straight_decel

    Design choice
    -------------
    Turn is treated as a single semantic maneuver even if acceleration
    or deceleration occurs within it. Straight segments are further split
    into longitudinal speed-change subtypes because those are often useful
    for autopilot discrimination.
    """
    segments, spans = _empty_segment_result()

    if features is None or len(features) < 2 or len(t) != len(features):
        return segments, spans

    N = len(features)

    altitude = features[:, 0]
    v_alt = features[:, 2]
    speed_xy = features[:, 3]
    a_alt = features[:, 4]
    curvature = features[:, 8]
    yaw_rate = features[:, 9]

    # --------------------------------------------------------------
    # Step 1. Crop takeoff / landing using altitude + vertical settling
    # --------------------------------------------------------------
    alt_95 = np.percentile(altitude, 95)
    target_alt = alt_95 - altitude_margin

    flight_start_idx = 0
    state = 0
    for i in range(N):
        if state == 0 and altitude[i] >= target_alt:
            state = 1
        elif state == 1 and abs(v_alt[i]) <= vz_small:
            state = 2
        elif state == 2 and abs(a_alt[i]) <= az_small:
            flight_start_idx = i
            break

    flight_end_idx = N
    state = 0
    for i in range(N - 1, flight_start_idx, -1):
        if state == 0 and altitude[i] >= target_alt:
            state = 1
        elif state == 1 and abs(v_alt[i]) <= vz_small:
            state = 2
        elif state == 2 and abs(a_alt[i]) <= az_small:
            flight_end_idx = i
            break

    if flight_start_idx > 0:
        seg = _make_segment("takeoff", t, features, 0, flight_start_idx)
        if seg is not None:
            segments["takeoff"] = seg
            spans["takeoff"] = seg["span"]

    if flight_end_idx < N - 1:
        seg = _make_segment("landing", t, features, flight_end_idx, N)
        if seg is not None:
            segments["landing"] = seg
            spans["landing"] = seg["span"]

    flight_t = t[flight_start_idx:flight_end_idx]
    flight_features = features[flight_start_idx:flight_end_idx]

    if len(flight_features) < 2:
        return segments, spans

    speed_xy_f = flight_features[:, 3]
    curvature_f = flight_features[:, 8]
    yaw_rate_f = flight_features[:, 9]

    M = len(flight_features)

    # longitudinal acceleration in straight analysis
    speed_xy_smooth = _safe_savgol(speed_xy_f, window_length=21, polyorder=2)
    a_long = np.gradient(speed_xy_smooth, dt)

    is_moving = speed_xy_f >= min_moving_speed
    has_valid_heading = speed_xy_f >= min_valid_speed_for_heading

    # --------------------------------------------------------------
    # Step 2. Explicit turn detection with hysteresis
    # --------------------------------------------------------------
    raw_turn_on = (
        has_valid_heading
        & (
            (np.abs(yaw_rate_f) >= yaw_rate_on)
            | (curvature_f >= curvature_on)
        )
    )

    raw_turn_off = (
        has_valid_heading
        & (
            (np.abs(yaw_rate_f) >= yaw_rate_off)
            | (curvature_f >= curvature_off)
        )
    )

    is_turn = np.zeros(M, dtype=bool)
    active = False
    for i in range(M):
        if not active:
            if raw_turn_on[i]:
                active = True
                is_turn[i] = True
        else:
            if raw_turn_off[i]:
                is_turn[i] = True
            else:
                active = False

    min_turn_len = max(1, int(round(min_turn_duration / dt)))
    merge_gap_len = max(0, int(round(max_merge_gap / dt)))

    is_turn = _fill_short_false_gaps(is_turn, merge_gap_len)
    is_turn = _remove_short_true_runs(is_turn, min_turn_len)

    # --------------------------------------------------------------
    # Step 3. Straight detection
    # --------------------------------------------------------------
    raw_straight = (
        is_moving
        & (~is_turn)
        & (np.abs(yaw_rate_f) <= straight_yaw_rate_max)
        & (curvature_f <= straight_curvature_max)
    )

    min_straight_len = max(1, int(round(min_straight_duration / dt)))
    is_straight = _fill_short_false_gaps(raw_straight, merge_gap_len)
    is_straight = _remove_short_true_runs(is_straight, min_straight_len)

    # --------------------------------------------------------------
    # Step 4. Unknown region
    # --------------------------------------------------------------
    is_unknown = ~(is_turn | is_straight)

    min_unknown_len = max(1, int(round(min_unknown_duration / dt)))
    is_unknown = _remove_short_true_runs(is_unknown, min_unknown_len)

    # normalize priority
    is_straight = is_straight & (~is_turn)
    is_unknown = ~(is_turn | is_straight)

    # --------------------------------------------------------------
    # Step 5. Export primary segments
    # --------------------------------------------------------------
    for s, e in _mask_to_ranges(is_turn):
        seg = _make_segment("turn", flight_t, flight_features, s, e)
        if seg is not None:
            segments["turn"].append(seg)
            spans["turn"].append(seg["span"])

    for s, e in _mask_to_ranges(is_straight):
        seg = _make_segment("straight", flight_t, flight_features, s, e)
        if seg is not None:
            segments["straight"].append(seg)
            spans["straight"].append(seg["span"])

    for s, e in _mask_to_ranges(is_unknown):
        seg = _make_segment("unknown", flight_t, flight_features, s, e)
        if seg is not None:
            segments["unknown"].append(seg)
            spans["unknown"].append(seg["span"])

    # --------------------------------------------------------------
    # Step 6. Straight sub-segmentation
    # --------------------------------------------------------------
    min_acc_len = max(1, int(round(min_accel_duration / dt)))

    for s, e in _mask_to_ranges(is_straight):
        local_a_long = a_long[s:e]

        local_accel = local_a_long >= long_acc_threshold
        local_decel = local_a_long <= -long_acc_threshold

        local_accel = _fill_short_false_gaps(local_accel, merge_gap_len)
        local_accel = _remove_short_true_runs(local_accel, min_acc_len)

        local_decel = _fill_short_false_gaps(local_decel, merge_gap_len)
        local_decel = _remove_short_true_runs(local_decel, min_acc_len)

        local_const = ~(local_accel | local_decel)

        for ls, le in _mask_to_ranges(local_accel):
            gs, ge = s + ls, s + le
            seg = _make_segment("straight_accel", flight_t, flight_features, gs, ge)
            if seg is not None:
                segments["straight_accel"].append(seg)
                spans["straight_accel"].append(seg["span"])

        for ls, le in _mask_to_ranges(local_decel):
            gs, ge = s + ls, s + le
            seg = _make_segment("straight_decel", flight_t, flight_features, gs, ge)
            if seg is not None:
                segments["straight_decel"].append(seg)
                spans["straight_decel"].append(seg["span"])

        for ls, le in _mask_to_ranges(local_const):
            gs, ge = s + ls, s + le
            seg = _make_segment("straight_const", flight_t, flight_features, gs, ge)
            if seg is not None:
                segments["straight_const"].append(seg)
                spans["straight_const"].append(seg["span"])

    return segments, spans