import numpy as np

def extract_segments(t, features, dt=0.02, min_turn_duration=0.5, 
                     yaw_rate_threshold=0.2, straight_duration=1.0, heading_margin=0.2):
    """
    Analyzes kinematic features to split the flight into takeoff, straight, turn, and landing segments.
    """
    spans = {'takeoff': None, 'landing': None, 'straight': [], 'turn': []}
    segments = {'takeoff': None, 'landing': None, 'straight': [], 'turn': []}

    if features is None or len(features) < 2:
        return segments, spans
        
    N = len(features)
    alt = features[:, 0]
    vz_norm = np.abs(features[:, 2])
    az_norm = np.abs(features[:, 4])
    
    alt_50 = np.percentile(alt, 50)
    target_alt = alt_50 - 0.1
    
    # [Step 1] Landing/Take-off Detection
    flight_start_idx = 0
    state = 0
    for i in range(N):
        if state == 0 and alt[i] >= target_alt: state = 1
        elif state == 1 and vz_norm[i] <= 0.2: state = 2
        elif state == 2 and az_norm[i] <= 0.2:
            flight_start_idx = i
            break
                
    flight_end_idx = N
    state = 0
    for i in range(N - 1, flight_start_idx, -1):
        if state == 0 and alt[i] >= target_alt: state = 1
        elif state == 1 and vz_norm[i] <= 0.2: state = 2
        elif state == 2 and az_norm[i] <= 0.2:
            flight_end_idx = i
            break

    if flight_start_idx > 0:
        spans['takeoff'] = (t[0], t[flight_start_idx])
        segments['takeoff'] = {
            'label': 'takeoff', 'time': t[0:flight_start_idx],
            'features': features[0:flight_start_idx], 'span': (t[0], t[flight_start_idx])
        }
    
    if flight_end_idx < N - 1:
        spans['landing'] = (t[flight_end_idx], t[-1])
        segments['landing'] = {
            'label': 'landing', 'time': t[flight_end_idx:],
            'features': features[flight_end_idx:], 'span': (t[flight_end_idx], t[-1])
        }
                
    flight_t = t[flight_start_idx:flight_end_idx]
    flight_features = features[flight_start_idx:flight_end_idx]
    
    if len(flight_features) == 0:
        return segments, spans
        
    # [Step 2] Straight Flight Detection
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
            'label': 'straight', 'time': flight_t[start:safe_end],
            'features': flight_features[start:safe_end], 'span': (flight_t[start], flight_t[safe_end])
        })

    straight_indices = np.where(is_straight)[0]
    if len(straight_indices) == 0:
        return segments, spans 
        
    first_straight_idx = straight_indices[0]

    # [Step 3] Turn Detection
    is_turning = ~is_straight 
    is_turning[:first_straight_idx] = False
    
    edges = np.diff(is_turning.astype(int))
    turn_starts = np.where(edges == 1)[0] + 1
    turn_ends = np.where(edges == -1)[0] + 1
    
    if is_turning[0]: turn_starts = np.insert(turn_starts, 0, 0)
    if is_turning[-1]: turn_ends = np.append(turn_ends, len(is_turning))
        
    min_length = int(min_turn_duration / dt) 
    
    for start, end in zip(turn_starts, turn_ends):
        if (end - start) < min_length: continue
            
        safe_end = min(end, len(flight_t) - 1)
        spans['turn'].append((flight_t[start], flight_t[safe_end]))
        segments['turn'].append({
            'label': 'turn', 'time': flight_t[start:safe_end],
            'features': flight_features[start:safe_end], 'span': (flight_t[start], flight_t[safe_end])
        })
        
    return segments, spans