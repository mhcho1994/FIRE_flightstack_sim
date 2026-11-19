import config
import numpy as np

# ==============================================================================
# [Module 1] Motion Primitive Probability & HMM Smoothing
# ==============================================================================

def _calculate_emission_probs(features, hover_thresh=0.5, pitch_ascending=45.0, 
                              pitch_descending=-45.0, vz_ascending=0.5, vz_descending=-0.5, 
                              turn_sharpness=0.01):
    """
    Computes a soft probability distribution over 6 motion primitives.
    States: 0:Hovering, 1:Ascending, 2:Descending, 3:Straight, 4:Left_Turn, 5:Right_Turn
    """
    n_frames = len(features['t_window'])
    prob_matrix = np.zeros((n_frames, 6))

    for i in range(n_frames):
        vx, vy, vz = features['vx_reg'][i], features['vy_reg'][i], features['vz_reg'][i]
        pitch = features['pitch_deg'][i]
        sharpness = features['turn_sharpness'][i]
        direction = features['turn_direction'][i]

        speed_3d = np.sqrt(vx**2 + vy**2 + vz**2)
        scores = np.zeros(6)

        # 1. Hovering Score
        scores[0] = max(0.0, 1 - (speed_3d / hover_thresh))

        # 2 & 3. Ascending and Descending Scores
        if pitch > 0 and vz > 0:
            scores[1] = min(1.0, pitch / pitch_ascending) * min(1.0, vz / vz_ascending)
        elif pitch < 0 and vz < 0:
            scores[2] = min(1.0, pitch / pitch_descending) * min(1.0, vz / vz_descending)

        # 4 & 5. Turn Left / Right Scores
        turn_score = min(1.0, sharpness / (turn_sharpness * 1.0))
        if direction > 0:
            scores[4] = turn_score
        else:
            scores[5] = turn_score

        # 6. Straight Score (Fallback)
        max_other_score = np.max([scores[0], scores[1], scores[2], scores[4], scores[5]])
        scores[3] = max(0.0, 1.0 - max_other_score)

        # Normalization
        scores += 1e-6
        prob_matrix[i, :] = scores / np.sum(scores)

    return prob_matrix

def _smooth_with_viterbi(emission_probs):
    """
    Decodes the most likely sequence of primitives using Viterbi algorithm.
    """
    n_frames, n_states = emission_probs.shape
    pi = config.HMM_PI # Initial probs
    
    # Transition Matrix A (Physical Rules Engine)
    A = config.HMM_A
    
    eps = 1e-10
    log_pi, log_A, log_B = np.log(pi + eps), np.log(A + eps), np.log(emission_probs + eps)
    
    V = np.zeros((n_frames, n_states))
    ptr = np.zeros((n_frames, n_states), dtype=int)
    
    V[0, :] = log_pi + log_B[0, :]
    for t in range(1, n_frames):
        for j in range(n_states):
            seq_probs = V[t-1, :] + log_A[:, j]
            best_prev_state = np.argmax(seq_probs)
            ptr[t, j] = best_prev_state
            V[t, j] = seq_probs[best_prev_state] + log_B[t, j]
            
    best_path = np.zeros(n_frames, dtype=int)
    best_path[-1] = np.argmax(V[-1, :])
    for t in range(n_frames - 2, -1, -1):
        best_path[t] = ptr[t+1, best_path[t+1]]
        
    state_names = ["hovering", "ascending", "descending", "straight", "turn_left", "turn_right"]
    return np.array([state_names[idx] for idx in best_path])

def get_segmentation_details(features):
    """
    Public interface to retrieve both emission probabilities and Viterbi smoothed labels.
    """
    probs = _calculate_emission_probs(features)
    viterbi_labels = _smooth_with_viterbi(probs)
    return probs, viterbi_labels

# ==============================================================================
# [Module 2] Legacy Interface Adapter (Plug-and-Play Output)
# ==============================================================================

def extract_segments(features_dict, dt=None, min_turn_duration=None, 
                     yaw_rate_threshold=None, straight_duration=None, heading_margin=None):
    """
    Analyzes geometric features (from kinematic_processor.py) to split the flight.
    Returns the exact same `segments` and `spans` dictionary structure as the legacy code.
    
    Note: Legacy arguments (dt, yaw_rate_threshold, etc.) are ignored as the new
    pipeline relies on PCA, Regression, and HMM parameters internally.
    """
    # Output structure identical to legacy code (now including hovering)
    spans = {'ascending': [], 'descending': [], 'straight': [], 'turn_left': [], 'turn_right': [], 'hovering': []}
    segments = {'ascending': [], 'descending': [], 'straight': [], 'turn_left': [], 'turn_right': [], 'hovering': []}

    if features_dict is None or len(features_dict['t_window']) < 2:
        return segments, spans

    t_window = features_dict['t_window']
    n_frames = len(t_window)

    # 1. Run the Probabilistic & HMM pipeline
    emission_probs = _calculate_emission_probs(features_dict)
    smoothed_labels = _smooth_with_viterbi(emission_probs)

    # 2. Parse the contiguous blocks of labels and populate the legacy dictionary
    # We find the start and end indices of identical consecutive labels.
    current_label = smoothed_labels[0]
    start_idx = 0

    for i in range(1, n_frames + 1):
        # Trigger block extraction if label changes OR end of array is reached
        if i == n_frames or smoothed_labels[i] != current_label:
            end_idx = i - 1
            span_end_time = t_window[i] if i < n_frames else t_window[end_idx]
            span = (t_window[start_idx], span_end_time)
            
            # Reconstruct a subset of the feature dictionary for this block
            # (Matches the legacy expectation of providing subset features)
            subset_features = {k: v[start_idx:end_idx+1] for k, v in features_dict.items()}
            
            segment_data = {
                'label': current_label,
                'time': t_window[start_idx:end_idx+1],
                'features': subset_features,
                'span': span
            }

            # Map to dictionary keys (all are lists now)
            spans[current_label].append(span)
            segments[current_label].append(segment_data)

            # Reset for the next block
            if i < n_frames:
                current_label = smoothed_labels[i]
                start_idx = i

    return segments, spans


if __name__ == "__main__":
    import data_extractor
    import kinematic_processor
    import matplotlib.pyplot as plt

    # 1. Parse raw data
    log_path = "./data/sitl_logs/run_003/ardu_logs/raw/logs/00000001.BIN"
    print(f"[Info] Parsing raw flight data from:\n       {log_path}")
    raw_flight_data = data_extractor.parse_ardu_bin(log_path)

    if raw_flight_data is None:
        print("[Error] Failed to parse the specified ArduPilot BIN file.")
    else:
        print("[Info] Data successfully extracted. Processing kinematics...")
        
        # 2. Extract features
        features = kinematic_processor.compute_kinematics_pca(raw_flight_data)

        if features is None:
            print("[Error] Could not process kinematics (insufficient data).")
        else:
            print("[Info] Kinematics processed successfully. Calculating emission probabilities...")
            
            # 3. Calculate emission probabilities
            probs = _calculate_emission_probs(features)
            viterbi_labels = _smooth_with_viterbi(probs)
            
            t = features['t_window']
            state_config = {
                0: {'key': 'hovering',   'name': 'Hovering',   'color': 'tab:gray'},
                1: {'key': 'ascending',  'name': 'Ascending',  'color': 'tab:orange'},
                2: {'key': 'descending', 'name': 'Descending', 'color': 'tab:blue'},
                3: {'key': 'straight',   'name': 'Straight',   'color': 'tab:green'},
                4: {'key': 'turn_left',  'name': 'Left Turn',  'color': 'tab:purple'},
                5: {'key': 'turn_right', 'name': 'Right Turn', 'color': 'tab:brown'}
            }


            state_names = [state_config[i]['name'] for i in range(6)]
            colors = [state_config[i]['color'] for i in range(6)]
            label_map = {state_config[i]['key']: i for i in range(6)}
            numeric_labels = [label_map[l] for l in viterbi_labels]


            # 4. Create plot using subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1.2]}, sharex=True)
            
            # Subplot 1: Stacked area chart for emission probabilities
            ax1.stackplot(t, probs.T, labels=state_names, colors=colors, alpha=0.8)
            ax1.set_title('HMM Emission Probabilities & Viterbi Decoding over Time', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Probability')
            ax1.set_ylim(0, 1.0)
            ax1.set_xlim(t[0], t[-1])
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.4)
            
            # Subplot 2: Viterbi Decoded State
            ax2.step(t, numeric_labels, where='post', color='black', linewidth=1.5, zorder=2)
            
            # Color background spans for better visibility
            start_idx = 0
            for i in range(1, len(t)):
                if numeric_labels[i] != numeric_labels[i-1] or i == len(t)-1:
                    ax2.axvspan(t[start_idx], t[i], color=colors[numeric_labels[start_idx]], alpha=0.4, lw=0, zorder=1)
                    start_idx = i
                    
            ax2.set_yticks(range(6))
            ax2.set_yticklabels(state_names)
            ax2.set_ylabel('Viterbi State')
            ax2.set_xlabel('Time (s)')
            ax2.grid(True, linestyle='--', alpha=0.4, axis='x')
            
            plt.tight_layout()
            
            # Save and optionally display
            save_file = "emission_probs_test_result.png"
            plt.savefig(save_file, dpi=300)
            plt.close()
            print(f"[Success] Plot saved to '{save_file}'")