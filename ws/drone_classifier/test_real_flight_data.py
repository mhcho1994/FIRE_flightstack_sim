"""
Real Flight Data Classification Test
Testing our trained PX4/ArduPilot classifier on actual _odom CSV log files
"""

import pandas as pd
import numpy as np
import torch
import glob
import json
from pathlib import Path
from model import DroneTrajectoryCNN, DroneTrajectoryCNNLSTM, DroneTrajectorMLP
from trajectory_processor import (
    load_robust_feature_scaler,
    process_rosbag_flight_data,
    transform_robust_features,
)
from dataset import compute_relative_window_features

print("="*90)
print("REAL FLIGHT DATA CLASSIFICATION TEST")
print("="*90)

# ==========================================
# 1. Find available model files
# ==========================================
print("\nSearching for model files...")
cnn_files = sorted(glob.glob('drone_cnn_*.pth'))
cnn_lstm_files = sorted(glob.glob('drone_cnn_lstm_*.pth'))

print(f"\nAvailable CNN models ({len(cnn_files)}):")
for i, f in enumerate(cnn_files):
    print(f"  [{i}] {f}")

print(f"\nAvailable CNN-LSTM models ({len(cnn_lstm_files)}):")
for i, f in enumerate(cnn_lstm_files):
    print(f"  [{i}] {f}")

# ==========================================
# 2. Load trained models
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# User selects CNN model
while True:
    try:
        cnn_idx = int(input(f"\nSelect CNN model [0-{len(cnn_files)-1}]: "))
        if 0 <= cnn_idx < len(cnn_files):
            cnn_path = cnn_files[cnn_idx]
            break
        print("Invalid selection. Try again.")
    except ValueError:
        print("Please enter a valid number.")

# User selects CNN-LSTM model
while True:
    try:
        cnn_lstm_idx = int(input(f"Select CNN-LSTM model [0-{len(cnn_lstm_files)-1}]: "))
        if 0 <= cnn_lstm_idx < len(cnn_lstm_files):
            cnn_lstm_path = cnn_lstm_files[cnn_lstm_idx]
            break
        print("Invalid selection. Try again.")
    except ValueError:
        print("Please enter a valid number.")

model_cnn = DroneTrajectoryCNN(num_features=7).to(device)
model_cnn.load_state_dict(torch.load(cnn_path, map_location=device))
model_cnn.eval()
print(f"\n[OK] Loaded: CNN model ({cnn_path})")

model_cnn_lstm = DroneTrajectoryCNNLSTM(
    num_features=7,
    lstm_hidden_size=32,
    num_lstm_layers=1
).to(device)
model_cnn_lstm.load_state_dict(torch.load(cnn_lstm_path, map_location=device))
model_cnn_lstm.eval()
print(f"[OK] Loaded: CNN-LSTM model ({cnn_lstm_path})")


def _robust_stats_candidate(model_path):
    stem = Path(model_path).stem
    for prefix in ("drone_cnn_lstm_", "drone_cnn_"):
        if stem.startswith(prefix):
            return f"robust_stats_{stem[len(prefix):]}.json"
    return None


def load_matching_robust_stats(cnn_path, cnn_lstm_path):
    candidates = [
        _robust_stats_candidate(cnn_path),
        _robust_stats_candidate(cnn_lstm_path),
    ]
    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            print(f"[OK] Loaded robust scaler stats: {candidate}")
            return load_robust_feature_scaler(candidate)

    stats_files = sorted(glob.glob("robust_stats_*.json"))
    if len(stats_files) == 1:
        print(f"[WARN] Matching robust stats not found; using only available stats: {stats_files[0]}")
        return load_robust_feature_scaler(stats_files[0])

    if len(stats_files) > 1:
        print("\nAvailable robust scaler stats:")
        for i, f in enumerate(stats_files):
            print(f"  [{i}] {f}")
        while True:
            try:
                stats_idx = int(input(f"Select robust scaler stats [0-{len(stats_files)-1}]: "))
                if 0 <= stats_idx < len(stats_files):
                    return load_robust_feature_scaler(stats_files[stats_idx])
                print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a valid number.")

    raise FileNotFoundError("No robust_stats_*.json found. Train once to fit and save SITL robust scaler stats.")


robust_stats = load_matching_robust_stats(cnn_path, cnn_lstm_path)

# ==========================================
# 3. Create windows from features
# ==========================================
def create_windows(features, kernel_size=25, step_size=50, relative=False):
    """Create windowed tensors from features.

    relative=False: returns (N, kernel_size, n_feat) for CNN/LSTM
    relative=True:  returns (N, n_feat) per-window scalar features for MLP
    """
    if features is None or len(features) < kernel_size:
        return None

    windows = []
    for start_idx in range(0, len(features) - kernel_size + 1, step_size):
        w = features[start_idx:start_idx + kernel_size]
        if relative:
            windows.append(compute_relative_window_features(w))
        else:
            windows.append(w)

    return np.array(windows) if windows else None

# ==========================================
# 4. Classification function
# ==========================================
def classify_windows(windows, models, device, relative=False):
    """Classify windowed features.

    relative=False: windows shape (N, n_feat, time) for CNN/LSTM
    relative=True:  windows shape (N, n_feat) scalars for MLP
    """
    if windows is None or len(windows) == 0:
        return None

    X = torch.from_numpy(windows).float().to(device)
    if not relative:
        X = X.permute(0, 2, 1)   # (N, time, feat) → (N, feat, time)

    results = {}
    with torch.no_grad():
        for model_name, model in models.items():
            out = model(X)
            prob = torch.softmax(out, dim=1)
            ardu_prob = prob[:, 1].mean().item()
            results[model_name] = {
                'prob_ardupilot': ardu_prob,
                'prob_px4': 1 - ardu_prob,
                'prediction': 'ArduPilot' if ardu_prob > 0.5 else 'PX4'
            }

    avg_ardu = np.mean([r['prob_ardupilot'] for r in results.values()])
    results['ensemble'] = {
        'prob_ardupilot': avg_ardu,
        'prob_px4': 1 - avg_ardu,
        'prediction': 'ArduPilot' if avg_ardu > 0.5 else 'PX4'
    }
    return results

# ==========================================
# 5. Process all CSV odom files
# ==========================================
print("\n" + "="*90)
print("FINDING AND PROCESSING CSV ODOM FILES")
print("="*90)

csv_files = sorted(glob.glob('/home/gayeonslee/FIRE/flightstack_sim/data/realflight/*.csv'))

print(f"\nFound {len(csv_files)} CSV odom files:")
for f in csv_files[:10]:
    print(f"   {Path(f).name}")
if len(csv_files) > 10:
    print(f"   ... and {len(csv_files)-10} more")

# Dict of models to test
models = {
    'cnn': model_cnn,
    'cnn_lstm': model_cnn_lstm,
}

classification_results = []

for csv_file in csv_files:  # Test all files
    print(f"\n{'='*90}")
    print(f"Processing: {Path(csv_file).name}")
    print(f"{'='*90}")
    
    # Load and process CSV file by continuous non-empty segments.
    processed_segments = process_rosbag_flight_data(csv_file)
    if not processed_segments:
        print("[FAIL] Failed to process CSV file")
        continue

    print(f"[OK] Processing {len(processed_segments)} continuous segment(s)")

    for segment in processed_segments:
        segment_idx = segment['segment_index']
        row_start = segment['row_start']
        row_end = segment['row_end']
        t_loc, traj_raw, t_resampled, traj_resampled, features, segments, spans = segment['data']
        # MLP uses raw time-series to compute relative features per window;
        # CNN/LSTM uses robust-scaled time-series directly.
        is_mlp = any(isinstance(m, DroneTrajectorMLP) for m in models.values())
        if not is_mlp:
            features = transform_robust_features(features, robust_stats)

        print(f"\n--- Segment {segment_idx} | rows {row_start}-{row_end} ---")
        print(f"[OK] Extracted {len(features)} feature samples")

        # Create windows
        windows = create_windows(features, relative=is_mlp)
        if windows is None:
            print("[FAIL] Failed to create windows")
            continue

        if is_mlp:
            # Apply robust scaling to the per-window scalar features
            windows = transform_robust_features(windows, robust_stats)

        print(f"[OK] Created {len(windows)} windows")

        # Classify
        print(f"Running inference...")
        results = classify_windows(windows, models, device, relative=is_mlp)
        
        if results is None:
            print("[FAIL] Classification failed")
            continue
        
        # Display results
        print(f"\nCLASSIFICATION RESULTS:")
        print(f"{'='*90}")
        
        for model_name, pred in results.items():
            if model_name == 'ensemble':
                print(f"\n[FINAL] {model_name.upper():15s}")
            else:
                print(f"\n{model_name.upper():15s}")
            print(f"   PX4:       {pred['prob_px4']*100:6.2f}%")
            print(f"   ArduPilot: {pred['prob_ardupilot']*100:6.2f}%")
            print(f"   → {pred['prediction']}")
        
        # Store result
        classification_results.append({
            'filename': Path(csv_file).name,
            'segment_index': segment_idx,
            'row_start': row_start,
            'row_end': row_end,
            't_start': segment['t_start'],
            't_end': segment['t_end'],
            'n_windows': len(windows),
            'ensemble_prediction': results['ensemble']['prediction'],
            'ensemble_ardupilot_prob': results['ensemble']['prob_ardupilot'],
            'cnn_prediction': results['cnn']['prediction'],
            'lstm_prediction': results['cnn_lstm']['prediction'],
        })

# ==========================================
# 6. Summary
# ==========================================
print("\n" + "="*90)
print("SUMMARY")
print("="*90)

if len(classification_results) > 0:
    df = pd.DataFrame(classification_results)
    print("\nResults:")
    print(df.to_string(index=False))
    
    ardu_count = (df['ensemble_prediction'] == 'ArduPilot').sum()
    px4_count = (df['ensemble_prediction'] == 'PX4').sum()
    
    print(f"\nEnsemble Predictions:")
    print(f"   ArduPilot: {ardu_count} segments")
    print(f"   PX4:       {px4_count} segments")
    
    # Save
    with open('real_flight_test_results.json', 'w') as f:
        json.dump({
            'test_date': '2026-03-30',
            'n_files': int(df['filename'].nunique()),
            'n_segments': len(df),
            'results': classification_results,
            'summary': {
                'ardupilot': int(ardu_count),
                'px4': int(px4_count)
            }
        }, f, indent=2)
    print(f"\n[OK] Saved: real_flight_test_results.json")
else:
    print("\n[FAIL] No results")

print("\n" + "="*90)
