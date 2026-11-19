import os
import numpy as np
import pywt
from pathlib import Path
from scipy.stats import kurtosis

# Import our new modular pipeline components
from data_extractor import parse_px4_ulog, parse_ardu_bin, parse_real_csv
from kinematic_processor import compute_kinematics_diff, FEATURE_MAP
from flight_segmenter import extract_segments

def compute_dwt_statistics(timeseries_matrix, waveletname='db4', level=3):
    """
    Extracts DWT-based statistical features from time-series data.
    """
    flight_features = []
    min_len = pywt.Wavelet(waveletname).dec_len * (2 ** level)
    
    if timeseries_matrix.shape[0] < min_len:
        pad_width = min_len - timeseries_matrix.shape[0]
        timeseries_matrix = np.pad(timeseries_matrix, ((0, pad_width), (0, 0)), mode='edge')
    
    for i in range(timeseries_matrix.shape[1]):
        signal = timeseries_matrix[:, i]
        coeffs = pywt.wavedec(signal, waveletname, level=level)
        
        coeffs_to_use = coeffs[:3] 
        for coeff in coeffs_to_use:
            mean_val = np.mean(coeff)
            std_val = np.std(coeff)
            energy_val = np.sum(np.square(coeff)) 
            max_val = np.max(coeff)
            min_val = np.min(coeff)
            kurt_val = kurtosis(coeff)
            peak_loc_max = np.argmax(coeff) / len(coeff)
            peak_loc_min = np.argmin(coeff) / len(coeff)
            
            flight_features.extend([
                mean_val, std_val, energy_val, max_val, min_val, 
                kurt_val, peak_loc_max, peak_loc_min
            ])
            
    return np.array(flight_features)

def pad_sequences(X_list):
    """
    Pads variable-length sequences with zeros to match the maximum length.
    Returns a 3D numpy array: (num_samples, max_length, num_features).
    """
    max_len = max(len(x) for x in X_list)
    num_features = X_list[0].shape[1]
    X_padded = np.zeros((len(X_list), max_len, num_features))
    
    for i, x in enumerate(X_list):
        X_padded[i, :len(x), :] = x
        
    return X_padded

def process_dataset_folder(base_folder, is_sitl=True, measurement_type='mocap'):
    """
    Crawls folders, runs the ETL pipeline (Extract -> Kinematics -> Segment), 
    and returns extracted Turn segments.
    """
    base_dir = Path("data") / base_folder
    X_ts, y, runs = [], [], []
    
    if not base_dir.exists():
        print(f"[Warning] Directory not found: {base_dir}")
        return X_ts, np.array(y), runs

    run_folders = sorted([f for f in os.listdir(base_dir) if f.startswith("run_") and (base_dir / f).is_dir()])
    
    # AI Classification Target Features
    target_features = ['XY-Accel', 'XY-Jerk', 'Curvature']
    target_indices = [FEATURE_MAP[f] for f in target_features]

    # Firmware configuration map (name, class_label, sub_path, extension)
    fw_configs = [
        ('px4', 0, "raw" if is_sitl else "processed", '.ulg' if is_sitl else '.csv'),
        ('ardu', 1, "raw/logs" if is_sitl else "processed", '.bin' if is_sitl else '.csv'),
        ('cogni', 2, "processed", '.csv')
    ]

    for run_folder in run_folders:
        run_dir = base_dir / run_folder
        
        for fw_name, class_label, sub_path, target_ext in fw_configs:
            # Skip Cogni if it's SITL data
            if is_sitl and fw_name == 'cogni': continue
                
            fw_dir = run_dir / f"{fw_name}_logs" / sub_path
            if not fw_dir.exists(): continue
                
            for file in os.listdir(fw_dir):
                if file.lower().endswith(target_ext):
                    file_path = str(fw_dir / file)
                    
                    # [Step 1: Extract Raw Data]
                    if is_sitl and fw_name == 'px4': raw_data = parse_px4_ulog(file_path)
                    elif is_sitl and fw_name == 'ardu': raw_data = parse_ardu_bin(file_path)
                    else: raw_data = parse_real_csv(file_path, measurement_type)
                    
                    if raw_data is None: continue
                    
                    # [Step 2 & 3: Transform (Kinematics -> Segment)]
                    t_full, feat_full = compute_kinematics_diff(raw_data)
                    segs, spans = extract_segments(t_full, feat_full)
                    
                    # [Step 4: Load Turn Features]
                    count_in_run = 0
                    if segs and segs['turn'] and len(segs['turn']) > 0:
                        for turn_segment in segs['turn']:
                            feat_turn = turn_segment['features'][:, target_indices]
                            X_ts.append(feat_turn)
                            y.append(class_label)
                            runs.append(f"{base_folder}/{run_folder}")
                            count_in_run += 1
                            
                    if count_in_run > 0:
                        print(f"    - {run_folder} [{fw_name.upper()}]: {count_in_run} turn segments extracted.")
                    break # Process only the first valid file per firmware

    return X_ts, np.array(y), runs

def main():
    cache_dir = Path("ws/drone_classifier_modules/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process SITL Logs
    print("\n[Info] Executing ETL Pipeline for SITL data...")
    X_sitl_ts, y_sitl, _ = process_dataset_folder("sitl_logs", is_sitl=True)
    
    if len(X_sitl_ts) > 0:
        print(f"[Info] Extracting DWT features for SITL data...")
        X_sitl_dwt = np.array([compute_dwt_statistics(ts) for ts in X_sitl_ts])
        X_sitl_seq = pad_sequences(X_sitl_ts)       # Pad sequences for 1D-CNN

        np.savez(cache_dir / "sitl_features.npz", X=X_sitl_dwt, X_seq=X_sitl_seq, y=y_sitl)
        print(f"[Success] Cached SITL features (Shape: {X_sitl_dwt.shape})")
    else:
        print("[Warning] No valid SITL features extracted.")

    # 2. Process Real Flight Logs
    print("\n[Info] Executing ETL Pipeline for Real flight data...")
    # test_folders = ["260417_flight_logs","260424_flight_logs", "260501_flight_logs_old"] 
    test_folders = ["260527_flight_logs_1", "260527_flight_logs_2"]
    # X_real_ts, y_real_list, runs_real_list = [], [], []
    
    for folder in test_folders:
        X_ts, y, runs = process_dataset_folder(folder, is_sitl=False, measurement_type='mocap')
        # X_real_ts.extend(X_ts)
        # y_real_list.extend(y.tolist())
        # runs_real_list.extend(runs)

    if len(X_ts) > 0:
        print(f"[Info] Extracting DWT features for Real flight data...")
        # X_real_dwt = np.array([compute_dwt_statistics(ts) for ts in X_real_ts])
        X_real_dwt = np.array([compute_dwt_statistics(ts) for ts in X_ts])
        # X_real_seq = pad_sequences(X_real_ts)            # Pad sequences for 1D-CNN
        X_real_seq = pad_sequences(X_ts)            # Pad sequences for 1D-CNN


        # Pre-filter NaN values during the caching stage
        valid_indices = ~np.isnan(X_real_dwt).any(axis=1)
        removed_count = len(X_real_dwt) - np.sum(valid_indices)
        if removed_count > 0:
            print(f"[Warning] Removed {removed_count} corrupted segments containing NaN values.")
            
        X_real_dwt = X_real_dwt[valid_indices]
        X_real_seq = X_real_seq[valid_indices]
        # y_real = np.array(y_real_list)[valid_indices]
        y_real = np.array(y)[valid_indices]
        # runs_real = np.array(runs_real_list)[valid_indices]
        runs_real = np.array(runs)[valid_indices]

        cache_file = cache_dir / f"{folder}_features.npz"
        np.savez(cache_file, X=X_real_dwt, X_seq=X_real_seq, y=y_real, runs=runs_real)
        # np.savez(cache_dir / "real_features.npz", X=X_real_dwt, X_seq=X_real_seq, y=y_real, runs=runs_real)
        print(f"[Success] Cached Real features (Shape: {X_real_dwt.shape})")
    else:
        print("[Warning] No valid real flight features extracted.")

if __name__ == "__main__":
    main()