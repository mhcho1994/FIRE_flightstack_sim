import os
import numpy as np
import pywt
from pathlib import Path
from scipy.stats import kurtosis

from data_loader import load_px4_dataset, load_ardu_dataset, load_cogni_dataset

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

def main():
    cache_dir = Path("ws/drone_classifier_svm/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # 1. SITL Data Processing
    # =====================================================================
    sitl_cache_file = cache_dir / "sitl_features.npz"
    print("\n[Info] Processing SITL flight data...")
    X_px4_sitl, y_px4_sitl, _ = load_px4_dataset("sitl_logs")
    X_ardu_sitl, y_ardu_sitl, _ = load_ardu_dataset("sitl_logs", data_type='raw')
    
    X_sitl_ts = X_px4_sitl + X_ardu_sitl
    y_sitl = np.concatenate((y_px4_sitl, y_ardu_sitl))

    if len(X_sitl_ts) >= 5:
        print(f"[Info] Extracting DWT features for {len(X_sitl_ts)} SITL segments...")
        X_sitl_dwt = np.array([compute_dwt_statistics(ts) for ts in X_sitl_ts])
        np.savez(sitl_cache_file, X=X_sitl_dwt, y=y_sitl)
        print(f"[Success] Saved SITL features to '{sitl_cache_file}' (Shape: {X_sitl_dwt.shape})")
    else:
        print("[Warning] Not enough SITL data found.")

    # =====================================================================
    # 2. Real Flight Data Processing
    # =====================================================================
    real_cache_file = cache_dir / "real_features.npz"
    # Add more folders here as needed in the future
    test_folders = ["260501_flight_logs_old"] 
    X_real_ts, y_real, runs_real = [], [], []

    print("\n[Info] Processing Real flight data...")
    for folder in test_folders:
        X_p, y_p, r_p = load_px4_dataset(folder, data_type='processed', measurement_type='mocap')
        X_a, y_a, r_a = load_ardu_dataset(folder, data_type='processed', measurement_type='mocap')
        X_c, y_c, r_c = load_cogni_dataset(folder, data_type='processed', measurement_type='mocap')
        
        X_real_ts.extend(X_p + X_a + X_c)
        y_real.extend(y_p.tolist() + y_a.tolist() + y_c.tolist())
        runs_real.extend(r_p + r_a + r_c)

    if len(X_real_ts) > 0:
        print(f"[Info] Extracting DWT features for {len(X_real_ts)} Real flight segments...")
        X_real_dwt = np.array([compute_dwt_statistics(ts) for ts in X_real_ts])
        
        # Pre-filter NaN values during the saving stage to cache only clean data
        valid_indices = ~np.isnan(X_real_dwt).any(axis=1)
        removed_count = len(X_real_dwt) - np.sum(valid_indices)
        if removed_count > 0:
            print(f"[Warning] Removed {removed_count} corrupted segments containing NaN values.")
            
        X_real_dwt = X_real_dwt[valid_indices]
        y_real = np.array(y_real)[valid_indices]
        runs_real = np.array(runs_real)[valid_indices]

        np.savez(real_cache_file, X=X_real_dwt, y=y_real, runs=runs_real)
        print(f"[Success] Saved Real features to '{real_cache_file}' (Shape: {X_real_dwt.shape})")
    else:
        print("[Warning] No real flight data segments were loaded.")

    print("\n[Success] Feature extraction and caching completed successfully!")

if __name__ == "__main__":
    main()