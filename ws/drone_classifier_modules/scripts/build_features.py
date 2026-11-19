import argparse
import os
import sys

import numpy as np

# Add src/ to the python path so modules can be imported directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import config
import dataset_manager
from feature_builder import compute_dwt_statistics, pad_sequences


def main():
    parser = argparse.ArgumentParser(description="Extract features from flight logs.")
    parser.add_argument('--max-runs', type=int, default=None, help='Limit the number of runs processed (useful for testing)')
    args = parser.parse_args()

    cache_dir = config.CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process SITL Logs
    print("\n[Info] Executing ETL Pipeline for SITL data...")
    X_sitl_ts, _, y_sitl, _ = dataset_manager.process_dataset_folder(config.SITL_FOLDER, is_sitl=True, max_runs=args.max_runs)
    
    if len(X_sitl_ts) > 0:
        print(f"[Info] Extracting DWT features for SITL data...")
        X_sitl_dwt = np.array([compute_dwt_statistics(ts, config.WAVELET_NAME, config.WAVELET_LEVEL) for ts in X_sitl_ts])
        X_sitl_seq = pad_sequences(X_sitl_ts)       # Pad sequences for 1D-CNN

        cache_file = cache_dir / f"{config.SITL_FOLDER}_features.npz"
        np.savez(cache_file, X=X_sitl_dwt, X_seq=X_sitl_seq, y=y_sitl)
        print(f"[Success] Cached SITL features (Shape: {X_sitl_dwt.shape})")
    else:
        print("[Warning] No valid SITL features extracted.")

    # 2. Process Real Flight Logs
    print("\n[Info] Executing ETL Pipeline for Real flight data...")
    test_folders = config.REAL_FLIGHT_FOLDERS
    
    for folder in test_folders:
        X_ts, _, y, runs = dataset_manager.process_dataset_folder(folder, is_sitl=False, measurement_type='mocap', max_runs=args.max_runs)

        if len(X_ts) > 0:
            print(f"[Info] Extracting DWT features for Real flight data...")
            X_real_dwt = np.array([compute_dwt_statistics(ts, config.WAVELET_NAME, config.WAVELET_LEVEL) for ts in X_ts])
            X_real_seq = pad_sequences(X_ts)            # Pad sequences for 1D-CNN

            # Pre-filter NaN values during the caching stage
            valid_indices = ~np.isnan(X_real_dwt).any(axis=1)
            removed_count = len(X_real_dwt) - np.sum(valid_indices)
            if removed_count > 0:
                print(f"[Warning] Removed {removed_count} corrupted segments containing NaN values.")
                
            X_real_dwt = X_real_dwt[valid_indices]
            X_real_seq = X_real_seq[valid_indices]
            y_real = np.array(y)[valid_indices]
            runs_real = np.array(runs)[valid_indices]

            cache_file = cache_dir / f"{folder}_features.npz"
            np.savez(cache_file, X=X_real_dwt, X_seq=X_real_seq, y=y_real, runs=runs_real)
            print(f"[Success] Cached Real features (Shape: {X_real_dwt.shape})")
        else:
            print(f"[Warning] No valid real flight features extracted for {folder}.")


if __name__ == "__main__":
    main()
