import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import argparse

import config
# Import common evaluation utilities
from evaluation_utils import (plot_confusion_matrix, plot_pca_2d_projection,
                              print_detailed_prediction_map)


def main():
    parser = argparse.ArgumentParser(description="Train SVM Model on SITL features and evaluate on Real Data.")
    parser.add_argument("--sitl-folder", type=str, default=config.SITL_FOLDER, help="Name of the SITL folder to load features from.")
    parser.add_argument("--no-real", action="store_true", help="Disable evaluation on real flight data.")
    args = parser.parse_args()

    cache_dir = config.CACHE_DIR
    sitl_cache = cache_dir / f"{args.sitl_folder}_features.npz"

    # =====================================================================
    # 1. Load Pre-extracted Features
    # =====================================================================
    if not sitl_cache.exists():
        print(f"[Error] Feature cache not found. Please run 'feature_builder.py' first.")
        return

    print("\n[Info] Loading cached DWT features...")
    sitl_data = np.load(sitl_cache)
    X_sitl, y_sitl = sitl_data['X'], sitl_data['y']
    print(f"  -> SITL Features Loaded (Shape: {X_sitl.shape})")

    # =====================================================================
    # 2. Train Model (SITL)
    # =====================================================================
    X_train, X_test, y_train, y_test = train_test_split(X_sitl, y_sitl, test_size=0.1, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n[Info] Training SVM model...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    target_names_sitl = ['PX4', 'ArduPilot']

    print("\n================ SITL Classification Results ================")
    print(f"Accuracy on SITL Test Data: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=target_names_sitl, zero_division=0))

    # =====================================================================
    # 3. Evaluate on Real Flight Data
    # =====================================================================
    if args.no_real:
        print("\n[Info] Skipping real flight evaluation as requested.")
        return

    test_folders = config.REAL_FLIGHT_FOLDERS
    
    X_real_list, y_real_list, runs_real_list = [], [], []

    print("\n[Info] Loading cached Real Flight features...")
    for folder in test_folders:
        real_cache = cache_dir / f"{folder}_features.npz"
        if real_cache.exists():
            real_data = np.load(real_cache)
            X_real_list.append(real_data['X'])
            y_real_list.append(real_data['y'])
            runs_real_list.append(real_data['runs'])
            print(f"  -> Loaded '{folder}' features.")

    if len(X_real_list) > 0:
        X_real = np.vstack(X_real_list)
        y_real = np.hstack(y_real_list)
        runs_real = np.hstack(runs_real_list)

        print(f"\n[Info] Testing on Real Flight Features (Shape: {X_real.shape})...")
        
        X_real_scaled = scaler.transform(X_real)
        y_real_pred = model.predict(X_real_scaled)
        
        correct_count = np.sum(y_real_pred == y_real)
        print(f"Final Real Data Accuracy: {(correct_count / len(y_real)) * 100:.2f}% ({correct_count}/{len(y_real)})")
        
        print_detailed_prediction_map(y_real, y_real_pred, runs_real)
        
        print("\n[Real Data Classification Report]")
        # print(classification_report(y_real, y_real_pred, labels=[0, 1, 2], target_names=['PX4', 'ArduPilot', 'Cogni'], zero_division=0))
        print(classification_report(y_real, y_real_pred, labels=[0, 1], target_names=['PX4', 'ArduPilot'], zero_division=0))    # Quick Fix for the current dataset imbalance (no Cogni samples)

        # Call visualization functions (passing the SVM model name)
        # plot_confusion_matrix(y_real, y_real_pred, target_names=['PX4', 'ArduPilot', 'Cogni'], model_name="SVM")
        plot_confusion_matrix(y_real, y_real_pred, target_names=['PX4', 'ArduPilot'], model_name="SVM")                         # Quick Fix for the current dataset imbalance (no Cogni samples)
        plot_pca_2d_projection(X_train_scaled, X_test_scaled, y_train, y_test, X_real_scaled, y_real, y_real_pred, model_name="SVM")

    else:
        print("\n[Warning] No real flight cache found. Skipping real flight evaluation.")
        # Draw plots based on SITL even if there is no real flight data
        plot_confusion_matrix(y_test, y_pred, target_names=target_names_sitl, model_name="SVM")
        plot_pca_2d_projection(X_train_scaled, X_test_scaled, y_train, y_test, model_name="SVM")

if __name__ == "__main__":
    main()