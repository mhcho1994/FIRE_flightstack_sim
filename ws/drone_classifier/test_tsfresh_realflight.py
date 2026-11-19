"""
Real flight classification using tsfresh + LightGBM pipeline.

Usage:
    python test_tsfresh_realflight.py [model.pkl] [features.json] [robust_stats.json]

If arguments are omitted, the script auto-selects the most recent files.
"""

import sys
import json
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from trajectory_processor import (
    load_robust_feature_scaler,
    process_rosbag_flight_data,
    transform_robust_features,
    feature_names,
)

WINDOW_SIZE = 25
STEP_SIZE   = 100
FEAT_NAMES  = feature_names()

FC_PARAMS = {
    "skewness":                  None,
    "kurtosis":                  None,
    "autocorrelation":           [{"lag": l} for l in [1, 2, 3, 5, 8]],
    "fft_coefficient":           [{"coeff": c, "attr": "abs"} for c in range(1, 7)],
    "sample_entropy":            None,
    "number_peaks":              [{"n": 2}, {"n": 5}],
    "absolute_sum_of_changes":   None,
    "longest_strike_above_mean": None,
    "mean_second_derivative_central": None,
}


def zscore_window(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu  = window.mean(axis=0)
    std = window.std(axis=0)
    return (window - mu) / (std + eps)


# ── helpers ──────────────────────────────────────────────────────────────────

def windows_to_tsfresh_df(windows: np.ndarray) -> pd.DataFrame:
    N, W, F = windows.shape
    ids  = np.repeat(np.arange(N), W)
    time = np.tile(np.arange(W), N)
    df = pd.DataFrame(windows.reshape(-1, F), columns=FEAT_NAMES)
    df.insert(0, "time", time)
    df.insert(0, "id",   ids)
    return df


def build_windows(feat_series: np.ndarray) -> np.ndarray:
    windows = []
    n = len(feat_series)
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        w = feat_series[start : start + WINDOW_SIZE]
        windows.append(zscore_window(w))
    return np.array(windows, dtype=np.float32) if windows else None


def auto_select(pattern, label):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {label} found matching '{pattern}'")
    chosen = files[-1]
    print(f"[AUTO] {label}: {chosen}")
    return chosen


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) == 4:
        model_path  = sys.argv[1]
        feat_path   = sys.argv[2]
        stats_path  = sys.argv[3]
    else:
        model_path  = auto_select("tsfresh_lgbm_*.pkl",      "model")
        feat_path   = auto_select("tsfresh_features_*.json", "feature list")
        stats_path  = auto_select("robust_stats_*.json",     "robust stats")

    print(f"\n{'='*80}")
    print("  tsfresh + LightGBM — Real Flight Classification")
    print(f"{'='*80}\n")

    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    with open(feat_path) as f:
        selected_features = json.load(f)
    robust_stats = load_robust_feature_scaler(stats_path)

    print(f"[OK] Model loaded ({model_path})")
    print(f"[OK] {len(selected_features)} selected features")
    print(f"[OK] Robust scaler loaded\n")

    csv_files = sorted(glob.glob(
        "/home/gayeonslee/FIRE/flightstack_sim/data/realflight/*.csv"
    ))
    print(f"Found {len(csv_files)} CSV files\n")

    all_results = []

    for csv_file in csv_files:
        fname = Path(csv_file).name
        segments = process_rosbag_flight_data(csv_file)
        if not segments:
            print(f"[FAIL] {fname}")
            continue

        for seg in segments:
            seg_idx   = seg["segment_index"]
            _, _, _, _, features, _, _ = seg["data"]

            if features is None or len(features) < WINDOW_SIZE:
                continue

            # robust-scale using SITL stats
            features_scaled = transform_robust_features(features, robust_stats)

            windows = build_windows(features_scaled)
            if windows is None or len(windows) == 0:
                continue

            # tsfresh extraction
            df = windows_to_tsfresh_df(windows)
            X_raw = extract_features(
                df,
                column_id="id",
                column_sort="time",
                default_fc_parameters=FC_PARAMS,
                impute_function=impute,
                n_jobs=0,
                show_warnings=False,
                disable_progressbar=True,
            )
            # keep only the features selected during training
            X = X_raw.reindex(columns=selected_features, fill_value=0.0).values

            # per-window probabilities → aggregate
            probs      = clf.predict_proba(X)           # (N_windows, 2)
            ardu_prob  = float(probs[:, 1].mean())
            prediction = "ArduPilot" if ardu_prob > 0.5 else "PX4"

            print(f"{fname}  seg{seg_idx}  "
                  f"ArduPilot={ardu_prob*100:.1f}%  → {prediction}  "
                  f"({len(windows)} windows)")

            all_results.append({
                "file":       fname,
                "segment":    seg_idx,
                "n_windows":  len(windows),
                "ardu_prob":  ardu_prob,
                "prediction": prediction,
            })

    print(f"\n{'='*80}")
    ardu_cnt = sum(1 for r in all_results if r["prediction"] == "ArduPilot")
    px4_cnt  = sum(1 for r in all_results if r["prediction"] == "PX4")
    print(f"Total segments: {len(all_results)}")
    print(f"  ArduPilot: {ardu_cnt}")
    print(f"  PX4:       {px4_cnt}")

    # save
    out_path = "tsfresh_realflight_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
