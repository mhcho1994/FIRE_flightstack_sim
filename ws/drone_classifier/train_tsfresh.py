"""
tsfresh + LightGBM drone autopilot classifier.

Pipeline:
  SITL logs → 7-col kinematic time series
            → sliding windows (window_size=25, step_size=50)
            → tsfresh feature extraction (~hundreds of statistical features)
            → select_features (Benjamini-Yekutieli test against PX4/ArduPilot label)
            → LightGBM binary classifier
            → save model + selected feature names + robust scaler stats
"""

import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

from dataset import extract_flight_signatures
from trajectory_processor import (
    fit_robust_feature_scaler,
    save_robust_feature_scaler,
    transform_robust_features,
    feature_names,
)

# ── constants ────────────────────────────────────────────────────────────────
PX4_FOLDER   = "../../data/px4_logs"
ARDU_FOLDER  = "../../data/ardu_logs"
WINDOW_SIZE  = 25    # 0.5 s at 50 Hz
STEP_SIZE    = 100   # 2 s stride — fewer windows, faster extraction
FEAT_NAMES   = feature_names()   # 7 column names for tsfresh

# After within-window z-score normalisation, mean≈0 and variance≈1 for every
# window, so those statistics carry no discriminative signal.  We keep only
# features that describe *shape*, *temporal structure*, and *frequency content*.
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

# ── helpers ──────────────────────────────────────────────────────────────────

def windows_to_tsfresh_df(windows: np.ndarray, id_offset: int = 0) -> pd.DataFrame:
    """
    Convert (N, window_size, n_feat) array → long-format tsfresh DataFrame.
    Columns: id, time, feat0, feat1, ...
    """
    N, W, F = windows.shape
    ids  = np.repeat(np.arange(N) + id_offset, W)
    time = np.tile(np.arange(W), N)
    data = windows.reshape(-1, F)
    df = pd.DataFrame(data, columns=FEAT_NAMES)
    df.insert(0, "time", time)
    df.insert(0, "id",   ids)
    return df


def zscore_window(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Column-wise z-score within a single window → scale-invariant representation."""
    mu  = window.mean(axis=0)
    std = window.std(axis=0)
    return (window - mu) / (std + eps)


def build_windows(data_list, labels_list):
    """Slide windows; z-score each window column-wise before appending."""
    all_windows, all_labels = [], []
    for feat_series, label in zip(data_list, labels_list):
        n = len(feat_series)
        for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
            w = feat_series[start : start + WINDOW_SIZE]
            all_windows.append(zscore_window(w))
            all_labels.append(label)
    return np.array(all_windows, dtype=np.float32), np.array(all_labels, dtype=np.int64)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    robust_stats_path  = f"robust_stats_{ts}.json"
    model_out          = f"tsfresh_lgbm_{ts}.pkl"
    feat_names_out     = f"tsfresh_features_{ts}.json"

    print(f"\n{'='*70}")
    print(f"  tsfresh + LightGBM Trainer  ({ts})")
    print(f"{'='*70}\n")

    # ── 1. load SITL data ────────────────────────────────────────────────────
    from pathlib import Path as _P
    px4_files  = [(_P(p), 0) for p in sorted(_P(PX4_FOLDER).glob("*.ulg"))]
    ardu_files = [(_P(p), 1) for p in sorted(_P(ARDU_FOLDER).glob("*.bin"))]

    import random
    random.shuffle(px4_files); random.shuffle(ardu_files)

    test_ratio    = 0.2
    px4_test_cnt  = max(1, int(len(px4_files)  * test_ratio)) if len(px4_files)  > 1 else 0
    ardu_test_cnt = max(1, int(len(ardu_files) * test_ratio)) if len(ardu_files) > 1 else 0

    train_files = px4_files[px4_test_cnt:]  + ardu_files[ardu_test_cnt:]
    test_files  = px4_files[:px4_test_cnt]  + ardu_files[:ardu_test_cnt]

    print(f"Files — train: {len(train_files)}  test: {len(test_files)}")
    print("⏳ Loading train trajectories...")
    train_data, train_labels = extract_flight_signatures(train_files)
    print("⏳ Loading test trajectories...")
    test_data,  test_labels  = extract_flight_signatures(test_files)

    # ── 2. robust scaling (fit on train only) ────────────────────────────────
    print("📏 Fitting robust scaler...")
    robust_stats = fit_robust_feature_scaler(train_data)
    train_data   = [transform_robust_features(f, robust_stats) for f in train_data]
    test_data    = [transform_robust_features(f, robust_stats) for f in test_data]
    save_robust_feature_scaler(robust_stats, robust_stats_path)
    print(f"   Saved: {robust_stats_path}")

    # ── 3. sliding windows ───────────────────────────────────────────────────
    train_windows, train_y = build_windows(train_data, train_labels)
    test_windows,  test_y  = build_windows(test_data,  test_labels)
    print(f"Windows — train: {len(train_windows)}  test: {len(test_windows)}")

    # ── 4. tsfresh feature extraction ────────────────────────────────────────
    print("\n🔬 Extracting tsfresh features on TRAIN set (this takes a while)...")
    train_df = windows_to_tsfresh_df(train_windows, id_offset=0)
    train_series_y = pd.Series(train_y, index=np.arange(len(train_windows)))

    X_train_raw = extract_features(
        train_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=FC_PARAMS,
        impute_function=impute,
        n_jobs=0,
        show_warnings=False,
    )
    print(f"   Extracted {X_train_raw.shape[1]} features from {len(train_windows)} windows")

    # ── 5. feature selection ─────────────────────────────────────────────────
    print("\n🔍 Selecting statistically relevant features (Benjamini-Yekutieli)...")
    X_train_sel = select_features(X_train_raw, train_series_y, fdr_level=0.05)
    selected_features = list(X_train_sel.columns)
    print(f"   Selected {len(selected_features)} / {X_train_raw.shape[1]} features")

    with open(feat_names_out, "w") as f:
        json.dump(selected_features, f, indent=2)
    print(f"   Saved feature list: {feat_names_out}")

    # ── 6. tsfresh features on TEST set ──────────────────────────────────────
    print("\n🔬 Extracting tsfresh features on TEST set...")
    test_df = windows_to_tsfresh_df(test_windows, id_offset=0)
    X_test_raw = extract_features(
        test_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=FC_PARAMS,
        impute_function=impute,
        n_jobs=0,
        show_warnings=False,
    )
    X_test_sel = X_test_raw.reindex(columns=selected_features, fill_value=0.0)

    # ── 7. LightGBM training ─────────────────────────────────────────────────
    print("\n🌲 Training LightGBM...")

    X_tr = X_train_sel.values
    X_te = X_test_sel.values
    y_tr = train_y
    y_te = test_y

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    y_pred = clf.predict(X_te)
    test_acc = accuracy_score(y_te, y_pred) * 100
    print(f"\nTest accuracy: {test_acc:.1f}%")
    print(classification_report(y_te, y_pred, target_names=["PX4", "ArduPilot"]))

    # ── 8. save model ────────────────────────────────────────────────────────
    with open(model_out, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved: {model_out}")

    # top-20 important features
    importances = sorted(
        zip(selected_features, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop-20 most important features:")
    for name, imp in importances[:20]:
        print(f"  {imp:6.0f}  {name}")


if __name__ == "__main__":
    main()
