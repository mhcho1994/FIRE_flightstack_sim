"""
DWT + LightGBM drone autopilot classifier.

Key idea (from drone_classifier_svm/train_dwt_svm.py, enhanced):
  1. Extract TURN segments from SITL flights — turns force the control loop
     to respond to lateral maneuvers, making PX4 vs ArduPilot differences visible.
  2. Z-score normalize each signal channel WITHIN the segment so all DWT
     statistics become scale-invariant (works for both fast SITL and slow real flight).
  3. Apply DWT (db4, level=3) → cA3 (trend), cD3, cD2 (oscillation detail).
  4. Extract 8 statistics per coefficient: mean, std, energy, max, min,
     kurtosis, peak_loc_max, peak_loc_min.
  5. LightGBM (better generalisation than SVM for tabular data).

Channels used per turn segment (3 × 3 coefficients × 8 stats = 72 features):
  ch0: speed_xy       — horizontal speed profile (control smoothness)
  ch1: ah             — vertical acceleration (altitude hold during turn)
  ch2: yaw_rate       — yaw execution smoothness
"""

import sys
import json
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import lightgbm as lgb
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score, classification_report

# ── constants ─────────────────────────────────────────────────────────────────
PX4_FOLDER  = "../../data/px4_logs"
ARDU_FOLDER = "../../data/ardu_logs"
TEST_RATIO  = 0.2
WAVELET     = "db4"
LEVEL       = 3
MIN_SEG_LEN = 25   # minimum samples in a turn segment (0.5 s at 50 Hz)

# ── helpers ───────────────────────────────────────────────────────────────────
def _safe_savgol(signal, window_length=21, poly_order=3):
    n = len(signal)
    wl = min(int(window_length), n)
    if wl % 2 == 0: wl -= 1
    if wl <= poly_order or wl < 2: return signal.copy()
    return savgol_filter(signal, window_length=wl, polyorder=poly_order)


def _mask_to_ranges(mask):
    if len(mask) == 0: return []
    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if mask[0]:  starts = [0] + starts
    if mask[-1]: ends = ends + [len(mask)]
    return list(zip(starts, ends))


def _remove_short(mask, min_len):
    out = mask.copy()
    for s, e in _mask_to_ranges(mask):
        if (e - s) < min_len: out[s:e] = False
    return out


def _fill_gaps(mask, max_gap):
    out = mask.copy()
    for s, e in _mask_to_ranges(~mask):
        if s == 0 or e == len(mask): continue
        if (e - s) <= max_gap: out[s:e] = True
    return out


# ── turn extraction ───────────────────────────────────────────────────────────
def extract_turns(t, feat7col, traj_resampled, dt=0.02):
    """
    Return list of turn-segment dicts from a single trajectory.
    Adaptive speed thresholds — works for both fast SITL and slow real flight.

    feat7col cols: 0=vh, 1=speed_xy, 2=ah, 3=curvature, 4=yaw_rate, 5=yaw_aa, 6=spd_curv
    traj_resampled: (N, 6) — x, y, z, vx, vy, vz
    """
    N = len(feat7col)
    speed_xy = feat7col[:, 1]
    yaw_rate = feat7col[:, 4]
    curvature = feat7col[:, 3]

    pos_spd = speed_xy[speed_xy > 0.02]
    spd_ref = float(np.percentile(pos_spd, 70)) if len(pos_spd) > 10 else 1.0
    min_heading_speed = max(0.08, 0.35 * spd_ref)

    has_heading = speed_xy >= min_heading_speed

    yaw_rate_on  = 0.15
    yaw_rate_off = 0.08
    curv_on      = 0.03
    curv_off     = 0.015
    min_turn_len = max(MIN_SEG_LEN, int(0.4 / dt))
    merge_gap    = max(0, int(0.25 / dt))

    raw_on  = has_heading & ((np.abs(yaw_rate) >= yaw_rate_on)  | (curvature >= curv_on))
    raw_off = has_heading & ((np.abs(yaw_rate) >= yaw_rate_off) | (curvature >= curv_off))

    is_turn = np.zeros(N, dtype=bool)
    active = False
    for i in range(N):
        if not active:
            if raw_on[i]: active = True; is_turn[i] = True
        else:
            if raw_off[i]: is_turn[i] = True
            else: active = False

    is_turn = _fill_gaps(is_turn, merge_gap)
    is_turn = _remove_short(is_turn, min_turn_len)

    turns = []
    for s, e in _mask_to_ranges(is_turn):
        seg_feat = feat7col[s:e]
        turns.append({
            "speed_xy":  seg_feat[:, 1],
            "ah":        seg_feat[:, 2],
            "yaw_rate":  seg_feat[:, 4],
            "n":         e - s,
        })
    return turns


# ── DWT feature extraction ────────────────────────────────────────────────────
def _dwt_stats(coeff):
    """8 statistics from one DWT coefficient array."""
    n = len(coeff)
    if n == 0:
        return [0.0] * 8
    return [
        float(np.mean(coeff)),
        float(np.std(coeff)),
        float(np.sum(np.square(coeff))),
        float(np.max(coeff)),
        float(np.min(coeff)),
        float(scipy_kurtosis(coeff)),
        float(np.argmax(coeff) / max(n - 1, 1)),   # peak loc max (0–1)
        float(np.argmin(coeff) / max(n - 1, 1)),   # peak loc min (0–1)
    ]


def zscore(x, eps=1e-8):
    """Z-score normalise 1D array."""
    return (x - np.mean(x)) / (np.std(x) + eps)


def extract_dwt_features(turn_seg):
    """
    72-dim feature vector from one turn segment.
    Channels: speed_xy, ah, yaw_rate — each z-scored before DWT.
    """
    min_len = pywt.Wavelet(WAVELET).dec_len * (2 ** LEVEL)
    channels = [turn_seg["speed_xy"], turn_seg["ah"], turn_seg["yaw_rate"]]
    feats = []
    for ch in channels:
        sig = zscore(ch.astype(np.float64))
        # pad if too short
        if len(sig) < min_len:
            sig = np.pad(sig, (0, min_len - len(sig)), mode="edge")
        coeffs = pywt.wavedec(sig, WAVELET, level=LEVEL)
        for coeff in coeffs[:3]:        # cA3, cD3, cD2
            feats.extend(_dwt_stats(coeff))
    return np.array(feats, dtype=np.float32)   # 3 ch × 3 coeffs × 8 stats = 72


# ── data loading ──────────────────────────────────────────────────────────────
def load_turns_from_files(file_list):
    """Load SITL logs, extract turn segments, return (feature_matrix, labels)."""
    from trajectory_processor import process_px4_flight_data, process_ardu_flight_data

    X, y = [], []
    for path, label in file_list:
        path = str(path)
        if label == 0:
            result = process_px4_flight_data(path)
        else:
            result = process_ardu_flight_data(path)

        if result is None or result[4] is None:
            continue
        _, _, t_res, traj_res, feat7, _, _ = result
        if len(feat7) < MIN_SEG_LEN:
            continue

        dt = 1.0 / 50.0
        turns = extract_turns(t_res, feat7, traj_res, dt=dt)
        for turn in turns:
            if turn["n"] < MIN_SEG_LEN:
                continue
            fv = extract_dwt_features(turn)
            X.append(fv)
            y.append(label)

    return np.array(X, dtype=np.float32) if X else np.empty((0, 72)), np.array(y)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_out = f"dwt_lgbm_{ts}.pkl"

    print(f"\n{'='*70}")
    print(f"  DWT + LightGBM Trainer  ({ts})")
    print(f"{'='*70}\n")

    px4_files  = [(p, 0) for p in sorted(Path(PX4_FOLDER).glob("*.ulg"))]
    ardu_files = [(p, 1) for p in sorted(Path(ARDU_FOLDER).glob("*.bin"))]
    random.shuffle(px4_files); random.shuffle(ardu_files)

    px4_test  = max(1, int(len(px4_files)  * TEST_RATIO)) if len(px4_files)  > 1 else 0
    ardu_test = max(1, int(len(ardu_files) * TEST_RATIO)) if len(ardu_files) > 1 else 0

    train_files = px4_files[px4_test:]  + ardu_files[ardu_test:]
    test_files  = px4_files[:px4_test]  + ardu_files[:ardu_test]

    print(f"Files — train: {len(train_files)}  test: {len(test_files)}")
    print("⏳ Extracting DWT features from train set...")
    X_train, y_train = load_turns_from_files(train_files)
    print("⏳ Extracting DWT features from test set...")
    X_test,  y_test  = load_turns_from_files(test_files)

    print(f"\nTurn segments — train: {len(X_train)}  test: {len(X_test)}")

    unique, counts = np.unique(y_train, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        name = "PX4" if lbl == 0 else "ArduPilot"
        print(f"  {name}: {cnt} ({100*cnt/len(y_train):.1f}%)")

    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=5,
        reg_alpha=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\nTest accuracy (SITL): {acc:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["PX4", "ArduPilot"]))

    with open(model_out, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved: {model_out}")

    # top features
    feat_names = [
        f"ch{c}_coeff{k}_{s}"
        for c in range(3)
        for k in range(3)
        for s in ["mean","std","energy","max","min","kurt","peak_max","peak_min"]
    ]
    importances = sorted(zip(feat_names, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nTop-15 features:")
    for name, imp in importances[:15]:
        ch_label = ["speed_xy", "ah", "yaw_rate"][int(name[2])]
        coeff_label = ["cA3", "cD3", "cD2"][int(name[8])]
        print(f"  {imp:5.0f}  {ch_label}__{coeff_label}__{name.split('_',3)[-1]}")


if __name__ == "__main__":
    main()
