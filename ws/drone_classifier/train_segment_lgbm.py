"""
Segment-based PX4 vs ArduPilot classifier.

Pipeline:
  SITL logs → 7-col kinematic time series (drone_classifier processor)
            → adapt to 10-col format (add altitude from traj_resampled)
            → adaptive flight segmentation (straight / turn / takeoff / landing)
            → extract per-segment scale-invariant ratio features
            → LightGBM (one row per segment)

Scale-invariant features used:
  - a_long_to_speed_ratio       (acceleration character)
  - yaw_rate_to_speed_ratio     (turn curvature)
  - speed_cv                    (speed regulation quality)
  - yaw_rate_cv                 (turn smoothness)
  - v_alt_cv                    (altitude hold quality)
  - peak_speed_rel_idx          (where in segment speed peaks)
  - peak_yaw_rate_rel_idx       (where in segment yaw rate peaks)
  - integral_yaw_rate_per_path  (total rotation per unit distance)
  - heading_change_norm         (heading change per unit path)
  + one-hot segment type
"""

import sys
import json
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score, classification_report

from dataset import extract_flight_signatures

# ── paths ─────────────────────────────────────────────────────────────────────
PX4_FOLDER  = "../../data/px4_logs"
ARDU_FOLDER = "../../data/ardu_logs"
TEST_RATIO  = 0.2

# ── segmentor helpers (inlined from trajectory_processing/trajectory_segmentor.py) ─
def _safe_savgol(signal, window_length=21, poly_order=3):
    n = len(signal)
    wl = min(int(window_length), n)
    if wl % 2 == 0:
        wl -= 1
    if wl <= poly_order or wl < 2:
        return signal.copy()
    return savgol_filter(signal, window_length=wl, polyorder=poly_order)


def _mask_to_ranges(mask):
    if len(mask) == 0:
        return []
    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if mask[0]:  starts = [0] + starts
    if mask[-1]: ends = ends + [len(mask)]
    return list(zip(starts, ends))


def _remove_short_true_runs(mask, min_len):
    out = mask.copy()
    for s, e in _mask_to_ranges(mask):
        if (e - s) < min_len:
            out[s:e] = False
    return out


def _fill_short_false_gaps(mask, max_gap):
    out = mask.copy()
    for s, e in _mask_to_ranges(~mask):
        if s == 0 or e == len(mask):
            continue
        if (e - s) <= max_gap:
            out[s:e] = True
    return out


def segment_flight(t, feat_10col, dt=0.02):
    """
    Segment a flight trajectory into straight/turn/takeoff/landing/unknown.

    Speed thresholds are derived adaptively from the trajectory's speed
    distribution so the function works for both SITL (fast) and real (slow)
    flights.

    feat_10col columns
    ------------------
    0: altitude  1: heading  2: v_alt  3: speed_xy  4: a_alt
    5: acc_xy_norm  6: j_alt  7: jerk_xy_norm  8: curvature  9: yaw_rate
    """
    N = len(feat_10col)
    altitude = feat_10col[:, 0]
    v_alt    = feat_10col[:, 2]
    speed_xy = feat_10col[:, 3]
    a_alt    = feat_10col[:, 4]
    curvature = feat_10col[:, 8]
    yaw_rate  = feat_10col[:, 9]

    # ── adaptive speed thresholds ────────────────────────────────────────────
    pos_speeds = speed_xy[speed_xy > 0.02]
    spd_ref = float(np.percentile(pos_speeds, 70)) if len(pos_speeds) > 10 else 1.0
    min_moving_speed        = max(0.05, 0.25 * spd_ref)
    min_valid_speed_heading = max(0.10, 0.40 * spd_ref)

    # ── takeoff / landing via altitude settling ──────────────────────────────
    alt_95 = np.percentile(altitude, 95)
    target_alt = alt_95 - 0.1

    vz_small = max(0.05, 0.15 * spd_ref)
    az_small = 0.20

    flight_start = 0
    state = 0
    for i in range(N):
        if state == 0 and altitude[i] >= target_alt:       state = 1
        elif state == 1 and abs(v_alt[i]) <= vz_small:    state = 2
        elif state == 2 and abs(a_alt[i]) <= az_small:
            flight_start = i; break

    flight_end = N
    state = 0
    for i in range(N - 1, flight_start, -1):
        if state == 0 and altitude[i] >= target_alt:       state = 1
        elif state == 1 and abs(v_alt[i]) <= vz_small:    state = 2
        elif state == 2 and abs(a_alt[i]) <= az_small:
            flight_end = i; break

    # ── straight / turn in the cruise phase ──────────────────────────────────
    f = feat_10col[flight_start:flight_end]
    tf = t[flight_start:flight_end]
    M = len(f)

    if M < 10:
        return {}, {}

    spd_f  = f[:, 3]
    curv_f = f[:, 8]
    yr_f   = f[:, 9]

    spd_smooth = _safe_savgol(spd_f, window_length=21, poly_order=3)
    a_long = np.gradient(spd_smooth, dt)

    is_moving = spd_f >= min_moving_speed
    has_heading = spd_f >= min_valid_speed_heading

    # turn hysteresis
    yaw_rate_on  = 0.15
    yaw_rate_off = 0.08
    curv_on  = 0.03
    curv_off = 0.015
    min_turn_len    = max(1, int(0.4 / dt))
    min_straight_len = max(1, int(0.5 / dt))
    merge_gap_len   = max(0, int(0.25 / dt))

    raw_turn_on  = has_heading & ((np.abs(yr_f) >= yaw_rate_on)  | (curv_f >= curv_on))
    raw_turn_off = has_heading & ((np.abs(yr_f) >= yaw_rate_off) | (curv_f >= curv_off))

    is_turn = np.zeros(M, dtype=bool)
    active = False
    for i in range(M):
        if not active:
            if raw_turn_on[i]:
                active = True; is_turn[i] = True
        else:
            if raw_turn_off[i]: is_turn[i] = True
            else: active = False

    is_turn = _fill_short_false_gaps(is_turn, merge_gap_len)
    is_turn = _remove_short_true_runs(is_turn, min_turn_len)

    raw_straight = (is_moving & (~is_turn)
                    & (np.abs(yr_f) <= 0.08) & (curv_f <= 0.015))
    is_straight = _fill_short_false_gaps(raw_straight, merge_gap_len)
    is_straight = _remove_short_true_runs(is_straight, min_straight_len)
    is_straight = is_straight & (~is_turn)

    # straight sub-types
    min_acc_len = max(1, int(0.4 / dt))
    long_acc_threshold = max(0.05, 0.3 * spd_ref)   # adaptive accel threshold

    segs = {"straight": [], "straight_accel": [], "straight_decel": [],
            "straight_const": [], "turn": [], "takeoff": None, "landing": None}

    for s, e in _mask_to_ranges(is_turn):
        segs["turn"].append({
            "label": "turn",
            "t": tf[s:e], "speed_xy": spd_f[s:e],
            "yaw_rate": yr_f[s:e], "curvature": curv_f[s:e],
            "v_alt": f[s:e, 2], "a_long": a_long[s:e],
            "heading": f[s:e, 1],
        })

    for s, e in _mask_to_ranges(is_straight):
        segs["straight"].append({
            "label": "straight",
            "t": tf[s:e], "speed_xy": spd_f[s:e],
            "yaw_rate": yr_f[s:e], "curvature": curv_f[s:e],
            "v_alt": f[s:e, 2], "a_long": a_long[s:e],
            "heading": f[s:e, 1],
        })
        loc_a = a_long[s:e]
        local_accel = _fill_short_false_gaps(
            _remove_short_true_runs(loc_a >= long_acc_threshold,  min_acc_len), merge_gap_len)
        local_decel = _fill_short_false_gaps(
            _remove_short_true_runs(loc_a <= -long_acc_threshold, min_acc_len), merge_gap_len)
        local_const = ~(local_accel | local_decel)

        for ls, le in _mask_to_ranges(local_accel):
            segs["straight_accel"].append({
                "label": "straight_accel",
                "t": tf[s+ls:s+le], "speed_xy": spd_f[s+ls:s+le],
                "yaw_rate": yr_f[s+ls:s+le], "curvature": curv_f[s+ls:s+le],
                "v_alt": f[s+ls:s+le, 2], "a_long": a_long[s+ls:s+le],
                "heading": f[s+ls:s+le, 1],
            })
        for ls, le in _mask_to_ranges(local_decel):
            segs["straight_decel"].append({
                "label": "straight_decel",
                "t": tf[s+ls:s+le], "speed_xy": spd_f[s+ls:s+le],
                "yaw_rate": yr_f[s+ls:s+le], "curvature": curv_f[s+ls:s+le],
                "v_alt": f[s+ls:s+le, 2], "a_long": a_long[s+ls:s+le],
                "heading": f[s+ls:s+le, 1],
            })
        for ls, le in _mask_to_ranges(local_const):
            segs["straight_const"].append({
                "label": "straight_const",
                "t": tf[s+ls:s+le], "speed_xy": spd_f[s+ls:s+le],
                "yaw_rate": yr_f[s+ls:s+le], "curvature": curv_f[s+ls:s+le],
                "v_alt": f[s+ls:s+le, 2], "a_long": a_long[s+ls:s+le],
                "heading": f[s+ls:s+le, 1],
            })

    # takeoff / landing segments
    if flight_start > 0:
        tf_seg = t[0:flight_start]
        f_seg  = feat_10col[0:flight_start]
        spd_seg = f_seg[:, 3]
        spd_sm  = _safe_savgol(spd_seg)
        segs["takeoff"] = {
            "label": "takeoff",
            "t": tf_seg, "speed_xy": spd_seg,
            "yaw_rate": f_seg[:, 9], "curvature": f_seg[:, 8],
            "v_alt": f_seg[:, 2], "a_long": np.gradient(spd_sm, dt),
            "heading": f_seg[:, 1],
        }
    if flight_end < N - 1:
        tf_seg = t[flight_end:]
        f_seg  = feat_10col[flight_end:]
        spd_seg = f_seg[:, 3]
        spd_sm  = _safe_savgol(spd_seg)
        segs["landing"] = {
            "label": "landing",
            "t": tf_seg, "speed_xy": spd_seg,
            "yaw_rate": f_seg[:, 9], "curvature": f_seg[:, 8],
            "v_alt": f_seg[:, 2], "a_long": np.gradient(spd_sm, dt),
            "heading": f_seg[:, 1],
        }

    return segs


# ── scale-invariant ratio feature extraction ──────────────────────────────────
SEG_TYPES = ["takeoff", "landing", "straight", "straight_accel",
             "straight_decel", "straight_const", "turn"]
EPS = 1e-6

def _ratio_features(seg) -> dict | None:
    """Extract scale-invariant ratio features from a single segment dict."""
    t       = seg["t"]
    spd     = seg["speed_xy"]
    yr      = seg["yaw_rate"]
    curv    = seg["curvature"]
    v_alt   = seg["v_alt"]
    a_long  = seg["a_long"]
    label   = seg["label"]

    n = len(t)
    if n < 5:
        return None

    dt = float(np.median(np.diff(t))) if n >= 2 else 0.02

    spd_mean   = float(np.mean(spd)) + EPS
    spd_std    = float(np.std(spd))
    yr_abs     = np.abs(yr)
    curv_abs   = np.abs(curv)
    a_long_abs = np.abs(a_long)
    v_alt_abs  = np.abs(v_alt)

    path_length = float(np.trapz(spd, t)) + EPS

    def cv(arr):
        m = float(np.mean(np.abs(arr))) + EPS
        return float(np.std(arr)) / m

    def rel_peak_idx(arr):
        if len(arr) == 0: return 0.5
        return float(np.argmax(np.abs(arr))) / max(len(arr) - 1, 1)

    row = {
        # segment type
        "seg_type": label,
        **{f"is_{s}": int(label == s) for s in SEG_TYPES},

        # core ratio features (scale-invariant)
        "a_long_to_speed":     float(np.mean(a_long_abs)) / spd_mean,
        "yaw_rate_to_speed":   float(np.mean(yr_abs))     / spd_mean,
        "curvature_to_speed":  float(np.mean(curv_abs))   / spd_mean,
        "v_alt_to_speed":      float(np.mean(v_alt_abs))  / spd_mean,

        # coefficient of variation features
        "speed_cv":     spd_std / spd_mean,
        "yaw_rate_cv":  cv(yr_abs),
        "a_long_cv":    cv(a_long),
        "v_alt_cv":     cv(v_alt),

        # normalized integrals (per unit path)
        "integral_yr_per_path":    float(np.trapz(yr_abs,   t)) / path_length,
        "integral_curv_per_path":  float(np.trapz(curv_abs, t)) / path_length,
        "integral_along_per_path": float(np.trapz(a_long_abs, t)) / path_length,

        # temporal shape (where do peaks occur, 0–1)
        "peak_speed_rel":    rel_peak_idx(spd),
        "peak_yr_rel":       rel_peak_idx(yr_abs),
        "peak_along_rel":    rel_peak_idx(a_long_abs),

        # heading change normalized
        "heading_change_per_path": float(np.sum(np.abs(np.diff(seg["heading"])))) / path_length,

        # peak-to-mean ratios
        "peak_to_mean_speed":  float(np.max(spd))          / spd_mean,
        "peak_to_mean_yr":     float(np.max(yr_abs + EPS)) / (float(np.mean(yr_abs)) + EPS),
        "peak_to_mean_along":  float(np.max(a_long_abs + EPS)) / (float(np.mean(a_long_abs)) + EPS),
    }
    return row


def segments_to_rows(segs, label):
    """Flatten segment dict → list of feature rows with autopilot label."""
    rows = []
    for key, seg_obj in segs.items():
        if seg_obj is None:
            continue
        seg_list = seg_obj if isinstance(seg_obj, list) else [seg_obj]
        for seg in seg_list:
            row = _ratio_features(seg)
            if row is not None:
                row["label"] = label
                rows.append(row)
    return rows


# ── 10-col feature adaptor ────────────────────────────────────────────────────

def build_10col(traj_resampled, features_7col):
    """
    Build 10-col feature array for the segmentor from our 7-col features
    + the resampled trajectory (which contains z for altitude, vx/vy for heading).

    Output cols: altitude, heading, vh, speed_xy, ah, 0, 0, 0, curvature, yaw_rate

    Handles both NED (SITL: z positive-down, flying drone z < 0)
    and ENU (mocap: z positive-up, flying drone z > 0).
    """
    N = len(features_7col)
    z = traj_resampled[:N, 2]
    # Auto-detect convention: NED drone flies with z < 0, ENU with z > 0
    altitude = -z if np.median(z) < 0 else z
    # heading from smoothed vx, vy
    vx = traj_resampled[:N, 3]
    vy = traj_resampled[:N, 4]
    heading = np.unwrap(np.arctan2(vy, vx))

    vh        = features_7col[:, 0]
    speed_xy  = features_7col[:, 1]
    ah        = features_7col[:, 2]
    curvature = features_7col[:, 3]
    yaw_rate  = features_7col[:, 4]
    zeros     = np.zeros(N)

    return np.column_stack([
        altitude, heading, vh, speed_xy, ah,
        zeros, zeros, zeros, curvature, yaw_rate
    ])


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_out    = f"segment_lgbm_{ts}.pkl"
    feat_cols_out = f"segment_features_{ts}.json"

    print(f"\n{'='*70}")
    print(f"  Segment-based LightGBM Trainer  ({ts})")
    print(f"{'='*70}\n")

    # ── 1. load SITL data (raw features + resampled trajectory) ─────────────
    px4_files  = [(p, 0) for p in sorted(Path(PX4_FOLDER).glob("*.ulg"))]
    ardu_files = [(p, 1) for p in sorted(Path(ARDU_FOLDER).glob("*.bin"))]
    random.shuffle(px4_files); random.shuffle(ardu_files)

    px4_test_cnt  = max(1, int(len(px4_files)  * TEST_RATIO)) if len(px4_files)  > 1 else 0
    ardu_test_cnt = max(1, int(len(ardu_files) * TEST_RATIO)) if len(ardu_files) > 1 else 0

    train_files = px4_files[px4_test_cnt:]  + ardu_files[ardu_test_cnt:]
    test_files  = px4_files[:px4_test_cnt]  + ardu_files[:ardu_test_cnt]

    print(f"Files — train: {len(train_files)}  test: {len(test_files)}")

    from trajectory_processor import (
        process_px4_flight_data,
        process_ardu_flight_data,
    )

    def load_segments(file_list):
        all_rows = []
        for path, lbl in file_list:
            path = str(path)
            if lbl == 0:
                result = process_px4_flight_data(path)
            else:
                result = process_ardu_flight_data(path)
            if result is None or result[4] is None:
                continue
            _, _, t_res, traj_res, feat7, _, _ = result
            if len(feat7) < 50:
                continue
            feat10 = build_10col(traj_res, feat7)
            dt = 1.0 / 50.0
            segs = segment_flight(t_res, feat10, dt=dt)
            rows = segments_to_rows(segs, lbl)
            all_rows.extend(rows)
        return all_rows

    print("⏳ Loading + segmenting train data...")
    train_rows = load_segments(train_files)
    print("⏳ Loading + segmenting test data...")
    test_rows  = load_segments(test_files)

    df_train = pd.DataFrame(train_rows).drop(columns=["seg_type"])
    df_test  = pd.DataFrame(test_rows).drop(columns=["seg_type"])

    label_col = "label"
    feature_cols = [c for c in df_train.columns if c != label_col]

    # drop columns with all-NaN
    df_train = df_train.dropna(axis=1, how="all")
    feature_cols = [c for c in feature_cols if c in df_train.columns]
    df_train[feature_cols] = df_train[feature_cols].fillna(0.0)
    df_test[feature_cols]  = df_test[feature_cols].reindex(columns=feature_cols).fillna(0.0)

    X_train = df_train[feature_cols].values
    y_train = df_train[label_col].values
    X_test  = df_test[feature_cols].values
    y_test  = df_test[label_col].values

    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nTrain class distribution:")
    for lbl, cnt in zip(unique, counts):
        name = "PX4" if lbl == 0 else "ArduPilot"
        print(f"  {name}: {cnt} segments ({100*cnt/len(y_train):.1f}%)")
    print(f"  Total: {len(y_train)} segments from {len(train_files)} flights\n")

    # ── 2. LightGBM ──────────────────────────────────────────────────────────
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=10,
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
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred) * 100
    print(f"\nTest accuracy: {test_acc:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["PX4", "ArduPilot"]))

    # ── 3. save ───────────────────────────────────────────────────────────────
    with open(model_out, "wb") as f:
        pickle.dump(clf, f)
    with open(feat_cols_out, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Model saved: {model_out}")
    print(f"Feature columns saved: {feat_cols_out}")

    importances = sorted(
        zip(feature_cols, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop-15 most important features:")
    for name, imp in importances[:15]:
        print(f"  {imp:6.0f}  {name}")


if __name__ == "__main__":
    main()
