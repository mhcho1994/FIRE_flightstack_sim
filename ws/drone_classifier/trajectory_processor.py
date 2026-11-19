#!/usr/bin/env python3
"""
Trajectory Pre-processing Pipeline for PX4 and ArduPilot SITL Logs

What this script does:
---------------------
- TODO: write down the main steps of the pipeline in a concise manner
- TODO: estimate yaw angle and compute slip rate (heading - yaw_rate) to capture side-slip during turns
"""
from pathlib import Path
import numpy as np
import json
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pyulog import ULog
from pymavlink import mavutil
import os
import pandas as pd

# from plot_trajectory import plot_combined_xy_trajectory, plot_full_trajectory_with_spans, plot_turn_segment_features


# =====================================================================
# HELPERS
# =====================================================================
def _safe_savgol(signal, window_length=21, poly_order=5):
    """
    Apply Savitzky-Golay smoothing safely.

    If the signal is too short, return the original signal.
    """
    n = len(signal)
    window_length = int(window_length)
    poly_order = int(poly_order)

    # Adjust window length
    wl = min(window_length, n) if min(window_length, n) % 2 == 1 else min(window_length, n) - 1

    # If the adjusted window length is still too short, return the original signal
    if wl <= poly_order:
        return signal

    return savgol_filter(signal, window_length=wl, polyorder=poly_order)


ROBUST_SCALE_FEATURE_INDICES = tuple(range(7))


def fit_robust_feature_scaler(feature_arrays, feature_indices=ROBUST_SCALE_FEATURE_INDICES):
    """
    Fit robust scaling stats on SITL training features only.

    Real camera odometry must use transform_robust_features with these saved
    stats; do not fit on real/test data.
    """
    valid_arrays = [features for features in feature_arrays if features is not None and len(features) > 0]
    if len(valid_arrays) == 0:
        raise ValueError("Cannot fit robust scaler without feature data.")

    all_features = np.vstack(valid_arrays)
    indices = np.array(feature_indices, dtype=int)
    selected = all_features[:, indices]

    center = np.nanmedian(selected, axis=0)
    q1 = np.nanpercentile(selected, 25, axis=0)
    q3 = np.nanpercentile(selected, 75, axis=0)
    scale = q3 - q1
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0

    return {
        "feature_indices": indices.tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
    }


def transform_robust_features(features, stats, use_signed_log=True):
    """Apply pre-fitted robust scaling stats without fitting."""
    if features is None:
        return None

    transformed = features.copy()
    indices = np.array(stats["feature_indices"], dtype=int)
    center = np.array(stats["center"], dtype=float)
    scale = np.array(stats["scale"], dtype=float)
    transformed[:, indices] = (transformed[:, indices] - center) / scale

    if use_signed_log:
        transformed[:, indices] = np.sign(transformed[:, indices]) * np.log1p(np.abs(transformed[:, indices]))

    return transformed


def save_robust_feature_scaler(stats, path):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_robust_feature_scaler(path):
    with open(path, "r") as f:
        return json.load(f)


# =====================================================================
# FEATURE EXTRACTION
# =====================================================================
def feature_names():
    return [
        "Upward Velocity (m/s)",
        "In-Plane Speed (m/s)",
        "Upward Acceleration (m/s²)",
        "Curvature (1/m)",
        "Yaw Rate (rad/s)",
        "Yaw Angular Accel (rad/s²)",
        "Speed x Curvature (1/s)",
    ]

def resample_and_extract_features(t,
                   x, y, z, *,
                   target_hz=50, smooth_window=21, poly_order=5,
                   pos_noise_std=0.0):
    """
    Extract uniformly resampled trajectory data.
    Compute kinematic features such as altitude, heading, speed, 
    acceleration, jerk, curvature, yaw rate, and slip rate.
    TODO: write down docstring

    Feature columns
    ---------------
    0 : vertical velocity
    1 : horizontal speed
    2 : vertical acceleration
    3 : curvature
    4 : yaw rate
    5 : yaw angular acceleration  (d(yaw_rate)/dt — PX4 jerk-limit signature)
    6 : speed × curvature         (turn coordination proxy)

    Notes
    -----
    - Input timestamps are de-duplicated before interpolation.
    - Heading and yaw-rate are derived from smoothed horizontal velocity.
    - Curvature is gated by horizontal speed to avoid blow-up at low speed.
    """
    # Position and velocity history
    t, unique_indices = np.unique(t, return_index=True)
    x, y, z = x[unique_indices], y[unique_indices], z[unique_indices]
    # vx, vy, vz = vx[unique_indices], vy[unique_indices], vz[unique_indices]
    
    # Reject if trajectory is short
    if len(t) < 2: return None

    # Time re-sampling
    dt = 1.0 / target_hz
    t_start, t_end = t[0], t[-1]
    t_resampled = np.arange(t_start, t_end, dt) 

    if len(t_resampled) < 2: return None
    
    # Trajectory re-sampling
    x_resampled = interp1d(t, x, bounds_error=True, fill_value=(x[0], x[-1]))(t_resampled)
    y_resampled = interp1d(t, y, bounds_error=True, fill_value=(y[0], y[-1]))(t_resampled)
    z_resampled = interp1d(t, z, bounds_error=True, fill_value=(z[0], z[-1]))(t_resampled)

    # Inject Gaussian position noise before smoothing to simulate camera/mocap odometry.
    # The SG filter suppresses high-freq components but preserves the elevated
    # derivative variance that characterises real sensor data.
    if pos_noise_std > 0.0:
        x_resampled += np.random.normal(0.0, pos_noise_std, x_resampled.shape)
        y_resampled += np.random.normal(0.0, pos_noise_std, y_resampled.shape)
        z_resampled += np.random.normal(0.0, pos_noise_std, z_resampled.shape)

    x_smooth = _safe_savgol(x_resampled, window_length=smooth_window, poly_order=poly_order)
    y_smooth = _safe_savgol(y_resampled, window_length=smooth_window, poly_order=poly_order)
    z_smooth = _safe_savgol(z_resampled, window_length=smooth_window, poly_order=poly_order)
    # vx_smooth = _safe_savgol(vx_resampled, window_length=smooth_window, poly_order=poly_order)
    # vy_smooth = _safe_savgol(vy_resampled, window_length=smooth_window, poly_order=poly_order)
    # vz_smooth = _safe_savgol(vz_resampled, window_length=smooth_window, poly_order=poly_order)
    
    # Compute velcocity
    # TODO: Check whether the SG filter is appropriate for acceleration, jerk calculation
    vx = np.gradient(x_smooth, dt)
    vy = np.gradient(y_smooth, dt)
    vz = np.gradient(z_smooth, dt)

    vx_smooth = _safe_savgol(vx, window_length=smooth_window, poly_order=poly_order)
    vy_smooth = _safe_savgol(vy, window_length=smooth_window, poly_order=poly_order)
    vz_smooth = _safe_savgol(vz, window_length=smooth_window, poly_order=poly_order)

    # Change the sign of z and vz to represent altitude and upward velocity
    h_smooth = -z_smooth
    vh_smooth = -vz_smooth

    # Compute horizontal speed
    v_xy = np.vstack((vx_smooth, vy_smooth)).T
    speed_xy = np.linalg.norm(v_xy, axis=1)


    # Compute acceleration
    # TODO: Check whether the SG filter is appropriate for acceleration, jerk calculation
    ax = np.gradient(vx_smooth, dt)
    ay = np.gradient(vy_smooth, dt)
    az = np.gradient(vz_smooth, dt)

    ah = -az
    a_xy = np.vstack((ax, ay)).T
    acc_norm_xy = np.linalg.norm(a_xy, axis=1)

    ax_smooth = _safe_savgol(ax, window_length=smooth_window, poly_order=poly_order)
    ay_smooth = _safe_savgol(ay, window_length=smooth_window, poly_order=poly_order)
    az_smooth = _safe_savgol(az, window_length=smooth_window, poly_order=poly_order)

    jx = np.gradient(ax_smooth, dt)
    jy = np.gradient(ay_smooth, dt)
    jz = np.gradient(az_smooth, dt)

    j_alt = -jz
    j_xy = np.vstack((jx, jy)).T
    jerk_norm_xy = np.linalg.norm(j_xy, axis=1)

    heading = np.unwrap(np.arctan2(vy_smooth, vx_smooth))
    yaw_rate = np.gradient(heading, dt)
    yaw_rate_smooth = _safe_savgol(yaw_rate, window_length=smooth_window, poly_order=poly_order)

    yaw_angular_accel = np.gradient(yaw_rate_smooth, dt)
    yaw_angular_accel_smooth = _safe_savgol(yaw_angular_accel, window_length=smooth_window, poly_order=poly_order)

    v_vec_3d = np.vstack((vx_smooth, vy_smooth, vz_smooth)).T
    a_vec_3d = np.vstack((ax, ay, az)).T
    cross_va = np.cross(v_vec_3d, a_vec_3d)
    cross_mag = np.linalg.norm(cross_va, axis=1)

    speed_3d = np.linalg.norm(v_vec_3d, axis=1)
    curvature = cross_mag / np.maximum(speed_3d**3, 1e-3)

    # Additional gating for low-speed robustness
    curvature[speed_xy < 0.5] = 0.0
    curvature_smooth = _safe_savgol(curvature, window_length=smooth_window, poly_order=poly_order)

    speed_curvature = speed_xy * curvature_smooth

    resampled_traj = np.vstack(
        (
            x_smooth,
            y_smooth,
            z_smooth,
            vx_smooth,
            vy_smooth,
            vz_smooth,
        )
    ).T

    features = np.vstack(
        (
            vh_smooth,
            speed_xy,
            ah,
            curvature_smooth,
            yaw_rate_smooth,
            yaw_angular_accel_smooth,
            speed_curvature,
        )
    ).T


    # #### EXCLUDE non-flight data ####
    # FLIGHT_THRESHOLD = 4.0  
    # altitude = np.abs(z_new)  
    # flight_mask = altitude > FLIGHT_THRESHOLD
    # filtered_features = features[flight_mask]
    
    # if len(filtered_features) == 0:
    #     FLIGHT_THRESHOLD = 0.1
    #     flight_mask = altitude > FLIGHT_THRESHOLD
    #     filtered_features = features[flight_mask]
    
    # if len(filtered_features) == 0:
    #     return None
    
    # # 💡 [반영 2] Robust Normalization (Outlier 무시 정규화)
    # # 극단적인 Min/Max 대신 하위 1%와 상위 99% 값을 기준으로 스케일링합니다.
    # if use_global_normalize:
    #     stats = get_global_normalization_stats()
    #     p1 = stats['p1']
    #     p99 = stats['p99']
    # else:
    #     # Per-file 정규화 시에도 Robust Scaler 로직 적용
    #     p1 = np.percentile(filtered_features, 1, axis=0, keepdims=True)
    #     p99 = np.percentile(filtered_features, 99, axis=0, keepdims=True)
        
    # feature_range = p99 - p1
    # feature_range[feature_range == 0] = 1  # 0으로 나누기 방지
    
    # # -1 ~ 1 사이로 변환 후, 1% / 99%를 벗어난 극단값은 -1 또는 1로 잘라냄(Clip)
    # normalized_features = 2 * (filtered_features - p1) / feature_range - 1
    # normalized_features = np.clip(normalized_features, -1.0, 1.0)


    return t_resampled, resampled_traj, features

# =====================================================================
# PROCESSING TRAJECTORY DATA
# =====================================================================
def process_px4_flight_data(ulog_path, *, target_hz=50, t_attention=1.0, size_threshold=10, pos_noise_std=0.0):
    try:
        # Read .ulg file
        ulog = ULog(ulog_path)
        loc_data = ulog.get_dataset('vehicle_local_position').data
        
        # Get time, position and velocity
        t_raw = loc_data['timestamp'] / 1e6
        x_raw, y_raw, z_raw = loc_data['x'], loc_data['y'], loc_data['z']

        #will change to computing velocity from position data to ensure consistency
        vx_raw, vy_raw, vz_raw = loc_data['vx'], loc_data['vy'], loc_data['vz']
            
        # Check length of trajectory
        if t_raw.size <= size_threshold: return None, None, None, None, None, None, None

        # Compute features
        t_resampled, traj_resampled, feat_extracted = \
        resample_and_extract_features(t_raw, x_raw, y_raw, z_raw, target_hz=target_hz, smooth_window=target_hz*t_attention+1, poly_order=5, pos_noise_std=pos_noise_std)

        segments = []
        spans = []

        traj_raw = np.vstack((x_raw, y_raw, z_raw, vx_raw, vy_raw, vz_raw)).T
        return t_raw, traj_raw, t_resampled, traj_resampled, feat_extracted, segments, spans

    except Exception as e:
        print(f"[PX4 Extract Error] {ulog_path}: {e}")
        return None, None, None, None, None, None, None

def process_ardu_flight_data(bin_path, *, target_hz=50, t_attention=1.0, size_threshold=10, pos_noise_std=0.0):
    try:
        # Read .BIN file — use EKF estimated position (XKF1/NKF1) instead of
        # simulator ground truth (SIM2) to match real-flight odometry noise characteristics.
        mlog = mavutil.mavlink_connection(bin_path)
        t_raw, x_raw, y_raw, z_raw, vx_raw, vy_raw, vz_raw = [], [], [], [], [], [], []

        while True:
            msg = mlog.recv_match(type=['XKF1', 'NKF1'], blocking=False)
            if not msg: break
            t_raw.append(msg.TimeUS / 1e6)
            x_raw.append(msg.PN); y_raw.append(msg.PE); z_raw.append(msg.PD)
            vx_raw.append(msg.VN); vy_raw.append(msg.VE); vz_raw.append(msg.VD)

        t_raw, x_raw, y_raw, z_raw, vx_raw, vy_raw, vz_raw = np.array(t_raw), np.array(x_raw), np.array(y_raw), np.array(z_raw), np.array(vx_raw), np.array(vy_raw), np.array(vz_raw)
            
        # Check length of trajectory
        if t_raw.size <= size_threshold: return None, None, None, None, None, None, None

        # Compute features
        t_resampled, traj_resampled, feat_extracted = \
        resample_and_extract_features(t_raw, x_raw, y_raw, z_raw, target_hz=target_hz, smooth_window=target_hz*t_attention+1, poly_order=5, pos_noise_std=pos_noise_std)

        segments = []
        spans = []

        traj_raw = np.vstack((x_raw, y_raw, z_raw, vx_raw, vy_raw, vz_raw)).T
        return t_raw, traj_raw, t_resampled, traj_resampled, feat_extracted, segments, spans

    except Exception as e:
        print(f"[ArduPilot Extract Error] {bin_path}: {e}")
        return None, None, None, None, None, None, None

def _contiguous_true_spans(mask, min_len):
    spans = []
    start = None

    for idx, is_valid in enumerate(mask):
        if is_valid and start is None:
            start = idx
        elif not is_valid and start is not None:
            if idx - start > min_len:
                spans.append((start, idx))
            start = None

    if start is not None and len(mask) - start > min_len:
        spans.append((start, len(mask)))

    return spans


def process_rosbag_flight_data(csv_path, target_hz=50, t_attention=1.0, size_threshold=10):
    try:
        import pandas as pd

        print(f"[ROS CSV] Reading: {csv_path}")
        df = pd.read_csv(csv_path, skip_blank_lines=False)
        cols = df.columns.tolist()
        cols_lower = {c.lower(): c for c in cols}

        print(f"[DEBUG] Available columns: {cols}")

        # Strict position extraction: accept ('x','y','z') OR ('gtx','gty','gtz')
        if all(k in cols_lower for k in ('x', 'y', 'z')):
            x_col, y_col, z_col = cols_lower['x'], cols_lower['y'], cols_lower['z']
            print(f"[DEBUG] Using columns: {x_col}, {y_col}, {z_col}")
        # elif all(k in cols_lower for k in ('xsmooth', 'ysmooth', 'zsmooth')):
        #     x_col, y_col, z_col = cols_lower['xsmooth'], cols_lower['ysmooth'], cols_lower['zsmooth']
            print(f"[DEBUG] Using columns: {x_col}, {y_col}, {z_col}")
        elif all(k in cols_lower for k in ('x_smooth', 'y_smooth', 'z_smooth')):
            x_col, y_col, z_col = cols_lower['x_smooth'], cols_lower['y_smooth'], cols_lower['z_smooth']
            print(f"[DEBUG] Using columns: {x_col}, {y_col}, {z_col}")
        elif all(k in cols_lower for k in ('gtx', 'gty', 'gtz')):
            x_col, y_col, z_col = cols_lower['gtx'], cols_lower['gty'], cols_lower['gtz']
            print(f"[DEBUG] Using columns: {x_col}, {y_col}, {z_col}")
        else:
            print(f"[ROS Bag Extract Error] {csv_path}: required position columns not found (need x,y,z or gtx,gty,gtz). Available: {cols}")
            print(f"[DEBUG] cols_lower keys: {list(cols_lower.keys())}")
            return []

        # Timestamp detection
        time_candidates = ['timestamp', 'time', 't', 'secs', 'sec', 'bag_time_ns', 'stamp', 'ros_time']
        time_col = None
        for name in time_candidates:
            if name in cols_lower:
                time_col = cols_lower[name]
                break
        if time_col is None:
            for k in cols_lower:
                if 'time' in k or 'stamp' in k:
                    time_col = cols_lower[k]
                    break
        if time_col is None:
            print(f"[ROS Bag Extract Error] {csv_path}: no time column found")
            return []

        t_loc = pd.to_numeric(df[time_col], errors='coerce').values
        if 'ns' in time_col.lower() or (np.nanmax(np.abs(t_loc)) > 1e12):
            t_loc = t_loc / 1e9

        x = pd.to_numeric(df[x_col], errors='coerce').values
        y = pd.to_numeric(df[y_col], errors='coerce').values
        z = pd.to_numeric(df[z_col], errors='coerce').values

        valid_mask = np.isfinite(t_loc) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        valid_spans = _contiguous_true_spans(valid_mask, size_threshold)

        if len(valid_spans) == 0:
            print(f"[ROS Bag Extract Error] {csv_path}: no valid continuous segment found")
            return []

        missing_rows = len(valid_mask) - int(valid_mask.sum())
        if missing_rows > 0:
            print(f"[INFO] Found {missing_rows} missing rows; split into {len(valid_spans)} segment(s)")

        processed_segments = []

        for segment_idx, (start, end) in enumerate(valid_spans):
            t_seg = t_loc[start:end]
            x_seg = x[start:end]
            y_seg = y[start:end]
            z_seg = z[start:end]

            print(f"[OK] Segment {segment_idx}: rows {start}-{end - 1}, {len(t_seg)} samples")
            print(f"     x: {x_seg.min():.3f}~{x_seg.max():.3f}, y: {y_seg.min():.3f}~{y_seg.max():.3f}, z: {z_seg.min():.3f}~{z_seg.max():.3f}")

            result = resample_and_extract_features(
                t_seg, x_seg, y_seg, z_seg,
                target_hz=target_hz,
                smooth_window=int(target_hz * t_attention) + 1,
                poly_order=5
            )
            if result is None:
                print(f"[ROS Bag Extract Error] {csv_path}: segment {segment_idx} feature extraction failed")
                continue

            t_resampled, traj_resampled, feat_extracted = result
            traj_raw = np.vstack((x_seg, y_seg, z_seg)).T
            spans = [(start, end - 1)]
            data = (t_seg, traj_raw, t_resampled, traj_resampled, feat_extracted, [], spans)

            processed_segments.append({
                'segment_index': segment_idx,
                'row_start': start,
                'row_end': end - 1,
                't_start': float(t_seg[0]),
                't_end': float(t_seg[-1]),
                'data': data,
            })

        return processed_segments

    except Exception as e:
        print(f"[ROS Bag Extract Error] {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


process_rosbag_flight_segments = process_rosbag_flight_data



def collect_all_runs_segment_dataframe(base_data_dir=Path("./data/sitl_logs"), target_hz=50.0, t_attention=1.0):
    """
    Collect segment-level features from all PX4 and ArduPilot runs.

    Parameters
    ----------
    base_data_dir : Path
        Root directory containing run_000, run_001, ...
    target_hz : float
        Target sampling frequency for resampling.
    t_attention : float
        Smoothing time window for attention-based feature extraction.

    Returns
    -------
    pd.DataFrame
        Combined segment-level dataset for both autopilots.
    """
    run_dirs = list(base_data_dir.glob("run_*"))
    num_runs = len([p for p in run_dirs if p.is_dir()])
    print(f"[Info] Starting Trajectory Pre-processing Pipeline: {num_runs} Trajectory Samples")

    for i in range(num_runs):
        run_folder = f"run_{i:03d}"
        run_dir = BASE_DATA_DIR / run_folder
        if not run_dir.exists(): continue

        px4_dir = run_dir / "px4_logs" / "raw"
        ardu_dir = run_dir / "ardu_logs" / "raw" / "logs"
        
        t_raw_px4, traj_raw_px4, t_resampled_px4, traj_resampled_px4, feat_px4, segments_px4, spans_px4 = (None,) * 7
        t_raw_ardu, traj_raw_ardu, t_resampled_ardu, traj_resampled_ardu, feat_ardu, segments_ardu, spans_ardu = (None,) * 7

        if px4_dir.exists():
            for file in os.listdir(px4_dir):
                if file.lower().endswith('.ulg'):
                    print(f"[Info] Processing PX4 Trajectory: No.{i+1} Trajectory Sample")
                    t_raw_px4, traj_raw_px4, t_resampled_px4, traj_resampled_px4, feat_px4, segments_px4, spans_px4 \
                        = process_px4_flight_data(str(px4_dir / file), target_hz=target_hz, t_attention=t_attention)
                    break 

        if ardu_dir.exists():
            for file in os.listdir(ardu_dir):
                if file.lower().endswith('.bin'):
                    print(f"[Info] Processing ArduPilot Trajectory: No.{i+1} Trajectory Sample")
                    t_raw_ardu, traj_raw_ardu, t_resampled_ardu, traj_resampled_ardu, feat_ardu, segments_ardu, spans_ardu \
                        = process_ardu_flight_data(str(ardu_dir / file), target_hz=target_hz, t_attention=t_attention)
                    break

        size_threshold = 10
        if (traj_raw_px4.size > size_threshold) or (traj_raw_ardu.size > size_threshold):
            plot_combined_xy_trajectory(
                traj_raw_px4, traj_raw_ardu, traj_resampled_px4, traj_resampled_ardu, 
                title=f"Combined X-Y Trajectory ({run_folder})", 
                save_figure=True,
                save_path=str(f"trajectory_xy_combined_{run_folder}.png")
            )

    #     if feat_px4 is not None and len(feat_px4) > 0:
    #         print(f"[{run_folder}] Generating PX4 Trajectory plot (Visualizing Segments)...")
    #         plot_full_trajectory_with_spans(
    #             t=t_px4, features=feat_px4, spans=spans_px4, 
    #             title=f"Trajectory [Segment Check]: PX4 ({run_folder})", 
    #             save_path=str(run_dir / f"features_px4_seg_check_{run_folder}.png"),
    #             line_color='tab:green' 
    #         )
            
    #         if turn_px4 and len(turn_px4) > 0:
    #             t_turn, feat_turn = turn_px4[0]
    #             plot_turn_segment_features(
    #                 t=t_turn, features=feat_turn, 
    #                 title=f"Trajectory [Isolated Turn]: PX4 ({run_folder})", 
    #                 save_path=str(run_dir / f"features_px4_turn_seg_{run_folder}.png"),
    #                 line_color='tab:green' 
    #             )

    
    #     if feat_ardu is not None and len(feat_ardu) > 0:
    #         print(f"[{run_folder}] Generating ArduPilot Trajectory plot (Visualizing Segments)...")
    #         plot_full_trajectory_with_spans(
    #             t=t_ardu, features=feat_ardu, spans=spans_ardu, 
    #             title=f"Trajectory [Segment Check]: ArduPilot ({run_folder})", 
    #             save_path=str(run_dir / f"features_ardupilot_seg_check_{run_folder}.png"),
    #             line_color='tab:orange' 
    #         )
            
    #         if turn_ardu and len(turn_ardu) > 0:
    #             t_turn, feat_turn = turn_ardu[0]
    #             plot_turn_segment_features(
    #                 t=t_turn, features=feat_turn, 
    #                 title=f"Trajectory [Isolated Turn]: ArduPilot ({run_folder})", 
    #                 save_path=str(run_dir / f"features_ardupilot_turn_seg_{run_folder}.png"),
    #                 line_color='tab:orange' 
    #             )

    print("\n[Info] All combined XY and segmentation plots generated successfully!")



if __name__ == "__main__":
    """
    Process trajectories data of PX4, Ardupilot
    """
    BASE_DATA_DIR = Path("./data/260408_sitl_logs_250") 
    TARGET_HZ = 50.0
    T_ATTENTION = 0.5

    collect_all_runs_segment_dataframe(base_data_dir = BASE_DATA_DIR, target_hz=TARGET_HZ, t_attention=T_ATTENTION)
