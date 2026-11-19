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
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pyulog import ULog
from pymavlink import mavutil
import os

from plot_trajectory import plot_combined_xy_trajectory, plot_full_trajectory_with_spans, plot_turn_segment_features


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


# =====================================================================
# FEATURE EXTRACTION
# =====================================================================
def feature_names():
    return [
        "Altitude (m)",
        "Heading (rad)",
        "Upward Velocity (m/s)",
        "In-Plane Speed (m/s)",
        "Upward Acceleration (m/s²)",
        "In-Plane Accel Norm (m/s²)",
        "Upward Jerk (m/s³)",
        "In-Plane Jerk Norm (m/s³)",
        "Curvature (1/m)",
        "Yaw Rate (rad/s)",
    ]

def resample_and_extract_features(t, 
                   x, y, z, vx, vy, vz, *, 
                   target_hz=50, smooth_window=21, poly_order=5):
    """
    Extract uniformly resampled trajectory data.
    Compute kinematic features such as altitude, heading, speed, 
    acceleration, jerk, curvature, yaw rate, and slip rate.
    TODO: write down docstring

    Feature columns
    ---------------
    0 : altitude
    1 : heading
    2 : vertical velocity
    3 : horizontal speed
    4 : vertical acceleration
    5 : horizontal acceleration norm
    6 : vertical jerk
    7 : horizontal jerk norm
    8 : curvature
    9 : yaw rate

    Notes
    -----
    - Input timestamps are de-duplicated before interpolation.
    - Heading and yaw-rate are derived from smoothed horizontal velocity.
    - Curvature is gated by horizontal speed to avoid blow-up at low speed.
    """
    # Position and velocity history
    t, unique_indices = np.unique(t, return_index=True)
    x, y, z = x[unique_indices], y[unique_indices], z[unique_indices]
    vx, vy, vz = vx[unique_indices], vy[unique_indices], vz[unique_indices]
    
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
    vx_resampled = interp1d(t, vx, bounds_error=True, fill_value=(vx[0], vx[-1]))(t_resampled)
    vy_resampled = interp1d(t, vy, bounds_error=True, fill_value=(vy[0], vy[-1]))(t_resampled)
    vz_resampled = interp1d(t, vz, bounds_error=True, fill_value=(vz[0], vz[-1]))(t_resampled)
    
    x_smooth = _safe_savgol(x_resampled, window_length=smooth_window, poly_order=poly_order)
    y_smooth = _safe_savgol(y_resampled, window_length=smooth_window, poly_order=poly_order)
    z_smooth = _safe_savgol(z_resampled, window_length=smooth_window, poly_order=poly_order)
    vx_smooth = _safe_savgol(vx_resampled, window_length=smooth_window, poly_order=poly_order)
    vy_smooth = _safe_savgol(vy_resampled, window_length=smooth_window, poly_order=poly_order)
    vz_smooth = _safe_savgol(vz_resampled, window_length=smooth_window, poly_order=poly_order)

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

    v_vec_3d = np.vstack((vx_smooth, vy_smooth, vz_smooth)).T
    a_vec_3d = np.vstack((ax, ay, az)).T
    cross_va = np.cross(v_vec_3d, a_vec_3d)
    cross_mag = np.linalg.norm(cross_va, axis=1)

    speed_3d = np.linalg.norm(v_vec_3d, axis=1)
    curvature = cross_mag / np.maximum(speed_3d**3, 1e-3)

    # Additional gating for low-speed robustness
    curvature[speed_xy < 0.5] = 0.0
    curvature_smooth = _safe_savgol(curvature, window_length=smooth_window, poly_order=poly_order)

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
            h_smooth,
            heading,
            vh_smooth,
            speed_xy,
            ah,
            acc_norm_xy,
            j_alt,
            jerk_norm_xy,
            curvature_smooth,
            yaw_rate_smooth,
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
def process_px4_flight_data(ulog_path, *, target_hz=50, t_attention=1.0, size_threshold = 10):
    try:
        # Read .ulg file
        ulog = ULog(ulog_path)
        loc_data = ulog.get_dataset('vehicle_local_position').data
        
        # Get time, position and velocity
        t_raw = loc_data['timestamp'] / 1e6
        x_raw, y_raw, z_raw = loc_data['x'], loc_data['y'], loc_data['z']
        vx_raw, vy_raw, vz_raw = loc_data['vx'], loc_data['vy'], loc_data['vz']
            
        # Check length of trajectory
        if t_raw.size <= size_threshold: return None, None, None, None, None, None, None

        # Compute features
        t_resampled, traj_resampled, feat_extracted, feature_names = \
        resample_and_extract_features(t_raw, x_raw, y_raw, z_raw, vx_raw, vy_raw, vz_raw, target_hz=target_hz, smooth_window=target_hz*t_attention+1, poly_order=5)
        
        # Extract segments and spans
        # segments, spans = extract_flight_segments(t_resampled, feat_extracted)
        segments = []
        spans = []
        
        # Return raw and processed data
        traj_raw = np.vstack(
            (
                x_raw,
                y_raw,
                z_raw,
                vx_raw,
                vy_raw,
                vz_raw,
            )
        ).T
        return t_raw, traj_raw, t_resampled, traj_resampled, feat_extracted, segments, spans
        
    except Exception as e:
        print(f"[PX4 Extract Error] {ulog_path}: {e}")
        return None, None, None, None, None, None, None

def process_ardu_flight_data(bin_path, *, target_hz=50, t_attention=1.0, size_threshold = 10):
    try:
        # Read .BIN file
        mlog = mavutil.mavlink_connection(bin_path)
        t_raw, x_raw, y_raw, z_raw, vx_raw, vy_raw, vz_raw = [], [], [], [], [], [], []
        
        while True:
            msg = mlog.recv_match(type='SIM2', blocking=False)
            if not msg: break
            t_raw.append(msg.TimeUS / 1e6)
            x_raw.append(msg.PN); y_raw.append(msg.PE); z_raw.append(msg.PD)
            vx_raw.append(msg.VN); vy_raw.append(msg.VE); vz_raw.append(msg.VD)

        t_raw, x_raw, y_raw, z_raw, vx_raw, vy_raw, vz_raw = np.array(t_raw), np.array(x_raw), np.array(y_raw), np.array(z_raw), np.array(vx_raw), np.array(vy_raw), np.array(vz_raw)
            
        # Check length of trajectory
        if t_raw.size <= size_threshold: return None, None, None, None, None, None, None

        # Compute features
        t_resampled, traj_resampled, feat_extracted = \
        resample_and_extract_features(t_raw, x_raw, y_raw, z_raw, vx_raw, vy_raw, vz_raw, target_hz=target_hz, smooth_window=target_hz*t_attention+1, poly_order=5)

        # Extract segments and spans
        # segments, spans = extract_flight_segments(t_resampled, feat_extracted)
        segments = []
        spans = []

        # Return raw and processed data
        traj_raw = np.vstack(
            (
                x_raw,
                y_raw,
                z_raw,
                vx_raw,
                vy_raw,
                vz_raw,
            )
        ).T
        return t_raw, traj_raw, t_resampled, traj_resampled, feat_extracted, segments, spans
        
    except Exception as e:
        print(f"[ArduPilot Extract Error] {bin_path}: {e}")
        return None, None, None, None, None, None, None

# def process_rosbag_flight_data(csv_path):
#     try:
#         data = np.genfromtxt(csv_path, delimiter=',', names=True)
#         t_loc = data['bag_time_ns'] / 1e9
#         x, y, z = data['x'], data['y'], data['z']
#         vx, vy, vz = data['vx'], data['vy'], data['vz']
        
#         extracted = extract_kinematic_features(t_loc, x, y, z, vx, vy, vz)
#         if extracted is None: return None, None, None, None, None, None
        
#         t_full, feat_full = extracted
#         segments, spans = extract_flight_segments(t_full, feat_full)
        
#         return x, y, t_full, feat_full, segments, spans
        
#     except Exception as e:
#         print(f"[ROS Bag Extract Error] {csv_path}: {e}")
#         return None, None, None, None, None, None


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