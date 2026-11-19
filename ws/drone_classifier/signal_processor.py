from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pyulog import ULog
from pymavlink import mavutil
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.use('Agg')  

TARGET_HZ = 50.0  
DT = 1.0 / TARGET_HZ

# 💡 [반영 2] Min-Max 대신 1%~99% Percentile을 사용하는 Robust 글로벌 통계로 변경
GLOBAL_FEATURE_STATS = {
    'p1': np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -2.0]),  # 하위 1% (노이즈 제외 최솟값)
    'p99': np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 2.0]), # 상위 99% (스파이크 제외 최댓값)
}

def set_global_normalization_stats(p1_stats, p99_stats):
    global GLOBAL_FEATURE_STATS
    GLOBAL_FEATURE_STATS = {
        'p1': p1_stats,
        'p99': p99_stats,
    }

def get_global_normalization_stats():
    return GLOBAL_FEATURE_STATS

def resample_and_extract_features(t, x, y, z, vx, vy, vz, t_att, yaw_att, source_type='sensor', use_global_normalize=False):
    """
    source_type: 'sensor' (Bin/ULog) 또는 'csv' (Visual/Post-processed Odom)
    """
    t_start = max(t[0], t_att[0])
    t_end = min(t[-1], t_att[-1])
    
    t_new = np.arange(t_start, t_end, DT)

    # 선형 보간 (50Hz 리샘플링)
    x_new = interp1d(t, x)(t_new)
    y_new = interp1d(t, y)(t_new)
    z_new = interp1d(t, z)(t_new)
    vx_new = interp1d(t, vx)(t_new)
    vy_new = interp1d(t, vy)(t_new)
    vz_new = interp1d(t, vz)(t_new)
    
    yaw_unwrapped = np.unwrap(yaw_att)
    yaw_new = interp1d(t_att, yaw_unwrapped)(t_new)
    
    v_vec = np.vstack((vx_new, vy_new, vz_new)).T
    speed = np.linalg.norm(v_vec, axis=1)
    
    # 💡 [반영 1] 데이터 소스별 스무딩 윈도우 차별화 (CSV는 계단 현상이 있으므로 강력한 스무딩)
    if source_type == 'csv':
        base_window = 41  # 약 0.8초 윈도우 (강한 노이즈 억제)
    else:
        base_window = 11  # 약 0.22초 윈도우 (일반 센서 데이터용)
        
    window = base_window if len(t_new) > base_window else (len(t_new) // 2 * 2 + 1)
    if window < 3: window = 3
    poly = 3 if window > 3 else 1

    # 1차 미분 (가속도)
    ax = savgol_filter(vx_new, window_length=window, polyorder=poly, deriv=1, delta=DT)
    ay = savgol_filter(vy_new, window_length=window, polyorder=poly, deriv=1, delta=DT)
    az = savgol_filter(vz_new, window_length=window, polyorder=poly, deriv=1, delta=DT)
    a_vec = np.vstack((ax, ay, az)).T
    accel_mag = np.linalg.norm(a_vec, axis=1)
    
    # 2차 미분 (Jerk)
    jx = savgol_filter(ax, window_length=window, polyorder=poly, deriv=1, delta=DT)
    jy = savgol_filter(ay, window_length=window, polyorder=poly, deriv=1, delta=DT)
    jz = savgol_filter(az, window_length=window, polyorder=poly, deriv=1, delta=DT)
    jerk_mag = np.linalg.norm(np.vstack((jx, jy, jz)).T, axis=1)
    
    # Curvature
    cross_va = np.cross(v_vec, a_vec)
    cross_mag = np.linalg.norm(cross_va, axis=1)
    curvature = cross_mag / (speed**3 + 1e-6)
    
    # Yaw Rate & Slip
    heading_traj = np.unwrap(np.arctan2(vy_new, vx_new))
    yaw_rate_traj = savgol_filter(heading_traj, window_length=window, polyorder=poly, deriv=1, delta=DT)
    yaw_rate_att = savgol_filter(yaw_new, window_length=window, polyorder=poly, deriv=1, delta=DT)
    slip_rate = yaw_rate_traj - yaw_rate_att
    
    # 💡 [반영 3] 가짜 스파이크 방어: 물리적 한계치 초과 값 강제 클리핑 (Clipping)
    # 실제 드론의 움직임 한계를 벗어나는 값(노이즈)은 잘라냅니다.
    MAX_JERK = 80.0       # 최대 Jerk 한계 (기체 특성에 맞춰 조절 가능)
    MAX_CURVATURE = 5.0   # 최대 곡률 한계
    jerk_mag = np.clip(jerk_mag, 0, MAX_JERK)
    curvature = np.clip(curvature, 0, MAX_CURVATURE)

    #(N, 7) 2D Array 
    features = np.vstack((speed, accel_mag, jerk_mag, curvature, yaw_rate_traj, yaw_rate_att, slip_rate)).T

    #### EXCLUDE non-flight data ####
    FLIGHT_THRESHOLD = 4.0  
    altitude = np.abs(z_new)  
    flight_mask = altitude > FLIGHT_THRESHOLD
    filtered_features = features[flight_mask]
    
    if len(filtered_features) == 0:
        FLIGHT_THRESHOLD = 0.1
        flight_mask = altitude > FLIGHT_THRESHOLD
        filtered_features = features[flight_mask]
    
    if len(filtered_features) == 0:
        return None
    
    # 💡 [반영 2] Robust Normalization (Outlier 무시 정규화)
    # 극단적인 Min/Max 대신 하위 1%와 상위 99% 값을 기준으로 스케일링합니다.
    if use_global_normalize:
        stats = get_global_normalization_stats()
        p1 = stats['p1']
        p99 = stats['p99']
    else:
        # Per-file 정규화 시에도 Robust Scaler 로직 적용
        p1 = np.percentile(filtered_features, 1, axis=0, keepdims=True)
        p99 = np.percentile(filtered_features, 99, axis=0, keepdims=True)
        
    feature_range = p99 - p1
    feature_range[feature_range == 0] = 1  # 0으로 나누기 방지
    
    # -1 ~ 1 사이로 변환 후, 1% / 99%를 벗어난 극단값은 -1 또는 1로 잘라냄(Clip)
    normalized_features = 2 * (filtered_features - p1) / feature_range - 1
    normalized_features = np.clip(normalized_features, -1.0, 1.0)

    return normalized_features, t_new, x_new, y_new, z_new, vx_new, vy_new, vz_new

def process_px4_ulog(ulog_path):
    try:
        ulog = ULog(ulog_path)
        loc_data = ulog.get_dataset('vehicle_local_position').data
        t_loc = loc_data['timestamp'] / 1e6
        x, y, z = loc_data['x'], loc_data['y'], loc_data['z']
        vx, vy, vz = loc_data['vx'], loc_data['vy'], loc_data['vz']
        
        att_data = ulog.get_dataset('vehicle_attitude').data
        t_att = att_data['timestamp'] / 1e6
        q0, q1, q2, q3 = att_data['q[0]'], att_data['q[1]'], att_data['q[2]'], att_data['q[3]']
        yaw_att = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
        
        # ULog는 센서 데이터이므로 source_type='sensor'
        return resample_and_extract_features(t_loc, x, y, z, vx, vy, vz, t_att, yaw_att, source_type='sensor')
    except Exception as e:
        print(f"[PX4 Data Process Error] {ulog_path}: {e}")
        return None

def process_ardu_bin(bin_path):
    try:
        mlog = mavutil.mavlink_connection(bin_path)
        t_loc, x, y, z, vx, vy, vz, t_att, yaw_att = [], [], [], [], [], [], [], [], []
        
        while True:
            msg = mlog.recv_match(type=['XKF1', 'NKF1', 'ATT'], blocking=False)
            if not msg: break
            if msg.get_type() in ['XKF1', 'NKF1']:
                t_loc.append(msg.TimeUS / 1e6)
                x.append(msg.PN); y.append(msg.PE); z.append(msg.PD)
                vx.append(msg.VN); vy.append(msg.VE); vz.append(msg.VD)
            elif msg.get_type() == 'ATT':
                t_att.append(msg.TimeUS / 1e6)
                yaw_att.append(np.radians(msg.Yaw))
                
        if len(x) < 50 or len(yaw_att) < 50: return None
        
        # Bin은 센서 데이터이므로 source_type='sensor'
        return resample_and_extract_features(
            np.array(t_loc), np.array(x), np.array(y), np.array(z), 
            np.array(vx), np.array(vy), np.array(vz), 
            np.array(t_att), np.array(yaw_att),
            source_type='sensor'
        )
    except Exception as e:
        print(f"[ArduPilot Data Process Error] {bin_path}: {e}")
        return None

def process_csv_odom(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        t = df['header_stamp_ns'].values / 1e9 
        x = df['x'].values
        y = df['y'].values
        z = df['z'].values
        
        time_diffs = np.concatenate([[DT], np.diff(t)]) 
        vx = np.concatenate([[0], np.diff(x) / np.diff(t)])
        vy = np.concatenate([[0], np.diff(y) / np.diff(t)])
        vz = np.concatenate([[0], np.diff(z) / np.diff(t)])
        
        qx = df['qx'].values
        qy = df['qy'].values
        qz = df['qz'].values
        qw = df['qw'].values
        
        yaw_att = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        
        if len(x) < 50 or len(yaw_att) < 50:
            return None
        
        # 💡 CSV 데이터는 계단형 노이즈가 심하므로 source_type='csv'로 지정하여 강한 스무딩 적용
        return resample_and_extract_features(
            t, x, y, z, vx, vy, vz, t, yaw_att, source_type='csv'
        )
    except Exception as e:
        print(f"[CSV Data Process Error] {csv_path}: {e}")
        return None


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        ulog = ULog(ulog_path)
        loc_data = ulog.get_dataset('vehicle_local_position').data
        t_loc = loc_data['timestamp'] / 1e6
        x_orig, y_orig, z_orig = loc_data['x'], loc_data['y'], loc_data['z']
        vx_orig, vy_orig, vz_orig = loc_data['vx'], loc_data['vy'], loc_data['vz']
        
        att_data = ulog.get_dataset('vehicle_attitude').data
        t_att = att_data['timestamp'] / 1e6
        q0, q1, q2, q3 = att_data['q[0]'], att_data['q[1]'], att_data['q[2]'], att_data['q[3]']
        yaw_att = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
        
        result = resample_and_extract_features(t_loc, x_orig, y_orig, z_orig, vx_orig, vy_orig, vz_orig, t_att, yaw_att)
        features, t_new, x_new, y_new, z_new, vx_new, vy_new, vz_new = result
        

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Original vs Resampling (50Hz)\n{os.path.basename(ulog_path)}', fontsize=14, fontweight='bold')
        
        # X Axis
        axes[0, 0].plot(t_loc, x_orig, 'b-', alpha=0.5, linewidth=1, label='Original Dataset', marker='.')
        axes[0, 0].plot(t_new, x_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[0, 0].set_ylabel('X Position (m)', fontsize=10)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('X Position')
        
        # Y Axis
        axes[0, 1].plot(t_loc, y_orig, 'b-', alpha=0.5, linewidth=1, label='Original Dataset', marker='.')
        axes[0, 1].plot(t_new, y_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[0, 1].set_ylabel('Y Position (m)', fontsize=10)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Y Position')
        
        # Z Axis
        axes[1, 0].plot(t_loc, z_orig, 'b-', alpha=0.5, linewidth=1, label='Original Dataset', marker='.')
        axes[1, 0].plot(t_new, z_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[1, 0].set_ylabel('Z Position (m)', fontsize=10)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Z Position')
        
        # VX Axis
        axes[1, 1].plot(t_loc, vx_orig, 'b-', alpha=0.5, linewidth=1, label='Original Dataset', marker='.')
        axes[1, 1].plot(t_new, vx_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[1, 1].set_ylabel('VX Velocity (m/s)', fontsize=10)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('VX Velocity')
        
        # VY Axis
        axes[2, 0].plot(t_loc, vy_orig, 'b-', alpha=0.5, linewidth=1, label='Original Dataset', marker='.')
        axes[2, 0].plot(t_new, vy_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[2, 0].set_xlabel('Time (s)', fontsize=10)
        axes[2, 0].set_ylabel('VY Velocity (m/s)', fontsize=10)
        axes[2, 0].legend(fontsize=9)
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_title('VY Velocity')
        
        # VZ Axis
        axes[2, 1].plot(t_loc, vz_orig, 'b-', alpha=0.5, linewidth=1, label='Original Dataset', marker='.')
        axes[2, 1].plot(t_new, vz_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[2, 1].set_xlabel('Time (s)', fontsize=10)
        axes[2, 1].set_ylabel('VZ Velocity (m/s)', fontsize=10)
        axes[2, 1].legend(fontsize=9)
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_title('VZ Velocity')
        
        plt.tight_layout()
        
        # Save PNG 
        png_filename = os.path.basename(ulog_path).replace('.ulg', '_comparison.png')
        png_path = os.path.join(output_dir, png_filename)
        plt.savefig(png_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Save Comparison Graph: {png_path}")
        return png_path
        
    except Exception as e:
        print(f"[Visualization Error] {ulog_path}: {e}")
        return None
    
