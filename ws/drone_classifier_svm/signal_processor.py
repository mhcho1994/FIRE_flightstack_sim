from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pyulog import ULog
from pymavlink import mavutil
import os
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib
# matplotlib.use('Agg')  

TARGET_HZ = 50.0  
DT = 1.0 / TARGET_HZ #20ms
FEATURE_MAP = {
    'Altitude': 0,
    'Heading': 1,
    'VZ': 2,
    'XY-Speed': 3,
    'AZ': 4,
    'XY-Accel': 5,
    'JZ': 6,
    'XY-Jerk': 7,
    'Curvature': 8,
    'YawRate': 9,
    'SlipRate': 10
}

def extract_flight_segments(t, features, dt=0.02, min_turn_duration=0.5, 
                          yaw_rate_threshold=0.2, straight_duration=1.0, heading_margin=0.2):
    spans = {'takeoff': None, 'landing': None, 'straight': [], 'turn': []}
    segments = {'takeoff': None, 'landing': None, 'straight': [], 'turn': []}

    if features is None or len(features) < 2:
        return segments, spans
        
    N = len(features)
    alt = features[:, 0]
    vz_norm = np.abs(features[:, 2])
    az_norm = np.abs(features[:, 4])
    jerk_norm = np.abs(features[:, 7])  
    
    alt_50 = np.percentile(alt, 50)
    target_alt = alt_50 - 0.1
    
    # [Step 1] Landing/Take-off
    flight_start_idx = 0
    state = 0
    for i in range(N):
        if state == 0 and alt[i] >= target_alt: state = 1
        # if state == 0: state = 1
        elif state == 1 and vz_norm[i] <= 0.2: state = 2
        elif state == 2 and az_norm[i] <= 0.2:
            flight_start_idx = i
            break
                
    flight_end_idx = N
    state = 0
    for i in range(N - 1, flight_start_idx, -1):
        if state == 0 and alt[i] >= target_alt: state = 1
        # if state == 0: state = 1
        elif state == 1 and vz_norm[i] <= 0.2: state = 2
        elif state == 2 and az_norm[i] <= 0.2:
            flight_end_idx = i
            break

    if flight_start_idx > 0:
        spans['takeoff'] = (t[0], t[flight_start_idx])
        segments['takeoff'] = {
            'label': 'takeoff',
            'time': t[0:flight_start_idx],
            'features': features[0:flight_start_idx],
            'span': (t[0], t[flight_start_idx])
        }
    
    if flight_end_idx < N - 1:
        spans['landing'] = (t[flight_end_idx], t[-1])
        segments['landing'] = {
            'label': 'landing',
            'time': t[flight_end_idx:],
            'features': features[flight_end_idx:],
            'span': (t[flight_end_idx], t[-1])
        }
                
    flight_t = t[flight_start_idx:flight_end_idx]
    flight_features = features[flight_start_idx:flight_end_idx]
    
    if len(flight_features) == 0:
        return segments, spans
        
    # [Step 2] Straight 
    is_straight = np.zeros(len(flight_features), dtype=bool)
    window_size = int(straight_duration / dt) 
    
    if len(flight_features) >= window_size:
        for i in range(len(flight_features) - window_size + 1):
            window_heading = flight_features[i : i + window_size, 1]
            window_yaw_rate_mag = np.abs(flight_features[i : i + window_size, 9])
            
            if np.max(window_yaw_rate_mag) <= yaw_rate_threshold:
                heading_diff = np.max(window_heading) - np.min(window_heading)
                if heading_diff <= heading_margin:
                    is_straight[i : i + window_size] = True

    edges = np.diff(is_straight.astype(int))
    st_starts = np.where(edges == 1)[0] + 1
    st_ends = np.where(edges == -1)[0] + 1
    if is_straight[0]: st_starts = np.insert(st_starts, 0, 0)
    if is_straight[-1]: st_ends = np.append(st_ends, len(is_straight))
    
    for start, end in zip(st_starts, st_ends):
        safe_end = min(end, len(flight_t) - 1)
        spans['straight'].append((flight_t[start], flight_t[safe_end]))
        segments['straight'].append({
            'label': 'straight',
            'time': flight_t[start:safe_end],
            'features': flight_features[start:safe_end],
            'span': (flight_t[start], flight_t[safe_end])
        })

    straight_indices = np.where(is_straight)[0]
    if len(straight_indices) == 0:
        return segments, spans 
        
    first_straight_idx = straight_indices[0]

    # [Step 3] Turn 
    is_turning = ~is_straight 
    is_turning[:first_straight_idx] = False
    
    edges = np.diff(is_turning.astype(int))
    turn_starts = np.where(edges == 1)[0] + 1
    turn_ends = np.where(edges == -1)[0] + 1
    
    if is_turning[0]: turn_starts = np.insert(turn_starts, 0, 0)
    if is_turning[-1]: turn_ends = np.append(turn_ends, len(is_turning))
        
    min_length = int(min_turn_duration / dt) 
    
    for start, end in zip(turn_starts, turn_ends):
        if (end - start) < min_length:
            continue
            
        safe_end = min(end, len(flight_t) - 1)
        spans['turn'].append((flight_t[start], flight_t[safe_end]))
        segments['turn'].append({
            'label': 'turn',
            'time': flight_t[start:safe_end],
            'features': flight_features[start:safe_end],
            'span': (flight_t[start], flight_t[safe_end])
        })
        
    return segments, spans

def extract_kinematic_features(t, x, y, z, vx, vy, vz, window_len=200, poly_order=3):
    t, unique_indices = np.unique(t, return_index=True)
    x, y, z = x[unique_indices], y[unique_indices], z[unique_indices]
    vx, vy, vz = vx[unique_indices], vy[unique_indices], vz[unique_indices]
    
    if len(t) < 2: return None

    t_start, t_end = t[0], t[-1]
    DT = 0.02 
    t_new = np.arange(t_start, t_end, DT) 
    
    x_new = interp1d(t, x, bounds_error=False, fill_value="extrapolate")(t_new)
    y_new = interp1d(t, y, bounds_error=False, fill_value="extrapolate")(t_new)
    z_new = interp1d(t, z, bounds_error=False, fill_value="extrapolate")(t_new)
    vx_new = interp1d(t, vx, bounds_error=False, fill_value="extrapolate")(t_new)
    vy_new = interp1d(t, vy, bounds_error=False, fill_value="extrapolate")(t_new)
    vz_new = interp1d(t, vz, bounds_error=False, fill_value="extrapolate")(t_new)
    
    def smooth(signal):
        wl = window_len
        if len(signal) < wl:
            wl = len(signal) if len(signal) % 2 != 0 else len(signal) - 1
            if wl <= poly_order: return signal 
        return savgol_filter(signal, window_length=wl, polyorder=poly_order)

    z_smooth, vx_smooth, vy_smooth, vz_smooth = smooth(z_new), smooth(vx_new), smooth(vy_new), smooth(vz_new)

    altitude = z_smooth
    v_alt = vz_smooth
    v_xy = np.vstack((vx_smooth, vy_smooth)).T
    speed_xy = np.linalg.norm(v_xy, axis=1)
    
    ax, ay, az = np.gradient(vx_smooth, DT), np.gradient(vy_smooth, DT), np.gradient(vz_smooth, DT)
    a_alt = az
    a_xy = np.vstack((ax, ay)).T
    acc_norm_xy = np.linalg.norm(a_xy, axis=1)
    
    jx, jy, jz = np.gradient(smooth(ax), DT), np.gradient(smooth(ay), DT), np.gradient(smooth(az), DT)
    j_alt = jz
    j_xy = np.vstack((jx, jy)).T
    jerk_norm_xy = np.linalg.norm(j_xy, axis=1)
    
    heading = np.unwrap(np.arctan2(vy_smooth, vx_smooth))
    raw_yaw_rate = np.gradient(heading, DT)
    yaw_rate = smooth(raw_yaw_rate)
    slip_rate = heading - yaw_rate

    v_vec_3d = np.vstack((vx_smooth, vy_smooth, vz_smooth)).T
    a_vec_3d = np.vstack((ax, ay, az)).T
    speed_3d = np.linalg.norm(v_vec_3d, axis=1)
    cross_va = np.cross(v_vec_3d, a_vec_3d)
    cross_mag = np.linalg.norm(cross_va, axis=1)
    raw_curvature = cross_mag / (speed_3d**3 + 1e-6)
    curvature = smooth(raw_curvature)

    features = np.vstack((
        altitude, heading, v_alt, speed_xy, a_alt, acc_norm_xy, 
        j_alt, jerk_norm_xy, curvature, yaw_rate, slip_rate
    )).T
    
    return t_new, features


def process_px4_flight_data(ulog_path):
    try:
        ulog = ULog(ulog_path)
        loc_data = ulog.get_dataset('vehicle_local_position').data
        t_loc = loc_data['timestamp'] / 1e6
        x, y, z = loc_data['x'], loc_data['y'], loc_data['z']
        vx, vy, vz = loc_data['vx'], loc_data['vy'], loc_data['vz']
        
        z = -z
        vz = -vz
            
        extracted = extract_kinematic_features(t_loc, x, y, z, vx, vy, vz)
        if extracted is None: return None, None, None, None, None
        
        t_full, feat_full = extracted
        segments, spans = extract_flight_segments(t_full, feat_full)
        
        # Bundle raw data into a single dictionary
        raw_data = {'t': t_loc, 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}
        
        return raw_data, t_full, feat_full, segments, spans
        
    except Exception as e:
        print(f"[PX4 Extract Error] {ulog_path}: {e}")
        return None, None, None, None, None

def process_ardu_flight_data(bin_path):
    try:
        mlog = mavutil.mavlink_connection(bin_path)
        t_loc, x, y, z, vx, vy, vz = [], [], [], [], [], [], []
        
        while True:
            msg = mlog.recv_match(type=['XKF1', 'NKF1'], blocking=False)
            if not msg: break
            t_loc.append(msg.TimeUS / 1e6) 
            # negative z and vz
            x.append(msg.PN); y.append(msg.PE); z.append(-msg.PD)
            vx.append(msg.VN); vy.append(msg.VE); vz.append(-msg.VD)
                
        if len(x) < 50: return None, None, None, None, None
        
        extracted = extract_kinematic_features(np.array(t_loc), np.array(x), np.array(y), np.array(z), np.array(vx), np.array(vy), np.array(vz))
        if extracted is None: return None, None, None, None, None
        
        t_full, feat_full = extracted
        segments, spans = extract_flight_segments(t_full, feat_full)
        
        # Bundle raw data into a single dictionary
        raw_data = {'t': t_loc, 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}
        
        return raw_data, t_full, feat_full, segments, spans
        
    except Exception as e:
        print(f"[ArduPilot Extract Error] {bin_path}: {e}")
        return None, None, None, None, None

def process_rosbag_flight_data(csv_path):
    try:
        data = np.genfromtxt(csv_path, delimiter=',', names=True)
        t_loc = data['bag_time_ns'] / 1e9
        x, y, z = data['x'], data['y'], data['z']
        vx, vy, vz = data['vx'], data['vy'], data['vz']
        
        print(f"[ROS Bag] ROS bag is used. check z direction")
        z = -z
        vz = -vz
        
        extracted = extract_kinematic_features(t_loc, x, y, z, vx, vy, vz)
        if extracted is None: return None, None, None, None, None
        
        t_full, feat_full = extracted
        segments, spans = extract_flight_segments(t_full, feat_full)
        
        # Bundle raw data into a single dictionary
        raw_data = {'t': t_loc, 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}
        
        return raw_data, t_full, feat_full, segments, spans
        
    except Exception as e:
        print(f"[ROS Bag Extract Error] {csv_path}: {e}")
        return None, None, None, None, None

def process_real_flight_data(csv_path, measurement_type='vision'):
    try:
        data = np.genfromtxt(csv_path, delimiter=',', names=True)
        try : 
                t_loc = data['time_s']
        except ValueError:
            t_loc = data['timestamp']

        # Select target columns based on measurement_type
        if measurement_type == 'mocap':
            try:
                x, y, z = data['gtx'], data['gty'], data['gtz']
            except (ValueError, IndexError):
                x, y, z = data['gt_x'], data['gt_y'], data['gt_z']
        elif measurement_type == 'vision':
            try:
                x, y, z = data['xsmooth'], data['ysmooth'], data['zsmooth']
            except (ValueError, IndexError):
                x, y, z = data['x_smooth'], data['y_smooth'], data['z_smooth']
        else:
            print(f"[Error] Unknown measurement_type: {measurement_type}")
            return None, None, None, None, None
        
        # Calculate Velocity (1st derivative)
        vx = np.gradient(x, t_loc)
        vy = np.gradient(y, t_loc)
        vz = np.gradient(z, t_loc)

        extracted = extract_kinematic_features(t_loc, x, y, z, vx, vy, vz)
        if extracted is None: return None, None, None, None, None

        t_full, feat_full = extracted
        segments, spans = extract_flight_segments(t_full, feat_full)

        # Bundle raw data into a single dictionary
        raw_data = {'t': t_loc, 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}

        return raw_data, t_full, feat_full, segments, spans

    except Exception as e:
        print(f"[Real Flight Extract Error] {csv_path}: {e}")
        return None, None, None, None, None


def main():
    pass
    # BASE_DATA_DIR = Path("data/sitl_logs") 
    # print("[Info] Starting Combined Pipeline: XY / Full Highlights / Segments...\n")

    # if not BASE_DATA_DIR.exists():
    #     print(f"[Error] Data directory not found: {BASE_DATA_DIR}")
    #     return

    # run_folders = sorted([f for f in os.listdir(BASE_DATA_DIR) 
    #                      if f.startswith("run_")  and (BASE_DATA_DIR / f).is_dir()])

    # if not run_folders:
    #     print(f"[Warning] No run folders found in {BASE_DATA_DIR}")
    #     return

    # for run_folder in run_folders:
    #     run_dir = BASE_DATA_DIR / run_folder
        
    #     px4_dir = run_dir / "px4_logs" / "raw"
    #     ardu_dir = run_dir / "ardu_logs" / "raw" / "logs"
        
    #     x_px4, y_px4, t_px4, feat_px4, segments_px4, spans_px4 = (None,) * 6
    #     x_ardu, y_ardu, t_ardu, feat_ardu, segments_ardu, spans_ardu = (None,) * 6
        
    #     # Load PX4 Data
    #     if px4_dir.exists():
    #         for file in os.listdir(px4_dir):
    #             if file.lower().endswith('.ulg'):
    #                 px4_result = process_px4_flight_data(str(px4_dir / file))
    #                 # [Fix] Correctly unpack the segments dictionary
    #                 x_px4, y_px4, t_px4, feat_px4, segments_px4, spans_px4 = px4_result
    #                 break 

    #     # Load ArduPilot Data
    #     if ardu_dir.exists():
    #         for file in os.listdir(ardu_dir):
    #             if file.lower().endswith('.bin'):
    #                 ardu_result = process_ardu_flight_data(str(ardu_dir / file))
    #                 # [Fix] Correctly unpack the segments dictionary
    #                 x_ardu, y_ardu, t_ardu, feat_ardu, segments_ardu, spans_ardu = ardu_result
    #                 break

    #     # Plot Combined XY Trajectory
    #     if (x_px4 is not None) or (x_ardu is not None):
    #         plot_combined_xy_trajectory(
    #             x_px4, y_px4, x_ardu, y_ardu, 
    #             title=f"Combined X-Y Trajectory ({run_folder})", 
    #             save_path=str(run_dir / f"trajectory_xy_combined_{run_folder}.png")
    #         )

    #     # Plot PX4 Segment Highlights and Isolated Turn
    #     if feat_px4 is not None and len(feat_px4) > 0:
    #         print(f"[{run_folder}] Generating PX4 Trajectory plot (Visualizing Segments)...")
    #         plot_full_trajectory_with_spans(
    #             t=t_px4, features=feat_px4, spans=spans_px4, 
    #             title=f"Trajectory [Segment Check]: PX4 ({run_folder})", 
    #             save_path=str(run_dir / f"features_px4_seg_check_{run_folder}.png"),
    #             line_color='tab:green' 
    #         )
            
    #         # [Fix] Access the dictionary properly to extract turn features
    #         if segments_px4 and segments_px4['turn'] and len(segments_px4['turn']) > 0:
    #             t_turn = segments_px4['turn'][0]['time']
    #             feat_turn = segments_px4['turn'][0]['features']
                
    #             plot_turn_segment_features(
    #                 t=t_turn, features=feat_turn, 
    #                 title=f"Trajectory [Isolated Turn]: PX4 ({run_folder})", 
    #                 save_path=str(run_dir / f"features_px4_turn_seg_{run_folder}.png"),
    #                 line_color='tab:green' 
    #             )

    #     # Plot ArduPilot Segment Highlights and Isolated Turn
    #     if feat_ardu is not None and len(feat_ardu) > 0:
    #         print(f"[{run_folder}] Generating ArduPilot Trajectory plot (Visualizing Segments)...")
    #         plot_full_trajectory_with_spans(
    #             t=t_ardu, features=feat_ardu, spans=spans_ardu, 
    #             title=f"Trajectory [Segment Check]: ArduPilot ({run_folder})", 
    #             save_path=str(run_dir / f"features_ardupilot_seg_check_{run_folder}.png"),
    #             line_color='tab:orange' 
    #         )
            
    #         # [Fix] Access the dictionary properly to extract turn features
    #         if segments_ardu and segments_ardu['turn'] and len(segments_ardu['turn']) > 0:
    #             t_turn = segments_ardu['turn'][0]['time']
    #             feat_turn = segments_ardu['turn'][0]['features']
                
    #             plot_turn_segment_features(
    #                 t=t_turn, features=feat_turn, 
    #                 title=f"Trajectory [Isolated Turn]: ArduPilot ({run_folder})", 
    #                 save_path=str(run_dir / f"features_ardupilot_turn_seg_{run_folder}.png"),
    #                 line_color='tab:orange' 
    #             )

    # print("\n[Info] All combined XY and segmentation plots generated successfully!")


    
if __name__ == "__main__":
    main()