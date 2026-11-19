
import numpy as np
from pyulog import ULog
from pymavlink import mavutil

#test purpose
import os 
from pathlib import Path

PX4_BASE_DIR = Path("../../data/px4_logs/")
ARDU_DIR = Path("../../data/ardu_logs/") #need to fix later 

#will delete in the future
def extract_from_ulog(ulog_path):
    try:
        ulog = ULog(ulog_path)
        data = ulog.get_dataset('vehicle_local_position').data
        
        x, y, z = data['x'], data['y'], data['z']
        vx, vy, vz = data['vx'], data['vy'], data['vz']
        
        features = np.array([
            np.max(x) - np.min(x), np.max(y) - np.min(y), np.max(z) - np.min(z),
            np.std(vx), np.std(vy), np.std(vz),
            np.mean(z), np.max(np.sqrt(vx**2 + vy**2 + vz**2))
        ])
        return features
#need to fix later 
    except Exception as e:
        print(f"[Warning] {ulog_path} Parsing Failed: {e}")
        return None

def parse_ulog_raw(ulog_path):
    try:
        ulog = ULog(ulog_path)
        output_px4 = { 'local_N': [],
                    'local_E': [],
                    'local_D': [],
                    'local_VN': [],
                    'local_VE': [],
                    'local_VD': [],
                    'local_heading': [],
                    'attitude_yaw': [],
                    'trj_yaw': [],
                    'trj_yaw_speed': []
                    }
        # 1. vehicle local position data (From VehicleLocalPosition.msg)
        loc_data = ulog.get_dataset('vehicle_local_position').data
        tsec_loc = loc_data['timestamp'] / 1e6 # micro to seconds
        output_px4['local_N'].append((tsec_loc, loc_data['x']))
        output_px4['local_E'].append((tsec_loc, loc_data['y']))
        output_px4['local_D'].append((tsec_loc, loc_data['z']))
        output_px4['local_VN'].append((tsec_loc, loc_data['vx']))
        output_px4['local_VE'].append((tsec_loc, loc_data['vy']))
        output_px4['local_VD'].append((tsec_loc, loc_data['vz']))
        output_px4['local_heading'].append((tsec_loc, loc_data['heading']))

        #location (North, East, Down (negative altitude) = x,y,z)
        x, y, z = loc_data['x'], loc_data['y'], loc_data['z']
        #velocity
        vx, vy, vz = loc_data['vx'], loc_data['vy'], loc_data['vz']
        #heading
        heading = loc_data['heading']
        
        # 2.yaw rate from attitude 
        att_data = ulog.get_dataset('vehicle_attitude').data
        tsec_att = att_data['timestamp'] / 1e6
        # quaternion to yaw
        q0, q1, q2, q3 = att_data['q[0]'], att_data['q[1]'], att_data['q[2]'], att_data['q[3]']
        yaw_att = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
        
        output_px4['attitude_yaw'].append((tsec_att, yaw_att))
        
        #3. trajectory setpoint data (From TrajectorySetpoint.msg)
        trj_data = ulog.get_dataset('trajectory_setpoint').data
        #setpoint timestamp
        tsec_trj = trj_data['timestamp'] / 1e6
        #setpoint yaw
        yaw_trj = trj_data['yaw']   
        #yaw setpoint speed
        yaw_sp_trj = trj_data['yawspeed']
        output_px4['trj_yaw'].append((tsec_trj, trj_data['yaw']))
        output_px4['trj_yaw_speed'].append((tsec_trj, trj_data['yawspeed']))  

        #메세지에 존재하나 offboard모드여서 기록안된 값들
        # pos_trj = trj_data['position']   
        # vel_trj = trj_data['velocity']   
        # acc_trj = trj_data['acceleration']   
        # jerk_trj = trj_data['jerk']   

        # print(f"position data: {x}, {y}, {z}\nvelocity data: {vx}, {vy}, {vz}\nattitude data: {yaw_att}")   
        # print(f"heading data: {heading}")
        print(f"local_N: {output_px4['local_N']}\nlocal_E: {output_px4['local_E']}\nlocal_D: {output_px4['local_D']}\n")
        print(f"local_VN: {output_px4['local_VN']}\nlocal_VE: {output_px4['local_VE']}\nlocal_VD: {output_px4['local_VD']}\n")
        print(f"local_heading: {output_px4['local_heading']}\n")
        print(f"attitude_yaw: {output_px4['attitude_yaw']}\ntrj_yaw: {output_px4['trj_yaw']}\ntrj_yaw_speed: {output_px4['trj_yaw_speed']}\n")
        print(f"trj_yaw: {output_px4['trj_yaw']}\ntrj_yaw_speed: {output_px4['trj_yaw_speed']}\n")
        print(f"just yaw sp trj:{yaw_sp_trj}")
        
        return output_px4
            

    except Exception as e:
        print(f"[Warning] {ulog_path} Parsing Failed: {e}")
        return None

#will delete in the future
def extract_from_bin(bin_path):
    try:
        mlog = mavutil.mavlink_connection(bin_path)
        
        x, y, z = [], [], []
        vx, vy, vz = [], [], []
        
        while True:
            msg = mlog.recv_match(type=['XKF1', 'NKF1'], blocking=False)
            if not msg:
                break
            
            # (North, East, Down) 방향의 위치(m)
            x.append(msg.PN)
            y.append(msg.PE)
            z.append(msg.PD)
            
            # VN, VE, VD는 각각 해당 방향의 속도(m/s)
            vx.append(msg.VN)
            vy.append(msg.VE)
            vz.append(msg.VD)
            
        if len(x) == 0:
            print(f"[Warning] {bin_path} No EKF position data (XKF1/NKF1).")
            return None
            
        x, y, z = np.array(x), np.array(y), np.array(z)
        vx, vy, vz = np.array(vx), np.array(vy), np.array(vz)
        
        # 앞서 PX4와 동일한 방식으로 8개의 특징값 계산
        features = np.array([
            np.max(x) - np.min(x), np.max(y) - np.min(y), np.max(z) - np.min(z),
            np.std(vx), np.std(vy), np.std(vz),
            np.mean(z), np.max(np.sqrt(vx**2 + vy**2 + vz**2))
        ])
        return features
    except Exception as e:
        print(f"[Warning] {bin_path} Parsing Failed: {e}")
        return None

def parse_bin_raw(bin_path):
    try:
        mlog = mavutil.mavlink_connection(bin_path)
        
        output_ardup = { 'SIM.Roll' : [],
                    'SIM.Pitch' : [],
                    'SIM.Yaw' : [],
                    'EK3.Roll' : [],
                    'EK3.Pitch' : [],
                    'EK3.Yaw' : [],
                    'EST.Roll' : [],
                    'EST.Pitch' : [],
                    'EST.Yaw' : [],
                    'EK3.PN' : [],
                    'EK3.PE' : [],
                    'EK3.PD' : [],
                    'EST.PN' : [],
                    'EST.PE' : [],
                    'EST.PD' : [],
                    }

        while True:
            m = mlog.recv_match(type=['XKF1','SIM'])
            if m is None:
                break
            t = m.get_type()

            if t == 'XKF1' and m.C == 0:
                # output_ardup attitude of first EKF3 lane
                tsec = m.TimeUS*1.0e-6
                output_ardup['EK3.Roll'].append((tsec, m.Roll))
                output_ardup['EK3.Pitch'].append((tsec, m.Pitch))
                output_ardup['EK3.Yaw'].append((tsec, m.Yaw))
                output_ardup['EK3.PN'].append((tsec, m.PN))
                output_ardup['EK3.PE'].append((tsec, m.PE))
                output_ardup['EK3.PD'].append((tsec, m.PD))

            if t == 'SIM':
                # simulator attitude
                tsec = m.TimeUS*1.0e-6
                output_ardup['SIM.Roll'].append((tsec, m.Roll))
                output_ardup['SIM.Pitch'].append((tsec, m.Pitch))
                output_ardup['SIM.Yaw'].append((tsec, m.Yaw))
        print(f"SIM Roll: {output_ardup['SIM.Roll']}\nSIM Pitch: {output_ardup['SIM.Pitch']}\nSIM Yaw: {output_ardup['SIM.Yaw']}\n")
        print(f"EK3 Roll: {output_ardup['EK3.Roll']}\nEK3 Pitch: {output_ardup['EK3.Pitch']}\nEK3 Yaw: {output_ardup['EK3.Yaw']}\n")
        print(f"EK3 PN: {output_ardup['EK3.PN']}\nEK3 PE: {output_ardup['EK3.PE']}\nEK3 PD: {output_ardup['EK3.PD']}\n")

        return output_ardup
        
    except Exception as e:
        print(f"[Warning] {bin_path} Parsing Failed: {e}")
        return None



#test purpose
if os.path.exists(PX4_BASE_DIR):
        for file in os.listdir(PX4_BASE_DIR):
            if file.endswith('.ulg'):
                parse_ulog_raw(os.path.join(PX4_BASE_DIR, file))

if os.path.exists(ARDU_DIR):
    for file in os.listdir(ARDU_DIR):
        if file.endswith('.BIN'):
            feat = parse_bin_raw(os.path.join(ARDU_DIR, file))