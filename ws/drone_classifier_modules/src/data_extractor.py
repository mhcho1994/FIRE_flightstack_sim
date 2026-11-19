import numpy as np
from pymavlink import mavutil
from pyulog import ULog


def parse_px4_ulog(ulog_path):
    """
    Parses a PX4 ULog file and extracts raw local position and velocity data.
    """
    try:
        ulog = ULog(ulog_path)
        loc_data = ulog.get_dataset('vehicle_local_position').data
        t_loc = loc_data['timestamp'] / 1e6
        x, y, z = loc_data['x'], loc_data['y'], loc_data['z']
        vx, vy, vz = loc_data['vx'], loc_data['vy'], loc_data['vz']
        
        # Invert Z-axis for standard NED to ENU-like visualization
        z = -z
        vz = -vz
            
        return {'t': t_loc, 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}
        
    except Exception as e:
        print(f"[PX4 Parse Error] {ulog_path}: {e}")
        return None

def parse_ardu_bin(bin_path):
    """
    Parses an ArduPilot DataFlash log (.bin) and extracts raw position and velocity data.
    """
    try:
        mlog = mavutil.mavlink_connection(bin_path)
        t_loc, x, y, z, vx, vy, vz = [], [], [], [], [], [], []
        
        while True:
            msg = mlog.recv_match(type=['XKF1', 'NKF1'], blocking=False)
            if not msg: break
            t_loc.append(msg.TimeUS / 1e6) 
            # Invert Z-axis for standard NED to ENU-like visualization
            x.append(msg.PN); y.append(msg.PE); z.append(-msg.PD)
            vx.append(msg.VN); vy.append(msg.VE); vz.append(-msg.VD)
                
        if len(x) < 50: return None
        
        return {
            't': np.array(t_loc), 'x': np.array(x), 'y': np.array(y), 'z': np.array(z), 
            'vx': np.array(vx), 'vy': np.array(vy), 'vz': np.array(vz)
        }
        
    except Exception as e:
        print(f"[ArduPilot Parse Error] {bin_path}: {e}")
        return None

def parse_real_csv(csv_path, measurement_type='vision'):
    """
    Parses a processed CSV file from real flight data (Mocap or Vision).
    """
    try:
        data = np.genfromtxt(csv_path, delimiter=',', names=True)
        try: 
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
            return None
        
        # Remove rows with NaN values in any of the critical columns
        valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z) & ~np.isnan(t_loc)
        t_loc = t_loc[valid_mask]
        x, y, z = x[valid_mask], y[valid_mask], z[valid_mask]
        
        if len(t_loc) < 10: return None
        
        # Calculate Velocity (1st derivative)
        vx = np.gradient(x, t_loc)
        vy = np.gradient(y, t_loc)
        vz = np.gradient(z, t_loc)

        return {'t': t_loc, 'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz}

    except Exception as e:
        print(f"[Real Flight Parse Error] {csv_path}: {e}")
        return None