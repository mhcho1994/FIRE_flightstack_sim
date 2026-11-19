import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

TARGET_HZ = 50.0  
DT = 1.0 / TARGET_HZ 

FEATURE_MAP = {
    'Altitude': 0, 'Heading': 1, 'VZ': 2, 'XY-Speed': 3, 'AZ': 4,
    'XY-Accel': 5, 'JZ': 6, 'XY-Jerk': 7, 'Curvature': 8,
    'YawRate': 9, 'SlipRate': 10
}

def compute_kinematics(raw_data_dict, window_len=200, poly_order=3):
    """
    Interpolates raw data to 50Hz and calculates 11 kinematic features.
    """
    if raw_data_dict is None:
        return None, None
        
    t = raw_data_dict['t']
    x, y, z = raw_data_dict['x'], raw_data_dict['y'], raw_data_dict['z']
    vx, vy, vz = raw_data_dict['vx'], raw_data_dict['vy'], raw_data_dict['vz']

    t, unique_indices = np.unique(t, return_index=True)
    x, y, z = x[unique_indices], y[unique_indices], z[unique_indices]
    vx, vy, vz = vx[unique_indices], vy[unique_indices], vz[unique_indices]
    
    if len(t) < 2: return None, None

    t_start, t_end = t[0], t[-1]
    t_new = np.arange(t_start, t_end, DT) 
    
    # Interpolation
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

    # Feature Calculation
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