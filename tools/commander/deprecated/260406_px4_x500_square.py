#!/usr/bin/env python3
import time
import math
from pymavlink import mavutil
import threading

def start_gcs_heartbeat(m, rate_hz: float = 1.0):
    """
    Send MAVLink HEARTBEAT as a Ground Control Station so PX4 considers GCS connected.
    Run in a background thread.
    """
    period = 1.0 / rate_hz
    stop_flag = {"stop": False}

    def _loop():
        while not stop_flag["stop"]:
            # GCS heartbeat
            m.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0
            )
            time.sleep(period)

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    return stop_flag

def wait_heartbeat(m):
    m.wait_heartbeat()
    print(f"Heartbeat from system={m.target_system} component={m.target_component}")

def print_statustext_nonblock(m, duration_s=0.0):
    """Drain STATUSTEXT for duration_s seconds (0 => just drain what's queued)."""
    t0 = time.time()
    while True:
        msg = m.recv_match(type="STATUSTEXT", blocking=False)
        if msg:
            # msg.severity, msg.text
            print(f"[STATUSTEXT sev={msg.severity}] {msg.text}")
            continue
        if duration_s <= 0:
            break
        if time.time() - t0 >= duration_s:
            break
        time.sleep(0.05)

def wait_command_ack(m, command, timeout_s=2.0):
    """Wait for COMMAND_ACK for a specific command."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
        if ack is None:
            continue
        if ack.command == command:
            return ack.result  # MAV_RESULT_*
    return None

def send_local_pos_sp(m, x_n, y_e, z_down, yaw_rad=None):
    IGNORE_VX = 1 << 3
    IGNORE_VY = 1 << 4
    IGNORE_VZ = 1 << 5
    IGNORE_AX = 1 << 6
    IGNORE_AY = 1 << 7
    IGNORE_AZ = 1 << 8
    IGNORE_YAW_RATE = 1 << 11

    if yaw_rad is None:
        IGNORE_YAW = 1 << 10
        type_mask = IGNORE_VX | IGNORE_VY | IGNORE_VZ | IGNORE_AX | IGNORE_AY | IGNORE_AZ | IGNORE_YAW | IGNORE_YAW_RATE
        yaw = 0.0
    else:
        type_mask = IGNORE_VX | IGNORE_VY | IGNORE_VZ | IGNORE_AX | IGNORE_AY | IGNORE_AZ | IGNORE_YAW_RATE
        yaw = float(yaw_rad)

    m.mav.set_position_target_local_ned_send(
        int(time.time() * 1000) & 0xFFFFFFFF,
        m.target_system,
        m.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        x_n, y_e, z_down,
        0, 0, 0,
        0, 0, 0,
        yaw, 0.0
    )

def get_local_pos_ned(m, timeout_s=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        msg = m.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=0.5)
        if msg:
            return (msg.x, msg.y, msg.z)
    return None

def set_mode_do_set_mode(m, mode_name: str, timeout_s=5.0):
    mapping = m.mode_mapping()
    if mode_name not in mapping:
        raise RuntimeError(f"Mode {mode_name} not in {list(mapping.keys())}")

    mode_val = mapping[mode_name]
    base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED

    # PX4 often gives int here, but handle tuple/list just in case
    if isinstance(mode_val, int):
        custom_mode = mode_val
    elif isinstance(mode_val, (tuple, list)) and len(mode_val) >= 2:
        base_mode = int(mode_val[0])
        custom_mode = int(mode_val[1])
    else:
        custom_mode = int(mode_val)

    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        float(base_mode), float(custom_mode),
        0, 0, 0, 0, 0
    )

    ack = wait_command_ack(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, timeout_s=2.0)
    if ack is not None:
        # 0=ACCEPTED, 1=TEMPORARILY_REJECTED, 2=DENIED, 3=UNSUPPORTED...
        print(f"DO_SET_MODE ACK result={ack}")

    # Wait for heartbeat mode update
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
        if hb:
            cur = mavutil.mode_string_v10(hb)
            if cur == mode_name:
                print(f"Mode set to {mode_name}")
                return True
        print_statustext_nonblock(m, duration_s=0.0)

    return False

def arm_disarm(m, arm=True, timeout_s=5.0):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1.0 if arm else 0.0,
        0, 0, 0, 0, 0, 0
    )
    ack = wait_command_ack(m, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout_s=2.0)
    if ack is not None:
        print(f"ARM_DISARM ACK result={ack}")

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
        if hb:
            armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
            if armed == arm:
                print("Armed" if arm else "Disarmed")
                return True
        print_statustext_nonblock(m, duration_s=0.0)
    return False

def hold_pos_stream(m, x, y, z_down, hold_s, rate_hz=30.0):
    dt = 1.0 / rate_hz
    t0 = time.time()
    while time.time() - t0 < hold_s:
        send_local_pos_sp(m, x, y, z_down)
        print_statustext_nonblock(m, duration_s=0.0)
        time.sleep(dt)

def main():
    connection = "udp:127.0.0.1:14550"
    takeoff_alt_m = 10.0
    square_size_m = 20.0
    cruise_speed_mps = 2.0
    corner_hold_s = 2.0
    rate_hz = 30.0

    print(f"Connecting: {connection}")
    m = mavutil.mavlink_connection(connection)
    wait_heartbeat(m)
    gcs_hb = start_gcs_heartbeat(m, rate_hz=1.0)


    # Read current local position
    p = get_local_pos_ned(m, timeout_s=3.0)
    if p is None:
        x0, y0, z0 = 0.0, 0.0, 0.0
        print("WARN: no LOCAL_POSITION_NED received; using (0,0,0).")
    else:
        x0, y0, z0 = p
        print(f"Current local NED: x={x0:.2f} y={y0:.2f} z={z0:.2f}")

    # Target takeoff z (NED): more negative => higher
    target_z = z0 - takeoff_alt_m

    # 1) PRIME setpoints longer + faster (PX4가 OFFBOARD 요구조건 만족하도록)
    prime_s = 4.0
    print(f"Priming setpoints for {prime_s}s at {rate_hz}Hz ...")
    hold_pos_stream(m, x0, y0, target_z, hold_s=prime_s, rate_hz=rate_hz)

    # 2) Try ARM first (some setups prefer arming in POSCTL/TAKEOFF)
    print("Trying to ARM (before OFFBOARD)...")
    arm_ok = arm_disarm(m, arm=True, timeout_s=5.0)
    if not arm_ok:
        print("ARM failed (before OFFBOARD). We'll try OFFBOARD first, then ARM again.")
    print_statustext_nonblock(m, duration_s=0.5)

    # 3) Switch to OFFBOARD (keep streaming setpoints during/after)
    print("Switching to OFFBOARD...")
    off_ok = set_mode_do_set_mode(m, "OFFBOARD", timeout_s=8.0)
    if not off_ok:
        print("OFFBOARD switch failed. Recent STATUSTEXT above should say why.")
        # Keep streaming a bit more and try again once
        print("Re-priming 2s then retry OFFBOARD...")
        hold_pos_stream(m, x0, y0, target_z, hold_s=2.0, rate_hz=rate_hz)
        off_ok = set_mode_do_set_mode(m, "OFFBOARD", timeout_s=8.0)
        if not off_ok:
            raise RuntimeError("OFFBOARD still failed. See STATUSTEXT output for the exact reason.")

    # 4) Ensure ARMED (if not already)
    if not arm_ok:
        print("Trying to ARM (in OFFBOARD)...")
        arm_ok = arm_disarm(m, arm=True, timeout_s=5.0)
        if not arm_ok:
            raise RuntimeError("ARM still failed. See STATUSTEXT output for the exact reason.")

    # 5) “Takeoff”: keep commanding the higher setpoint for a few seconds
    print("Climb/hold at takeoff altitude...")
    hold_pos_stream(m, x0, y0, target_z, hold_s=6.0, rate_hz=rate_hz)

    # 6) Square
    L = square_size_m
    wps = [(x0, y0), (x0 + L, y0), (x0 + L, y0 + L), (x0, y0 + L), (x0, y0)]
    print("Flying square...")
    for i in range(1, len(wps)):
        x_prev, y_prev = wps[i - 1]
        x1, y1 = wps[i]
        dist = math.hypot(x1 - x_prev, y1 - y_prev)
        leg_time = max(dist / max(cruise_speed_mps, 1e-3), 1.0)

        hold_pos_stream(m, x1, y1, target_z, hold_s=leg_time, rate_hz=rate_hz)
        hold_pos_stream(m, x1, y1, target_z, hold_s=corner_hold_s, rate_hz=rate_hz)
        print(f"Corner {i}/{len(wps)-1}: ({x1:.1f}, {y1:.1f})")

    # 7) Land
    print("Switching to LAND...")
    land_ok = set_mode_do_set_mode(m, "LAND", timeout_s=5.0)
    if not land_ok:
        print("LAND mode switch failed; STATUSTEXT should show why.")
    print("Done.")

if __name__ == "__main__":
    main()
