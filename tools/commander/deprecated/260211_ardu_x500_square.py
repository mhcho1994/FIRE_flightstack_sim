#!/usr/bin/env python3
"""
Takeoff -> fly a square -> land using pymavlink.

Tested pattern:
- Connect to SITL via UDP (e.g., udp:127.0.0.1:14550)
- Set mode GUIDED
- Arm
- Takeoff to target altitude
- Send LOCAL_NED position setpoints to trace a square
- Land

Notes:
- LOCAL_NED uses NED frame: x North (m), y East (m), z Down (m)
  So altitude +10m corresponds to z = -10.
- For smooth control, we stream setpoints at ~10 Hz while holding each leg.
"""

import time
import math
from pymavlink import mavutil


# ------------- Helpers -------------
def wait_heartbeat(m):
    m.wait_heartbeat()
    print(f"Heartbeat from system={m.target_system} component={m.target_component}")


def set_mode(m, mode: str, timeout_s: float = 10.0):
    """
    Works best with ArduPilot (GUIDED/LAND). PX4 uses OFFBOARD, etc.
    """
    if mode not in m.mode_mapping():
        raise RuntimeError(f"Mode {mode} not in mode mapping: {list(m.mode_mapping().keys())}")

    mode_id = m.mode_mapping()[mode]
    m.mav.set_mode_send(
        m.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb is None:
            continue
        current_mode = mavutil.mode_string_v10(hb)
        if current_mode == mode:
            print(f"Mode set to {mode}")
            return
    raise TimeoutError(f"Failed to set mode to {mode} within {timeout_s}s")


def arm(m, timeout_s: float = 10.0):
    m.mav.command_long_send(
        m.target_system,
        m.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0,
    )

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if hb is None:
            continue
        armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
        if armed:
            print("Armed")
            return
    raise TimeoutError("Failed to arm")


def takeoff(m, alt_m: float, timeout_s: float = 20.0):
    """
    Uses MAV_CMD_NAV_TAKEOFF. For ArduCopter in GUIDED this works.
    """
    m.mav.command_long_send(
        m.target_system,
        m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0,
        alt_m,
    )
    print(f"Takeoff command sent: alt={alt_m:.1f}m")

    # Wait until relative altitude reaches ~alt_m*0.9
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if msg is None:
            continue
        rel_alt = msg.relative_alt / 1000.0  # mm -> m
        if rel_alt >= 0.9 * alt_m:
            print(f"Reached rel_alt={rel_alt:.2f}m")
            return
    print("Takeoff wait timeout (continuing anyway)")


def send_local_position_setpoint(m, x_n: float, y_e: float, z_down: float, yaw_rad: float = None):
    """
    Send SET_POSITION_TARGET_LOCAL_NED.
    We ignore velocity/accel and only command position (plus optional yaw).
    """
    # type_mask bits: ignore what we don't provide.
    # See MAVLink SET_POSITION_TARGET_LOCAL_NED:
    # bit0-2 pos, bit3-5 vel, bit6-8 accel, bit9 force, bit10 yaw, bit11 yaw_rate
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
        int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms (fine to use wallclock here for SITL)
        m.target_system,
        m.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        type_mask,
        x_n, y_e, z_down,   # position
        0, 0, 0,            # velocity (ignored)
        0, 0, 0,            # accel (ignored)
        yaw, 0.0            # yaw, yaw_rate
    )


def hold_position(m, x_n: float, y_e: float, alt_m: float, hold_s: float, rate_hz: float = 10.0, yaw_rad: float = None):
    """
    Stream position setpoint for hold_s seconds.
    alt_m is Up altitude (positive up). Convert to z_down.
    """
    z_down = -alt_m
    dt = 1.0 / rate_hz
    t0 = time.time()
    while time.time() - t0 < hold_s:
        send_local_position_setpoint(m, x_n, y_e, z_down, yaw_rad=yaw_rad)
        time.sleep(dt)


def land(m, timeout_s: float = 5.0):
    # Option A: set LAND mode (ArduPilot)
    try:
        set_mode(m, "LAND", timeout_s=timeout_s)
        print("Landing (LAND mode)")
        return
    except Exception:
        pass

    # Option B: command land
    m.mav.command_long_send(
        m.target_system,
        m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0,
        0, 0, 0, 0, 0, 0, 0,
    )
    print("Landing command sent (NAV_LAND)")


# ------------- Main mission -------------
def main():
    # ---- User-tunable params ----
    connection = "udp:127.0.0.1:14550"
    takeoff_alt_m = 10.0
    square_size_m = 20.0        # edge length
    cruise_speed_mps = 2.0      # used to compute leg time (approx)
    corner_hold_s = 2.0
    setpoint_rate_hz = 50.0

    # Square in LOCAL_NED around origin (0,0) at fixed altitude.
    # Path: (0,0) -> (L,0) -> (L,L) -> (0,L) -> (0,0)
    L = square_size_m
    waypoints = [
        (0.0, 0.0),
        (L, 0.0),
        (L, L),
        (0.0, L),
        (0.0, 0.0),
    ]

    print(f"Connecting: {connection}")
    m = mavutil.mavlink_connection(connection)
    wait_heartbeat(m)

    # Some firmwares need a few setpoints before entering GUIDED/OFFBOARD reliably
    # We can "prime" by streaming the current (0,0,-alt) a little bit later; for now just proceed.

    set_mode(m, "GUIDED")
    arm(m)
    takeoff(m, takeoff_alt_m)

    # Optional: face along path (yaw) - here keep yaw fixed (None) or compute yaw per segment
    yaw_mode = "fixed"  # "fixed" or "along_path"
    fixed_yaw = None    # e.g. 0.0

    print("Flying square...")
    for i in range(1, len(waypoints)):
        x0, y0 = waypoints[i - 1]
        x1, y1 = waypoints[i]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)

        leg_time = max(dist / max(cruise_speed_mps, 1e-3), 1.0)

        if yaw_mode == "along_path":
            yaw = math.atan2(dy, dx)  # yaw in local frame
        else:
            yaw = fixed_yaw

        # Stream setpoints toward the next corner for leg_time seconds.
        # This is a simple "go to and hold" style; for tighter tracking you can add a feedback loop using LOCAL_POSITION_NED.
        hold_position(
            m,
            x_n=x1,
            y_e=y1,
            alt_m=takeoff_alt_m,
            hold_s=leg_time,
            rate_hz=setpoint_rate_hz,
            yaw_rad=yaw,
        )
        # Briefly hold at corner
        hold_position(
            m,
            x_n=x1,
            y_e=y1,
            alt_m=takeoff_alt_m,
            hold_s=corner_hold_s,
            rate_hz=setpoint_rate_hz,
            yaw_rad=yaw,
        )

        print(f"Corner {i}/{len(waypoints)-1}: ({x1:.1f}, {y1:.1f})")

    land(m)

    print("Done. (If you want, you can also wait for disarm here.)")


if __name__ == "__main__":
    main()
