#!/usr/bin/env python3
"""
Scenario Generator (Pattern-Specific) + Mission Plan Metadata Embedder

Purpose:
- Generate multiple run directories (run_000, run_001, ...)
- For each run, write a scenario.yaml file
- The mission geometry (waypoints) is common
- Autopilot-specific simulation settings are separated (ArduPilot / PX4)

Design philosophy:
- Keep scenario geometry minimal and pattern-specific
- Avoid overly generic scenario generators
- Enable fair comparison between PX4 and ArduPilot using identical trajectories
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import yaml


# ----------------------------------------------------------------------
# Type definition
# ----------------------------------------------------------------------
# Waypoint representation in LLA frame
# Convert (north, east, down) to (lat, lon, alt) for mission upload
# NOTE:
#   Down is positive in NED.
#   If altitude above home is +5m,
#   then Down = -5.0
NED = Tuple[float, float, float]
# NOTE:
# Alt is meters above WGS84 ellipsoid (or AMSL depending on your convention)
# AltitudeMode meaning in QGC mission plan
# 0 RelativeToHome:
# 1	RelativeToHome: (default mission) QGC mission waypoint
# 2	AMSL: mean sea level altitude
# 3	Terrain: relative to terrain altitude
LLA = Tuple[float, float, float]
# NOTE:
# Specify waypoint speed in m/s (float) for each waypoint
SPD = float

# ----------------------------------------------------------------------
# MAVLink Command IDs
# ----------------------------------------------------------------------
# MAVLink command IDs
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_CHANGE_SPEED = 178

# ----------------------------------------------------------------------
# NED → LLA conversion
# ----------------------------------------------------------------------
# WGS84 constants
_WGS84_A = 6378137.0                  # semi-major axis [m]
_WGS84_E2 = 6.69437999014e-3          # eccentricity squared

def ned_to_lla(ned: NED, home_lla: LLA) -> LLA:
    """
    Convert local NED offset (m) to geodetic LLA using a small angle approximation.
    Accurate for small distances (typ. <~10-20km) around home.
    If large, please use a proper transformation: NED -> ECEF -> Geodetic.
    If needed, notify mhcho to implement a more accurate conversion.

    Args:
        ned: (north_m, east_m, down_m)
        home_lla: (lat_deg, lon_deg, alt_m(AMSL or WGS84))

    Returns:
        (lat_deg, lon_deg, alt_rel_m)

    Notes:
        - down is positive in NED; altitude increases upward.
        - alt_rel_m here is computed as (-1) * down.
        - This is accurate for small distances (typ. <~10-20km) around home.
    """
    north, east, down = ned
    lat0_deg, lon0_deg, h0 = home_lla

    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)

    sin_lat0 = np.sin(lat0)
    denom = np.sqrt(1.0 - _WGS84_E2 * sin_lat0 * sin_lat0)

    # radii of curvature
    Rn = _WGS84_A * (1.0 - _WGS84_E2) / (denom ** 3)    # meridian radius
    Re = _WGS84_A / denom                               # prime-vertical radius

    dlat = north / (Rn + h0)
    dlon = east  / ((Re + h0) * np.cos(lat0))

    lat = lat0 + dlat
    lon = lon0 + dlon
    alt_rel = -down # because down positive -> altitude decreases and relative to home position

    return (float(np.degrees(lat)), float(np.degrees(lon)), float(alt_rel))


# ----------------------------------------------------------------------
# Mission Specification Container
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MissionSpec:
    """
    Container describing mission geometry.

    name            : mission identifier
    takeoff_alt_m   : takeoff altitude (positive above home)
    speed_m_s       : cruise speed between waypoints
    waypoints_ned   : list of (N, E, D) waypoints
    land            : whether to land at end of mission

    Note: 
      - Altitude mode fixed to RelativeToHome (1)
      - Lists below are aligned by mission item index:
            len(command) == len(waypoints_lla) == len(waypoints_ned) == len(speed_m_s)
      - For commands that don't have a position (e.g., DO_CHANGE_SPEED),
        waypoints_* entry is None.
      - speed_m_s is only meaningful for DO_CHANGE_SPEED; otherwise None.
    """
    name: str
    home_position: List[LLA]    # list of home positions (lat, lon, alt) for each run
    command: List[int]          # list of MAVLink command IDs for each waypoint
    takeoff_alt_m: float        # takeoff altitude (positive above home)
    waypoints_ned: List[Optional[NED]]      # aligned with command list (None if no position)
    waypoints_lla: List[Optional[LLA]]      # aligned with command list (None if no position)
    speed_m_s: List[Optional[SPD]]          # aligned with command list (None if unused)
    land: bool = True


# ----------------------------------------------------------------------
# Pattern 1: 3-Point Turn Observation
# ----------------------------------------------------------------------
def make_turn_3pts(
    home_position: List[LLA],
    settle_m: float,
    leg1_m: float,
    leg2_m: float,
    turn_deg: float,
    alt_m: float,
    speed_m_s: float,
    land: bool = True, 
) -> MissionSpec:
    """
    3-point turn with configurable turn angle.

    Convention:
      - NED frame: +N forward, +E right, +D down
      - Heading angle psi is measured clockwise from North toward East (deg).

    Geometry:
        P1 = (settle_m, 0, -alt): velocity/control settle pre-roll (leg1_m) segment
            Why different from TO position? - from TO position to first waypoint, we cannot control flight speed.
        P2 = P1 + leg_m * [cos(0), sin(0)] = (settle_m + leg_m, 0, -alt) (north leg)
        P3 = P2 + leg2_m * [cos(turn), sin(turn)] (turned leg)

    Mission layout:
      TAKEOFF
      WP P1 (settle) : speed uncontrolled
      DO_CHANGE_SPEED(speed_m_s) : applies for subsequent legs
      WP P2
      WP P3
      RTL or LAND
    """

    if not home_position:
        raise ValueError("home_position must be non-empty")
    
    d = -float(alt_m)
    settle = float(settle_m)
    l1 = float(leg1_m)
    l2 = float(leg2_m)

    dN1 = l1 * np.cos(0)
    dE1 = l1 * np.sin(0)
    psi = np.radians(float(turn_deg))
    dN2 = l2 * np.cos(psi)
    dE2 = l2 * np.sin(psi)

    P1: NED = (settle, 0.0, d)
    P2: NED = (P1[0] + dN1, P1[1] + dE1, d)
    P3: NED = (P2[0] + dN2, P2[1] + dE2, d)

    home = home_position

    # ------------------------------------------------------------------
    # Build aligned mission-item lists (same length)
    # ------------------------------------------------------------------
    commands: List[int] = [
        MAV_CMD_NAV_TAKEOFF,      # idx 0
        MAV_CMD_NAV_WAYPOINT,     # idx 1: P1
        MAV_CMD_DO_CHANGE_SPEED,  # idx 2
        MAV_CMD_NAV_WAYPOINT,     # idx 3: P2
        MAV_CMD_NAV_WAYPOINT,     # idx 4: P3
    ]
    if land:
        commands.append(MAV_CMD_NAV_LAND)  # idx 5

    waypoints_ned: List[Optional[NED]] = [
        None,   # TAKEOFF (we keep it None to avoid forcing fake points)
        P1,     # WP
        None,   # DO_CHANGE_SPEED
        P2,     # WP
        P3,     # WP
    ]
    if land:
        waypoints_ned.append(None)  # LAND: you may choose to land at home or last wp; keep None here

    waypoints_lla: List[Optional[LLA]] = [
        None,
        ned_to_lla(P1, home),
        None,
        ned_to_lla(P2, home),
        ned_to_lla(P3, home),
    ]
    if land:
        waypoints_lla.append(None)

    speeds: List[Optional[float]] = [
        None,        # TAKEOFF
        None,        # WP P1
        speed_m_s,   # DO_CHANGE_SPEED
        None,        # WP P2
        None,        # WP P3
    ]
    if land:
        speeds.append(None)

    # sanity check: all aligned
    assert len(commands) == len(waypoints_ned) == len(waypoints_lla) == len(speeds)

    return MissionSpec(
        name=f"turn3_settle_{int(round(turn_deg))}deg",
        home_position=home_position,
        command=commands,
        takeoff_alt_m=alt_m,
        waypoints_ned=waypoints_ned,
        waypoints_lla=waypoints_lla,
        speed_m_s=speeds,
        land=land,
    )

# ----------------------------------------------------------------------
# Pattern 2: 4-Point Square
# ----------------------------------------------------------------------
def make_quad_4pts(
    home_position: List[LLA],
    side1_m: float,
    side2_m: float,
    angle_deg: float,
    alt_m: float,
    speed_m_s: float,
    settle_m: float = 10.0,
    land: bool = True,
) -> MissionSpec:
    """
    4-point quadrilateral mission.

    This generalizes:
      - square      : side1_m == side2_m, angle_deg = 90
      - rectangle   : side1_m != side2_m, angle_deg = 90
      - parallelogram / skewed box : angle_deg != 90

    Convention:
      - NED frame: +N forward, +E right, +D down
      - angle_deg is measured clockwise from the first leg direction
        toward East.

    Geometry:
        P0 = (settle_m, 0, -alt)
        P1 = P0 + side1 along North
        P2 = P1 + side2 at angle_deg
        P3 = P0 + side2 at angle_deg

    Mission layout:
      TAKEOFF
      WP P0
      DO_CHANGE_SPEED
      WP P1
      WP P2
      WP P3
      WP P0
      LAND
    """

    if not home_position:
        raise ValueError("home_position must be non-empty")

    d = -float(alt_m)
    settle = float(settle_m)
    s1 = float(side1_m)
    s2 = float(side2_m)

    theta = np.radians(float(angle_deg))

    # First edge direction: North
    v1_n = s1
    v1_e = 0.0

    # Second edge direction: angle from North toward East
    v2_n = s2 * np.cos(theta)
    v2_e = s2 * np.sin(theta)

    P0: NED = (settle, 0.0, d)
    P1: NED = (P0[0] + v1_n, P0[1] + v1_e, d)
    P2: NED = (P1[0] + v2_n, P1[1] + v2_e, d)
    P3: NED = (P0[0] + v2_n, P0[1] + v2_e, d)

    commands: List[int] = [
        MAV_CMD_NAV_TAKEOFF,      # idx 0
        MAV_CMD_NAV_WAYPOINT,     # idx 1: P0 settle point
        MAV_CMD_DO_CHANGE_SPEED,  # idx 2
        MAV_CMD_NAV_WAYPOINT,     # idx 3: P1
        MAV_CMD_NAV_WAYPOINT,     # idx 4: P2
        MAV_CMD_NAV_WAYPOINT,     # idx 5: P3
        MAV_CMD_NAV_WAYPOINT,     # idx 6: back to P0
    ]
    if land:
        commands.append(MAV_CMD_NAV_LAND)

    waypoints_ned: List[Optional[NED]] = [
        None,
        P0,
        None,
        P1,
        P2,
        P3,
        P0,
    ]
    if land:
        waypoints_ned.append(None)

    waypoints_lla: List[Optional[LLA]] = [
        None,
        ned_to_lla(P0, home_position),
        None,
        ned_to_lla(P1, home_position),
        ned_to_lla(P2, home_position),
        ned_to_lla(P3, home_position),
        ned_to_lla(P0, home_position),
    ]
    if land:
        waypoints_lla.append(None)

    speeds: List[Optional[float]] = [
        None,
        None,
        float(speed_m_s),
        None,
        None,
        None,
        None,
    ]
    if land:
        speeds.append(None)

    assert len(commands) == len(waypoints_ned) == len(waypoints_lla) == len(speeds)

    if abs(side1_m - side2_m) < 1e-6 and abs(angle_deg - 90.0) < 1e-6:
        name = "square4"
    elif abs(angle_deg - 90.0) < 1e-6:
        name = "rectangle4"
    else:
        name = f"quad4_{int(round(angle_deg))}deg"

    return MissionSpec(
        name=name,
        home_position=home_position,
        command=commands,
        takeoff_alt_m=alt_m,
        waypoints_ned=waypoints_ned,
        waypoints_lla=waypoints_lla,
        speed_m_s=speeds,
        land=land,
    )


# ----------------------------------------------------------------------
# YAML Writer
# ----------------------------------------------------------------------
def write_scenario_yaml(
    run_dir: Path,
    mission: MissionSpec,
    *,
    ardupilot_dir: str,
    ardupilot_vehicle: str,
    ardupilot_frame: str,
    ardupilot_model: str,
    ardupilot_world: str,
    ardupilot_location: str,
    ardupilot_mavproxy_outport: int,
    ardupilot_connect_port: int,
    px4_dir: str,
    px4_vehicle: int,
    px4_frame: str,
    px4_world: str,
    px4_location: str,
    px4_qgc_outport: int,
    px4_connect_port: int,
) -> None:
    """
    Write scenario.yaml for one run directory.

    MissionSpec alignment:
      len(command) == len(waypoints_ned) == len(waypoints_lla) == len(speed_m_s)

    For commands without position/speed, entries are None -> YAML null.
    """

    run_id = int(run_dir.name.split("_")[-1])

    # Pick home position for this run (if multiple homes are provided)
    if not mission.home_position:
        raise ValueError("MissionSpec.home_position must be a non-empty list")

    home_idx = min(run_id, len(mission.home_position) - 1)
    home = mission.home_position  # [lat, lon, alt]

    def ned_to_yaml(wp: Optional[tuple[float, float, float]]):
        if wp is None:
            return None
        n, e, d = wp
        return [float(n), float(e), float(d)]

    def lla_to_yaml(wp: Optional[tuple[float, float, float]]):
        if wp is None:
            return None
        lat, lon, alt = wp
        return [float(lat), float(lon), float(alt)]

    def spd_to_yaml(v: Optional[float]):
        return None if v is None else float(v)

    # Optional: sanity check alignment
    L = len(mission.command)
    if not (len(mission.waypoints_ned) == len(mission.waypoints_lla) == len(mission.speed_m_s) == L):
        raise ValueError(
            "MissionSpec lists must be aligned: "
            "len(command)=len(waypoints_ned)=len(waypoints_lla)=len(speed_m_s)"
        )

    scenario = {
        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------
        "meta": {
            "run_id": run_id,
        },
        # --------------------------------------------------------------
        # Common configuration (shared mission geometry)
        # --------------------------------------------------------------
        "common": {
            "scenario": {
                "name": mission.name,
                "home_lla": [float(home[0]), float(home[1]), float(home[2])],
                "takeoff_alt_m": float(mission.takeoff_alt_m),
                "altitude_mode": 1,
                "land": bool(mission.land),
                "command": [int(c) for c in mission.command],
                "speed_m_s": [spd_to_yaml(v) for v in mission.speed_m_s],
                "waypoints_ned": [ned_to_yaml(wp) for wp in mission.waypoints_ned],
                "waypoints_lla": [lla_to_yaml(wp) for wp in mission.waypoints_lla],
            }
        },
        # --------------------------------------------------------------
        # Autopilot-specific simulation settings
        # --------------------------------------------------------------
        "autopilots": {
            "ardupilot": {
                "sim": {
                    "ardupilot_dir": ardupilot_dir,
                    "instance": 0,
                    "vehicle": ardupilot_vehicle,
                    "frame": ardupilot_frame,
                    "model": ardupilot_model,
                    "world": ardupilot_world,
                    "location": ardupilot_location,
                    "mavproxy_outport": int(ardupilot_mavproxy_outport),
                },
                "mavlink": {
                    "connect_url": f"udp:127.0.0.1:{int(ardupilot_connect_port)}"
                },
            },
            "px4": {
                "sim": {
                    "px4_dir": px4_dir,
                    "instance": 0,
                    "vehicle": int(px4_vehicle),
                    "frame": px4_frame,
                    "world": px4_world,
                    "location": px4_location,
                    "qgc_outport": int(px4_qgc_outport),
                },
                "mavlink": {
                    "connect_url": f"udp:127.0.0.1:{int(px4_connect_port)}"
                },
            },
        },
    }

    (run_dir / "scenario.yaml").write_text(
        yaml.safe_dump(scenario, sort_keys=False),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------
# Main Entry
# ----------------------------------------------------------------------
def main() -> int:
    """
    CLI entry point.

    Responsibilities:
        1. Parse arguments
        2. Build mission geometry
        3. Create run directories
        4. Generate scenario.yaml for each run
    """

    # --------------------------------------------------------------
    # Common arguments
    # --------------------------------------------------------------
    common_parser  = argparse.ArgumentParser(add_help=False)

    # Output settings
    common_parser.add_argument("--outdir", type=Path, default=Path("./data/sitl_logs"))
    common_parser.add_argument("--runs", type=int, default=1)
    common_parser.add_argument(
        "--home-lla",
        type=float,
        nargs=3,
        metavar=("LAT", "LON", "ALT"),
        default=(40.41176161953683, -86.93352081596879, 0.0),
    )

    # ArduPilot defaults
    common_parser.add_argument(
        "--ardupilot-dir",
        type=str,
        default="./ap/ardupilot",
    )
    common_parser.add_argument("--ardupilot-vehicle", type=str, default="ArduCopter")
    common_parser.add_argument("--ardupilot-frame", type=str, default="gazebo-iris")
    common_parser.add_argument("--ardupilot-model", type=str, default="JSON")
    common_parser.add_argument("--ardupilot-world", type=str, default="iris_runway")
    common_parser.add_argument("--ardupilot-location", type=str, default="Purdue")
    common_parser.add_argument("--ardupilot-mavproxy-outport", type=int, default=14551)
    common_parser.add_argument("--ardupilot-connect-port", type=int, default=14550)

    # PX4 defaults
    common_parser.add_argument(
        "--px4-dir",
        type=str,
        default="./ap/px4",
    )
    common_parser.add_argument("--px4-vehicle", type=int, default=4001)
    common_parser.add_argument("--px4-frame", type=str, default="gz_x500")
    common_parser.add_argument("--px4-world", type=str, default="default")
    common_parser.add_argument("--px4-location", type=str, default="Purdue")
    common_parser.add_argument("--px4-qgc-outport", type=int, default=14550)
    common_parser.add_argument("--px4-connect-port", type=int, default=14540)

    # --------------------------------------------------------------
    # Main parser + subparsers
    # --------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Scenario generator (pattern-specific)")
    subparsers = ap.add_subparsers(dest="pattern", required=True)

    # -------------------------
    # turn3pts
    # -------------------------
    turn3 = subparsers.add_parser(
        "turn3pts",
        parents=[common_parser],
    )

    def parse_turn_deg(v: str):
        if v.lower() == "random":
            return "random"
        try:
            return float(v)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                'turn-deg must be a float or "random"'
            ) from e
        
    def parse_alt_m(v: str):
        if v.lower() == "random":
            return "random"
        try:
            return float(v)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                'alt-m must be a float or "random"'
            ) from e
    
    turn3.add_argument("--settle-m", type=float, default=10.0)
    turn3.add_argument("--leg1-m", type=float, default=50.0)
    turn3.add_argument("--leg2-m", type=float, default=50.0)
    turn3.add_argument("--turn-deg", type=parse_turn_deg, default=90, help='Turn angle (deg) or "random"')
    turn3.add_argument('--alt-m', type=parse_alt_m, default=10, help='Altitude (m) or "random"')
    turn3.add_argument('--speed-m-s', type=float, default=6.0)
    turn3.add_argument('--land', type=bool, default=True, help="Whether to land at the end of the mission")

    # -------------------------
    # quad4: square / rectangle / skewed parallelogram
    # -------------------------
    quad4 = subparsers.add_parser(
        "quad4pts",
        parents=[common_parser],
    )

    def parse_random_float(v: str):
        if v.lower() == "random":
            return "random"
        try:
            return float(v)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                'value must be a float or "random"'
            ) from e

    quad4.add_argument("--settle-m", type=parse_random_float, default=10.0)
    quad4.add_argument("--side1-m", type=parse_random_float, default=50.0)
    quad4.add_argument("--side2-m", type=parse_random_float, default=50.0)
    quad4.add_argument("--angle-deg", type=parse_random_float, default=90.0)
    quad4.add_argument("--alt-m", type=parse_random_float, default=10.0)
    quad4.add_argument("--speed-m-s", type=parse_random_float, default=6.0)
    quad4.add_argument("--land", type=bool, default=True)


    # --------------------------------------------------------------
    # Parse CLI arguments
    # --------------------------------------------------------------
    args = ap.parse_args()

    # Creating output directory (if not exists)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Generate runs
    # --------------------------------------------------------------
    for i in range(args.runs):
        run_dir = args.outdir / f"run_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.pattern == "turn3pts":
            if args.turn_deg == "random":
                turn_deg = float(np.random.uniform(0.0, 360.0))
            else:
                turn_deg = float(args.turn_deg)

            if args.alt_m == "random":
                alt_m = float(np.random.uniform(0.1, 3.0))
            else:                
                alt_m = float(args.alt_m)  

            mission = make_turn_3pts(
                home_position=args.home_lla,
                settle_m=args.settle_m,
                leg1_m=args.leg1_m,
                leg2_m=args.leg2_m,
                turn_deg=turn_deg,
                alt_m=alt_m,
                speed_m_s=args.speed_m_s,
                land=args.land,
            )

        elif args.pattern == "quad4pts":
            settle_m = (
                float(np.random.uniform(5.0, 20.0))
                if args.settle_m == "random"
                else float(args.settle_m)
            )

            side1_m = (
                float(np.random.uniform(30.0, 120.0))
                if args.side1_m == "random"
                else float(args.side1_m)
            )

            side2_m = (
                float(np.random.uniform(30.0, 120.0))
                if args.side2_m == "random"
                else float(args.side2_m)
            )

            angle_deg = (
                float(np.random.uniform(45.0, 135.0))
                if args.angle_deg == "random"
                else float(args.angle_deg)
            )

            alt_m = (
                float(np.random.uniform(0.1, 3.0))
                if args.alt_m == "random"
                else float(args.alt_m)
            )

            speed_m_s = (
                float(np.random.uniform(0.5, 3.0))
                if args.speed_m_s == "random"
                else float(args.speed_m_s)
            )

            mission = make_quad_4pts(
                home_position=args.home_lla,
                settle_m=settle_m,
                side1_m=side1_m,
                side2_m=side2_m,
                angle_deg=angle_deg,
                alt_m=alt_m,
                speed_m_s=speed_m_s,
                land=args.land,
            )

        else:
            raise ValueError(f"Unsupported pattern: {args.pattern}")

        write_scenario_yaml(
            run_dir,
            mission,
            ardupilot_dir=args.ardupilot_dir,
            ardupilot_vehicle=args.ardupilot_vehicle,
            ardupilot_frame=args.ardupilot_frame,
            ardupilot_model=args.ardupilot_model,
            ardupilot_world=args.ardupilot_world,
            ardupilot_location=args.ardupilot_location,
            ardupilot_mavproxy_outport=args.ardupilot_mavproxy_outport,
            ardupilot_connect_port=args.ardupilot_connect_port,
            px4_dir=args.px4_dir,
            px4_vehicle=args.px4_vehicle,
            px4_frame=args.px4_frame,
            px4_world=args.px4_world,
            px4_location=args.px4_location,
            px4_qgc_outport=args.px4_qgc_outport,
            px4_connect_port=args.px4_connect_port,
        )

    print(
        f"Generated {args.runs} runs "
        f"with pattern='{args.pattern}' "
        f"under {args.outdir}"
    )
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())