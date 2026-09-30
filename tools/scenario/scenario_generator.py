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
RANDOM_SPEC = "random"

DEFAULT_EDGE_RANGE_M = (30.0, 120.0)
DEFAULT_VERTEX_RANGE_DEG = (0.0, 360.0)
DEFAULT_ALT_RANGE_M = (5.0, 50.0)
DEFAULT_SPEED_RANGE_M_S = (3.0, 12.0)
DEFAULT_LANDING_ALT_M = 5.0
DEFAULT_WIND_HORIZONTAL_MAGNITUDE_RANGE_M_S = (0.0, 10.0)
DEFAULT_WIND_HORIZONTAL_AZIMUTH_RANGE_DEG = (0.0, 360.0)
DEFAULT_WIND_VERTICAL_MAGNITUDE_RANGE_M_S = (0.0, 3.0)
DEFAULT_INERTIAL_SCALE_RANGE = (0.8, 1.2)

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


def _range_to_list(value_range: Tuple[float, float]) -> List[float]:
    return [float(value_range[0]), float(value_range[1])]


def _metadata_value(value, value_range: Tuple[float, float]):
    if value == RANDOM_SPEC:
        return {
            "mode": "random_uniform",
            "range": _range_to_list(value_range),
        }
    if isinstance(value, tuple):
        return [float(v) for v in value]
    return float(value)


def _sample_tuple_spec(value, count: int, value_range: Tuple[float, float], rng=None) -> Tuple[float, ...]:
    if value == RANDOM_SPEC:
        low, high = value_range
        return tuple(float(v) for v in (rng if rng is not None else np.random).uniform(low, high, size=count))
    return tuple(float(v) for v in value)


def _sample_float_spec(value, value_range: Tuple[float, float], rng=None) -> float:
    if value == RANDOM_SPEC:
        low, high = value_range
        return float((rng if rng is not None else np.random).uniform(low, high))
    return float(value)


def _sample_wind_direction_spec(value, rng=None) -> int:
    """Sample or normalize a vertical wind direction to either -1 or +1."""
    if value == RANDOM_SPEC:
        return int((rng if rng is not None else np.random).choice((-1, 1)))
    return int(value)


def wind_to_enu(
    horizontal_magnitude_m_s: float,
    horizontal_azimuth_deg: float,
    vertical_magnitude_m_s: float,
    vertical_direction: int,
) -> Tuple[float, float, float]:
    """Convert wind polar inputs to a Gazebo ENU velocity vector.

    Azimuth is the direction the velocity vector points toward, measured
    clockwise from North: North=0, East=90, South=180, West=270 degrees.
    Vertical direction +1 is up and -1 is down.
    """
    horizontal_magnitude_m_s = float(horizontal_magnitude_m_s)
    horizontal_azimuth_deg = float(horizontal_azimuth_deg)
    vertical_magnitude_m_s = float(vertical_magnitude_m_s)
    vertical_direction = int(vertical_direction)

    if horizontal_magnitude_m_s < 0.0:
        raise ValueError("horizontal wind magnitude must be non-negative")
    if not 0.0 <= horizontal_azimuth_deg <= 360.0:
        raise ValueError("horizontal wind azimuth must be in [0, 360]")
    if vertical_magnitude_m_s < 0.0:
        raise ValueError("vertical wind magnitude must be non-negative")
    if vertical_direction not in (-1, 1):
        raise ValueError("vertical wind direction must be -1 or +1")

    azimuth_rad = np.radians(horizontal_azimuth_deg)
    east = horizontal_magnitude_m_s * np.sin(azimuth_rad)
    north = horizontal_magnitude_m_s * np.cos(azimuth_rad)
    up = vertical_magnitude_m_s * vertical_direction

    # Avoid tiny floating-point artifacts such as cos(90 deg) ~= 6e-17.
    return tuple(0.0 if abs(v) < 1e-12 else float(v) for v in (east, north, up))


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
    home_position: LLA          # home position (lat, lon, alt)
    command: List[int]          # list of MAVLink command IDs for each waypoint
    takeoff_alt_m: float        # takeoff altitude (positive above home)
    landing_alt_m: float        # final approach altitude above home before LAND
    waypoints_ned: List[Optional[NED]]      # aligned with command list (None if no position)
    waypoints_lla: List[Optional[LLA]]      # aligned with command list (None if no position)
    speed_m_s: List[Optional[SPD]]          # aligned with command list (None if unused)
    land: bool = True


def _append_home_approach_and_land(
    commands: List[int],
    waypoints_ned: List[Optional[NED]],
    waypoints_lla: List[Optional[LLA]],
    speeds: List[Optional[SPD]],
    home_position: LLA,
    landing_alt_m: float,
) -> None:
    if float(landing_alt_m) < 0.0:
        raise ValueError(f"landing_alt_m must be non-negative, got {landing_alt_m}")

    approach_ned: NED = (0.0, 0.0, -float(landing_alt_m))
    approach_lla: LLA = (float(home_position[0]), float(home_position[1]), float(landing_alt_m))

    commands.extend([MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_LAND])
    waypoints_ned.extend([approach_ned, None])
    waypoints_lla.extend([approach_lla, None])
    speeds.extend([None, None])


# ----------------------------------------------------------------------
# Pattern: Planar N-Point Waypoint Set
# ----------------------------------------------------------------------
def make_planar_n_pts(
    home_position: LLA,
    n: int,
    edge_m: Tuple[float, ...],
    vertex_deg: Tuple[float, ...],
    speed_m_s: Tuple[float, ...],
    alt_m: float,
    land: bool = True,
) -> MissionSpec:
    """
    Build a planar waypoint set from edge lengths and absolute headings.

    Convention:
      - NED frame: +N forward, +E right, +D down
      - vertex_deg values are headings measured clockwise from North to East.
      - n includes the takeoff/home point, so n-1 edge definitions produce
        n-1 waypoint endpoints.

    Mission layout:
      TAKEOFF
      DO_CHANGE_SPEED(speed_0)
      WP P1
      ...
      DO_CHANGE_SPEED(speed_n-2)
      WP P(n-1)
      WP home approach at landing_alt_m
      LAND
    """

    if not home_position:
        raise ValueError("home_position must be non-empty")

    n = int(n)
    if n < 2:
        raise ValueError("n must be at least 2")

    expected = n - 1
    if len(edge_m) != expected:
        raise ValueError(f"edge_m must contain n-1 values ({expected}), got {len(edge_m)}")
    if len(vertex_deg) != expected:
        raise ValueError(f"vertex_deg must contain n-1 values ({expected}), got {len(vertex_deg)}")
    if len(speed_m_s) != expected:
        raise ValueError(f"speed_m_s must contain n-1 values ({expected}), got {len(speed_m_s)}")

    for angle in vertex_deg:
        if not 0.0 <= float(angle) <= 360.0:
            raise ValueError(f"vertex_deg values must be in [0, 360], got {angle}")

    d = -float(alt_m)
    cur_n = 0.0
    cur_e = 0.0
    planar_points: List[NED] = []

    for length, heading_deg in zip(edge_m, vertex_deg):
        theta = np.radians(float(heading_deg))
        cur_n += float(length) * np.cos(theta)
        cur_e += float(length) * np.sin(theta)
        planar_points.append((float(cur_n), float(cur_e), d))

    commands: List[int] = [MAV_CMD_NAV_TAKEOFF]
    waypoints_ned: List[Optional[NED]] = [None]
    waypoints_lla: List[Optional[LLA]] = [None]
    speeds: List[Optional[SPD]] = [None]

    for point, speed in zip(planar_points, speed_m_s):
        commands.extend([MAV_CMD_DO_CHANGE_SPEED, MAV_CMD_NAV_WAYPOINT])
        waypoints_ned.extend([None, point])
        waypoints_lla.extend([None, ned_to_lla(point, home_position)])
        speeds.extend([float(speed), None])

    if land:
        _append_home_approach_and_land(
            commands,
            waypoints_ned,
            waypoints_lla,
            speeds,
            home_position,
            alt_m,
        )

    assert len(commands) == len(waypoints_ned) == len(waypoints_lla) == len(speeds)

    return MissionSpec(
        name=f"planar{n}pts",
        home_position=home_position,
        command=commands,
        takeoff_alt_m=alt_m,
        landing_alt_m=alt_m,
        waypoints_ned=waypoints_ned,
        waypoints_lla=waypoints_lla,
        speed_m_s=speeds,
        land=land,
    )


# ----------------------------------------------------------------------
# Pattern: 3D N-Point Waypoint Set
# ----------------------------------------------------------------------
def make_three_d_n_pts(
    home_position: LLA,
    n: int,
    edge_m: Tuple[float, ...],
    vertex_deg: Tuple[float, ...],
    alt_m: Tuple[float, ...],
    speed_m_s: Tuple[float, ...],
    takeoff_alt_m: float,
    landing_alt_m: float = DEFAULT_LANDING_ALT_M,
    land: bool = True,
) -> MissionSpec:
    """
    Build a 3D waypoint set from planar edges and per-waypoint altitude.

    edge_m and vertex_deg define the horizontal path. alt_m contains positive
    altitude above home for each waypoint and is converted internally to NED
    down by negating it.
    """

    if not home_position:
        raise ValueError("home_position must be non-empty")

    n = int(n)
    if n < 2:
        raise ValueError("n must be at least 2")

    expected = n - 1
    if len(edge_m) != expected:
        raise ValueError(f"edge_m must contain n-1 values ({expected}), got {len(edge_m)}")
    if len(vertex_deg) != expected:
        raise ValueError(f"vertex_deg must contain n-1 values ({expected}), got {len(vertex_deg)}")
    if len(alt_m) != expected:
        raise ValueError(f"alt_m must contain n-1 values ({expected}), got {len(alt_m)}")
    if len(speed_m_s) != expected:
        raise ValueError(f"speed_m_s must contain n-1 values ({expected}), got {len(speed_m_s)}")

    for angle in vertex_deg:
        if not 0.0 <= float(angle) <= 360.0:
            raise ValueError(f"vertex_deg values must be in [0, 360], got {angle}")
    for altitude in alt_m:
        if float(altitude) < 0.0:
            raise ValueError(f"alt_m values must be non-negative, got {altitude}")
    if float(takeoff_alt_m) < 0.0:
        raise ValueError(f"takeoff_alt_m must be non-negative, got {takeoff_alt_m}")
    if float(landing_alt_m) < 0.0:
        raise ValueError(f"landing_alt_m must be non-negative, got {landing_alt_m}")

    cur_n = 0.0
    cur_e = 0.0
    points: List[NED] = []

    for length, heading_deg, altitude in zip(edge_m, vertex_deg, alt_m):
        theta = np.radians(float(heading_deg))
        cur_n += float(length) * np.cos(theta)
        cur_e += float(length) * np.sin(theta)
        points.append((float(cur_n), float(cur_e), -float(altitude)))

    commands: List[int] = [MAV_CMD_NAV_TAKEOFF]
    waypoints_ned: List[Optional[NED]] = [None]
    waypoints_lla: List[Optional[LLA]] = [None]
    speeds: List[Optional[SPD]] = [None]

    for point, speed in zip(points, speed_m_s):
        commands.extend([MAV_CMD_DO_CHANGE_SPEED, MAV_CMD_NAV_WAYPOINT])
        waypoints_ned.extend([None, point])
        waypoints_lla.extend([None, ned_to_lla(point, home_position)])
        speeds.extend([float(speed), None])

    if land:
        _append_home_approach_and_land(
            commands,
            waypoints_ned,
            waypoints_lla,
            speeds,
            home_position,
            landing_alt_m,
        )

    assert len(commands) == len(waypoints_ned) == len(waypoints_lla) == len(speeds)

    return MissionSpec(
        name=f"three_d{n}pts",
        home_position=home_position,
        command=commands,
        takeoff_alt_m=float(takeoff_alt_m),
        landing_alt_m=float(landing_alt_m),
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
    wind_m_s: Tuple[float, float, float],
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
    vehicle_parameters: Optional[dict] = None,
    seed: Optional[int] = None,
) -> None:
    """
    Write scenario.yaml for one run directory.

    MissionSpec alignment:
      len(command) == len(waypoints_ned) == len(waypoints_lla) == len(speed_m_s)

    For commands without position/speed, entries are None -> YAML null.
    """

    run_id = int(run_dir.name.split("_")[-1])

    # Pick home position for this run
    if not mission.home_position:
        raise ValueError("MissionSpec.home_position must be non-empty")

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
                "landing_alt_m": float(mission.landing_alt_m),
                # Constant wind velocity in the Gazebo ENU frame: [east, north, up].
                "wind_m_s": [float(v) for v in wind_m_s],
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

    if vehicle_parameters is not None:
        scenario["common"]["vehicle"] = vehicle_parameters
    if seed is not None:
        scenario["meta"]["seed"] = seed

    (run_dir / "scenario.yaml").write_text(
        yaml.safe_dump(scenario, sort_keys=False),
        encoding="utf-8",
    )


def write_metadata_yaml(outdir: Path, args: argparse.Namespace) -> None:
    """
    Write dataset-level metadata.yaml at the root of the SITL log directory.
    """

    pattern_ranges = {
        "planar_n_pts": {
            "n": {"default": 3, "unit": "count"},
            "edge_m": {
                "default": [50.0, 50.0],
                "length": "n-1",
                "random_range": _range_to_list(DEFAULT_EDGE_RANGE_M),
                "unit": "m",
            },
            "vertex_deg": {
                "default": [0.0, 90.0],
                "length": "n-1",
                "random_range": _range_to_list(DEFAULT_VERTEX_RANGE_DEG),
                "unit": "deg",
            },
            "speed_m_s": {
                "default": [6.0, 6.0],
                "length": "n-1",
                "random_range": _range_to_list(DEFAULT_SPEED_RANGE_M_S),
                "unit": "m/s",
            },
            "alt_m": {
                "default": 10.0,
                "random_range": _range_to_list(DEFAULT_ALT_RANGE_M),
                "unit": "m",
            },
            "landing_alt_m": {
                "default": DEFAULT_LANDING_ALT_M,
                "unit": "m",
            },
        },
        "three_d_n_pts": {
            "n": {"default": 3, "unit": "count"},
            "edge_m": {
                "default": [50.0, 50.0],
                "length": "n-1",
                "random_range": _range_to_list(DEFAULT_EDGE_RANGE_M),
                "unit": "m",
            },
            "vertex_deg": {
                "default": [0.0, 90.0],
                "length": "n-1",
                "random_range": _range_to_list(DEFAULT_VERTEX_RANGE_DEG),
                "unit": "deg",
            },
            "alt_m": {
                "default": [10.0, 10.0],
                "length": "n-1",
                "random_range": _range_to_list(DEFAULT_ALT_RANGE_M),
                "unit": "m",
            },
            "takeoff_alt_m": {
                "default": 10.0,
                "random_range": _range_to_list(DEFAULT_ALT_RANGE_M),
                "unit": "m",
            },
            "landing_alt_m": {
                "default": DEFAULT_LANDING_ALT_M,
                "unit": "m",
            },
            "speed_m_s": {
                "default": [6.0, 6.0],
                "length": "n-1",
                "random_range": _range_to_list(DEFAULT_SPEED_RANGE_M_S),
                "unit": "m/s",
            },
        },
    }

    current_pattern_parameters = {
        "n": args.n,
        "edge_m": _metadata_value(args.edge_m, args.edge_m_range),
        "edge_m_range": _range_to_list(args.edge_m_range),
        "vertex_deg": _metadata_value(args.vertex_deg, args.vertex_deg_range),
        "vertex_deg_range": _range_to_list(args.vertex_deg_range),
        "alt_m": _metadata_value(args.alt_m, args.alt_m_range),
        "alt_m_range": _range_to_list(args.alt_m_range),
        "speed_m_s": _metadata_value(args.speed_m_s, args.speed_m_s_range),
        "speed_m_s_range": _range_to_list(args.speed_m_s_range),
        "landing_alt_m": float(args.landing_alt_m),
        "land": args.land,
    }
    if args.pattern == "three_d_n_pts":
        current_pattern_parameters["takeoff_alt_m"] = _metadata_value(
            args.takeoff_alt_m,
            args.takeoff_alt_m_range,
        )
        current_pattern_parameters["takeoff_alt_m_range"] = _range_to_list(
            args.takeoff_alt_m_range
        )

    metadata = {
        "dataset": {
            "name": "sitl_logs",
            "description": (
                "SITL log dataset generated for comparing PX4 and ArduPilot "
                "under the same mission/trajectory setting."
            ),
            "num_runs": int(args.runs),
            "start_run_id": int(args.start_run_id),
            "end_run_id": int(args.start_run_id + args.runs - 1),
            "selected_pattern": args.pattern,
            "seed": args.seed,
            "random_streams": "SeedSequence(seed, spawn_key=(run_id, stream)); mission=0, wind=1, mass=2, inertia=3",
        },
        "patterns": {
            "selected": {
                "name": args.pattern,
                "parameters": current_pattern_parameters,
            },
            "available": pattern_ranges,
        },
        "simulation": {
            "environment": "Gazebo",
            "home_lla": [float(v) for v in args.home_lla],
            "wind": {
                "type": "constant",
                "frame": "ENU",
                "vector_order": ["east", "north", "up"],
                "horizontal_magnitude_m_s": _metadata_value(
                    args.wind_horizontal_magnitude_m_s,
                    args.wind_horizontal_magnitude_m_s_range,
                ),
                "horizontal_azimuth_deg": _metadata_value(
                    args.wind_horizontal_azimuth_deg,
                    args.wind_horizontal_azimuth_deg_range,
                ),
                "vertical_magnitude_m_s": _metadata_value(
                    args.wind_vertical_magnitude_m_s,
                    args.wind_vertical_magnitude_m_s_range,
                ),
                "vertical_direction": (
                    {"mode": "random_choice", "values": [-1, 1]}
                    if args.wind_vertical_direction == RANDOM_SPEC
                    else int(args.wind_vertical_direction)
                ),
            },
        },
        "flight_stacks": {
            "px4": {
                "firmware": "PX4",
                "vehicle_model": args.px4_frame.removeprefix("gz_"),
                "frame": args.px4_frame,
                "world": args.px4_world,
                "location": args.px4_location,
                "parameters": "default",
                "log_format": ".ulg",
            },
            "ardupilot": {
                "firmware": "ArduPilot",
                "vehicle": args.ardupilot_vehicle,
                "vehicle_model": args.ardupilot_frame.removeprefix("gazebo-"),
                "frame": args.ardupilot_frame,
                "model": args.ardupilot_model,
                "world": args.ardupilot_world,
                "location": args.ardupilot_location,
                "parameters": "default",
                "log_format": ".BIN",
            },
        },
        "notes": [
            "metadata.yaml stores dataset-level information shared across all runs.",
            "Each run directory contains a scenario.yaml file.",
            "scenario.yaml stores run-specific mission geometry and autopilot launch settings.",
            "When a parameter uses random_uniform, sampled values are stored in each run_XXX/scenario.yaml.",
        ],
    }

    if args.vehicle_model is not None:
        metadata["simulation"]["vehicle"] = {
            "model_name": args.vehicle_model,
            "inertial": {
                "mode": "scale", "target_link": "base_link",
                "mass_scale": _metadata_value(args.base_mass_scale, args.base_mass_scale_range),
                "inertia_scale": _metadata_value(args.base_inertia_scale, args.base_inertia_scale_range),
                "sampling": "independent_uniform",
            },
        }
        metadata["flight_stacks"]["ardupilot"]["vehicle_model"] = args.vehicle_model
        metadata["flight_stacks"]["px4"]["vehicle_model"] = args.vehicle_model

    (outdir / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False),
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
    common_parser.add_argument("--seed", type=int, help="non-negative seed; generated and recorded when omitted")
    common_parser.add_argument(
        "--vehicle-model", choices=("fire_px4vision",),
        help="enable per-run inertial SDFs (also enabled by either base scale option)",
    )
    for quantity in ("mass", "inertia"):
        common_parser.add_argument(
            f"--base-{quantity}-scale", type=str,
            help=f'base_link {quantity} multiplier (default 1.0), or "random"; rotors are unchanged',
        )
        common_parser.add_argument(
            f"--base-{quantity}-scale-range", type=float, nargs=2,
            default=DEFAULT_INERTIAL_SCALE_RANGE, metavar=("MIN", "MAX"),
            help=f"positive uniform range used when base {quantity} scale is random (default: 0.8 1.2)",
        )
    common_parser.add_argument(
        "--start-run-id",
        type=int,
        default=0,
        help="first run directory id, e.g. 10 creates run_010",
    )
    common_parser.add_argument(
        "--home-lla",
        type=float,
        nargs=3,
        metavar=("LAT", "LON", "ALT"),
        default=(40.41176161953683, -86.93352081596879, 0.0),
    )

    # Constant wind in Gazebo's ENU world frame. Azimuth is the direction the
    # vector points toward, clockwise from North (North=0, East=90 degrees).
    common_parser.add_argument(
        "--wind-horizontal-magnitude-m-s",
        type=str,
        default="0.0",
        help='horizontal wind magnitude in m/s, or "random"',
    )
    common_parser.add_argument(
        "--wind-horizontal-magnitude-m-s-range",
        type=float,
        nargs=2,
        default=DEFAULT_WIND_HORIZONTAL_MAGNITUDE_RANGE_M_S,
        metavar=("MIN", "MAX"),
        help="uniform range used when horizontal wind magnitude is random",
    )
    common_parser.add_argument(
        "--wind-horizontal-azimuth-deg",
        type=str,
        default="0.0",
        help='wind-vector azimuth in degrees (North=0, East=90), or "random"',
    )
    common_parser.add_argument(
        "--wind-horizontal-azimuth-deg-range",
        type=float,
        nargs=2,
        default=DEFAULT_WIND_HORIZONTAL_AZIMUTH_RANGE_DEG,
        metavar=("MIN", "MAX"),
        help="uniform range used when horizontal wind azimuth is random",
    )
    common_parser.add_argument(
        "--wind-vertical-magnitude-m-s",
        type=str,
        default="0.0",
        help='vertical wind magnitude in m/s, or "random"',
    )
    common_parser.add_argument(
        "--wind-vertical-magnitude-m-s-range",
        type=float,
        nargs=2,
        default=DEFAULT_WIND_VERTICAL_MAGNITUDE_RANGE_M_S,
        metavar=("MIN", "MAX"),
        help="uniform range used when vertical wind magnitude is random",
    )
    common_parser.add_argument(
        "--wind-vertical-direction",
        type=str,
        choices=("-1", "1", "+1", RANDOM_SPEC),
        default="1",
        help='vertical direction: 1=up, -1=down, or "random"',
    )

    # ArduPilot defaults
    common_parser.add_argument(
        "--ardupilot-dir",
        type=str,
        default="./ap/ardupilot",
    )
    common_parser.add_argument("--ardupilot-vehicle", type=str, default="ArduCopter")
    common_parser.add_argument("--ardupilot-frame", type=str, help="default: gazebo-px4vision with inertial scaling, otherwise gazebo-iris")
    common_parser.add_argument("--ardupilot-model", type=str, default="JSON")
    common_parser.add_argument("--ardupilot-world", type=str, help="default: default_fire_px4vision with inertial scaling, otherwise iris_runway")
    common_parser.add_argument("--ardupilot-location", type=str, default="Purdue")
    common_parser.add_argument("--ardupilot-mavproxy-outport", type=int, default=14551)
    common_parser.add_argument("--ardupilot-connect-port", type=int, default=14550)

    # PX4 defaults
    common_parser.add_argument(
        "--px4-dir",
        type=str,
        default="./ap/px4",
    )
    common_parser.add_argument("--px4-vehicle", type=int, help="default: 4006 with inertial scaling, otherwise 4001")
    common_parser.add_argument("--px4-frame", type=str, help="default: gz_fire_px4vision with inertial scaling, otherwise gz_x500")
    common_parser.add_argument("--px4-world", type=str, help="default: default_fire with inertial scaling, otherwise default")
    common_parser.add_argument("--px4-location", type=str, default="Purdue")
    common_parser.add_argument("--px4-qgc-outport", type=int, default=14550)
    common_parser.add_argument("--px4-connect-port", type=int, default=14540)

    # --------------------------------------------------------------
    # Main parser + subparsers
    # --------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Scenario generator (pattern-specific)")
    subparsers = ap.add_subparsers(dest="pattern", required=True)

    def parse_bool(v: str) -> bool:
        lowered = v.lower()
        if lowered in ("1", "true", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "no", "n", "off"):
            return False
        raise argparse.ArgumentTypeError("expected a boolean value")

    def parse_float_list(values: List[str]) -> Tuple[float, ...]:
        if len(values) == 1 and values[0].lower() == RANDOM_SPEC:
            return RANDOM_SPEC

        parsed: List[float] = []
        for value in values:
            for item in value.split(","):
                item = item.strip()
                if not item:
                    continue
                try:
                    parsed.append(float(item))
                except ValueError as e:
                    raise argparse.ArgumentTypeError(
                        f'expected float values or "random", got {item!r}'
                    ) from e
        return tuple(parsed)

    def parse_float_or_random(value: str):
        if value.lower() == RANDOM_SPEC:
            return RANDOM_SPEC
        try:
            return float(value)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f'expected a float or "random", got {value!r}'
            ) from e

    def parse_float_range(values: List[float]) -> Tuple[float, float]:
        low, high = float(values[0]), float(values[1])
        if low > high:
            raise argparse.ArgumentTypeError("range min must be <= max")
        return (low, high)

    # -------------------------
    # planar_n_pts
    # -------------------------
    planar = subparsers.add_parser(
        "planar_n_pts",
        parents=[common_parser],
    )

    planar.add_argument("--n", type=int, default=3)
    planar.add_argument(
        "--edge-m",
        type=str,
        nargs="+",
        default=("50.0", "50.0"),
        help='n-1 edge lengths in meters, or "random"',
    )
    planar.add_argument(
        "--edge-m-range",
        type=float,
        nargs=2,
        default=DEFAULT_EDGE_RANGE_M,
        metavar=("MIN", "MAX"),
        help="uniform range used when --edge-m random",
    )
    planar.add_argument(
        "--vertex-deg",
        type=str,
        nargs="+",
        default=("0.0", "90.0"),
        help='n-1 absolute headings in degrees [0, 360], clockwise from North, or "random"',
    )
    planar.add_argument(
        "--vertex-deg-range",
        type=float,
        nargs=2,
        default=DEFAULT_VERTEX_RANGE_DEG,
        metavar=("MIN", "MAX"),
        help="uniform range used when --vertex-deg random",
    )
    planar.add_argument("--alt-m", type=parse_float_or_random, default=10.0)
    planar.add_argument(
        "--alt-m-range",
        type=float,
        nargs=2,
        default=DEFAULT_ALT_RANGE_M,
        metavar=("MIN", "MAX"),
        help="uniform range used when --alt-m random",
    )
    planar.add_argument(
        "--speed-m-s",
        type=str,
        nargs="+",
        default=("6.0", "6.0"),
        help='n-1 speeds in m/s, one DO_CHANGE_SPEED before each waypoint, or "random"',
    )
    planar.add_argument(
        "--speed-m-s-range",
        type=float,
        nargs=2,
        default=DEFAULT_SPEED_RANGE_M_S,
        metavar=("MIN", "MAX"),
        help="uniform range used when --speed-m-s random",
    )
    planar.add_argument("--land", type=parse_bool, default=True)
    planar.add_argument(
        "--landing-alt-m",
        type=float,
        default=DEFAULT_LANDING_ALT_M,
        help="home approach waypoint altitude above home before LAND",
    )

    # -------------------------
    # three_d_n_pts
    # -------------------------
    three_d = subparsers.add_parser(
        "three_d_n_pts",
        parents=[common_parser],
    )

    three_d.add_argument("--n", type=int, default=3)
    three_d.add_argument(
        "--edge-m",
        type=str,
        nargs="+",
        default=("50.0", "50.0"),
        help='n-1 edge lengths in meters, or "random"',
    )
    three_d.add_argument(
        "--edge-m-range",
        type=float,
        nargs=2,
        default=DEFAULT_EDGE_RANGE_M,
        metavar=("MIN", "MAX"),
        help="uniform range used when --edge-m random",
    )
    three_d.add_argument(
        "--vertex-deg",
        type=str,
        nargs="+",
        default=("0.0", "90.0"),
        help='n-1 absolute headings in degrees [0, 360], clockwise from North, or "random"',
    )
    three_d.add_argument(
        "--vertex-deg-range",
        type=float,
        nargs=2,
        default=DEFAULT_VERTEX_RANGE_DEG,
        metavar=("MIN", "MAX"),
        help="uniform range used when --vertex-deg random",
    )
    three_d.add_argument(
        "--alt-m",
        type=str,
        nargs="+",
        default=("10.0", "10.0"),
        help='n-1 waypoint altitudes above home in meters, or "random"',
    )
    three_d.add_argument(
        "--alt-m-range",
        type=float,
        nargs=2,
        default=DEFAULT_ALT_RANGE_M,
        metavar=("MIN", "MAX"),
        help="uniform range used when --alt-m random",
    )
    three_d.add_argument(
        "--takeoff-alt-m",
        type=parse_float_or_random,
        default=10.0,
        help='takeoff altitude above home in meters, or "random"',
    )
    three_d.add_argument(
        "--takeoff-alt-m-range",
        type=float,
        nargs=2,
        default=DEFAULT_ALT_RANGE_M,
        metavar=("MIN", "MAX"),
        help="uniform range used when --takeoff-alt-m random",
    )
    three_d.add_argument(
        "--speed-m-s",
        type=str,
        nargs="+",
        default=("6.0", "6.0"),
        help='n-1 speeds in m/s, one DO_CHANGE_SPEED before each waypoint, or "random"',
    )
    three_d.add_argument(
        "--speed-m-s-range",
        type=float,
        nargs=2,
        default=DEFAULT_SPEED_RANGE_M_S,
        metavar=("MIN", "MAX"),
        help="uniform range used when --speed-m-s random",
    )
    three_d.add_argument("--land", type=parse_bool, default=True)
    three_d.add_argument(
        "--landing-alt-m",
        type=float,
        default=DEFAULT_LANDING_ALT_M,
        help="home approach waypoint altitude above home before LAND",
    )


    # --------------------------------------------------------------
    # Parse CLI arguments
    # --------------------------------------------------------------
    args = ap.parse_args()

    scaling_enabled = (args.vehicle_model is not None or args.base_mass_scale is not None
                       or args.base_inertia_scale is not None)
    if scaling_enabled:
        args.vehicle_model = args.vehicle_model or "fire_px4vision"
    args.ardupilot_frame = args.ardupilot_frame or ("gazebo-px4vision" if scaling_enabled else "gazebo-iris")
    args.ardupilot_world = args.ardupilot_world or ("default_fire_px4vision" if scaling_enabled else "iris_runway")
    args.px4_vehicle = args.px4_vehicle if args.px4_vehicle is not None else (4006 if scaling_enabled else 4001)
    args.px4_frame = args.px4_frame or ("gz_fire_px4vision" if scaling_enabled else "gz_x500")
    args.px4_world = args.px4_world or ("default_fire" if scaling_enabled else "default")
    if args.seed is not None and args.seed < 0:
        ap.error("--seed must be non-negative")
    if args.seed is None:
        args.seed = int(np.random.SeedSequence().entropy)

    for quantity in ("mass", "inertia"):
        key = f"base_{quantity}_scale"
        try:
            value = parse_float_or_random(getattr(args, key) or "1.0")
            value_range = parse_float_range(list(getattr(args, f"{key}_range")))
        except argparse.ArgumentTypeError as exc:
            ap.error(f"--base-{quantity}-scale: {exc}")
        if value != RANDOM_SPEC and (not np.isfinite(value) or value <= 0):
            ap.error(f"--base-{quantity}-scale must be finite and greater than zero")
        if any(not np.isfinite(v) or v <= 0 for v in value_range):
            ap.error(f"--base-{quantity}-scale-range must contain finite values greater than zero")
        setattr(args, key, value)
        setattr(args, f"{key}_range", value_range)
    if scaling_enabled and args.px4_frame != f"gz_{args.vehicle_model}":
        ap.error("inertial scaling requires --px4-frame gz_fire_px4vision")
    if scaling_enabled and (args.ardupilot_frame != "gazebo-px4vision" or args.ardupilot_model != "JSON"):
        ap.error("inertial scaling requires --ardupilot-frame gazebo-px4vision --ardupilot-model JSON")

    try:
        args.wind_horizontal_magnitude_m_s = parse_float_or_random(
            args.wind_horizontal_magnitude_m_s
        )
        args.wind_horizontal_magnitude_m_s_range = parse_float_range(
            list(args.wind_horizontal_magnitude_m_s_range)
        )
        args.wind_horizontal_azimuth_deg = parse_float_or_random(
            args.wind_horizontal_azimuth_deg
        )
        args.wind_horizontal_azimuth_deg_range = parse_float_range(
            list(args.wind_horizontal_azimuth_deg_range)
        )
        args.wind_vertical_magnitude_m_s = parse_float_or_random(
            args.wind_vertical_magnitude_m_s
        )
        args.wind_vertical_magnitude_m_s_range = parse_float_range(
            list(args.wind_vertical_magnitude_m_s_range)
        )
        if args.wind_vertical_direction != RANDOM_SPEC:
            args.wind_vertical_direction = int(args.wind_vertical_direction)
    except argparse.ArgumentTypeError as e:
        ap.error(str(e))

    if (
        args.wind_horizontal_magnitude_m_s != RANDOM_SPEC
        and args.wind_horizontal_magnitude_m_s < 0.0
    ):
        ap.error("--wind-horizontal-magnitude-m-s must be non-negative")
    if args.wind_horizontal_magnitude_m_s_range[0] < 0.0:
        ap.error("--wind-horizontal-magnitude-m-s-range must be non-negative")
    if (
        args.wind_horizontal_azimuth_deg != RANDOM_SPEC
        and not 0.0 <= args.wind_horizontal_azimuth_deg <= 360.0
    ):
        ap.error("--wind-horizontal-azimuth-deg must be in [0, 360]")
    if (
        args.wind_horizontal_azimuth_deg_range[0] < 0.0
        or args.wind_horizontal_azimuth_deg_range[1] > 360.0
    ):
        ap.error("--wind-horizontal-azimuth-deg-range must be within [0, 360]")
    if (
        args.wind_vertical_magnitude_m_s != RANDOM_SPEC
        and args.wind_vertical_magnitude_m_s < 0.0
    ):
        ap.error("--wind-vertical-magnitude-m-s must be non-negative")
    if args.wind_vertical_magnitude_m_s_range[0] < 0.0:
        ap.error("--wind-vertical-magnitude-m-s-range must be non-negative")

    if args.pattern in ("planar_n_pts", "three_d_n_pts"):
        try:
            args.edge_m_range = parse_float_range(list(args.edge_m_range))
            args.vertex_deg_range = parse_float_range(list(args.vertex_deg_range))
            args.alt_m_range = parse_float_range(list(args.alt_m_range))
            args.speed_m_s_range = parse_float_range(list(args.speed_m_s_range))
            args.edge_m = parse_float_list(list(args.edge_m))
            args.vertex_deg = parse_float_list(list(args.vertex_deg))
            args.speed_m_s = parse_float_list(list(args.speed_m_s))
            if args.pattern == "three_d_n_pts":
                args.alt_m = parse_float_list(list(args.alt_m))
                args.takeoff_alt_m_range = parse_float_range(list(args.takeoff_alt_m_range))
        except argparse.ArgumentTypeError as e:
            ap.error(str(e))

        expected = args.n - 1
        if args.n < 2:
            ap.error(f"{args.pattern} requires --n >= 2")
        if args.runs < 1:
            ap.error("--runs must be at least 1")
        if args.start_run_id < 0:
            ap.error("--start-run-id must be non-negative")
        if args.edge_m != RANDOM_SPEC and len(args.edge_m) != expected:
            ap.error(f"{args.pattern} requires {expected} --edge-m values")
        if args.vertex_deg != RANDOM_SPEC and len(args.vertex_deg) != expected:
            ap.error(f"{args.pattern} requires {expected} --vertex-deg values")
        if args.speed_m_s != RANDOM_SPEC and len(args.speed_m_s) != expected:
            ap.error(f"{args.pattern} requires {expected} --speed-m-s values")
        if args.pattern == "three_d_n_pts" and args.alt_m != RANDOM_SPEC and len(args.alt_m) != expected:
            ap.error(f"{args.pattern} requires {expected} --alt-m values")
        if args.edge_m != RANDOM_SPEC and any(v < 0.0 for v in args.edge_m):
            ap.error("--edge-m values must be non-negative")
        if args.edge_m_range[0] < 0.0:
            ap.error("--edge-m-range values must be non-negative")
        if args.vertex_deg != RANDOM_SPEC and any(v < 0.0 or v > 360.0 for v in args.vertex_deg):
            ap.error("--vertex-deg values must be in [0, 360]")
        if args.vertex_deg == RANDOM_SPEC and (
            args.vertex_deg_range[0] < 0.0 or args.vertex_deg_range[1] > 360.0
        ):
            ap.error("--vertex-deg-range must be within [0, 360]")
        if args.speed_m_s != RANDOM_SPEC and any(v < 0.0 for v in args.speed_m_s):
            ap.error("--speed-m-s values must be non-negative")
        if args.speed_m_s_range[0] < 0.0:
            ap.error("--speed-m-s-range values must be non-negative")
        if args.pattern == "planar_n_pts" and args.alt_m != RANDOM_SPEC and args.alt_m < 0.0:
            ap.error("--alt-m must be non-negative")
        if args.pattern == "three_d_n_pts" and args.alt_m != RANDOM_SPEC and any(v < 0.0 for v in args.alt_m):
            ap.error("--alt-m values must be non-negative")
        if args.alt_m_range[0] < 0.0:
            ap.error("--alt-m-range values must be non-negative")
        if args.landing_alt_m < 0.0:
            ap.error("--landing-alt-m must be non-negative")
        if (
            args.pattern == "three_d_n_pts"
            and args.takeoff_alt_m != RANDOM_SPEC
            and args.takeoff_alt_m < 0.0
        ):
            ap.error("--takeoff-alt-m must be non-negative")
        if args.pattern == "three_d_n_pts" and args.takeoff_alt_m_range[0] < 0.0:
            ap.error("--takeoff-alt-m-range values must be non-negative")

    # Creating output directory (if not exists)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Parse CLI arguments
    # --------------------------------------------------------------
    write_metadata_yaml(args.outdir, args)

    # --------------------------------------------------------------
    # Generate runs
    # --------------------------------------------------------------
    for offset in range(args.runs):
        run_id = args.start_run_id + offset
        run_dir = args.outdir / f"run_{run_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Stable per-run streams keep mission/wind unchanged when inertial
        # sampling is enabled, and allow regenerating a run by its ID.
        mission_rng, wind_rng, mass_rng, inertia_rng = [
            np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(run_id, stream)))
            for stream in range(4)
        ]
        vehicle_parameters = None
        if scaling_enabled:
            vehicle_parameters = {
                "model_name": args.vehicle_model,
                "inertial": {
                    "mode": "scale", "target_link": "base_link",
                    "mass_scale": _sample_float_spec(args.base_mass_scale, args.base_mass_scale_range, mass_rng),
                    "inertia_scale": _sample_float_spec(args.base_inertia_scale, args.base_inertia_scale_range, inertia_rng),
                },
            }

        wind_horizontal_magnitude_m_s = _sample_float_spec(
            args.wind_horizontal_magnitude_m_s,
            args.wind_horizontal_magnitude_m_s_range,
            wind_rng,
        )
        wind_horizontal_azimuth_deg = _sample_float_spec(
            args.wind_horizontal_azimuth_deg,
            args.wind_horizontal_azimuth_deg_range,
            wind_rng,
        )
        wind_vertical_magnitude_m_s = _sample_float_spec(
            args.wind_vertical_magnitude_m_s,
            args.wind_vertical_magnitude_m_s_range,
            wind_rng,
        )
        wind_vertical_direction = _sample_wind_direction_spec(
            args.wind_vertical_direction, wind_rng
        )
        wind_m_s = wind_to_enu(
            wind_horizontal_magnitude_m_s,
            wind_horizontal_azimuth_deg,
            wind_vertical_magnitude_m_s,
            wind_vertical_direction,
        )

        if args.pattern == "planar_n_pts":
            edge_m = _sample_tuple_spec(args.edge_m, args.n - 1, args.edge_m_range, mission_rng)
            vertex_deg = _sample_tuple_spec(args.vertex_deg, args.n - 1, args.vertex_deg_range, mission_rng)
            alt_m = _sample_float_spec(args.alt_m, args.alt_m_range, mission_rng)
            speed_m_s = _sample_tuple_spec(args.speed_m_s, args.n - 1, args.speed_m_s_range, mission_rng)

            mission = make_planar_n_pts(
                home_position=args.home_lla,
                n=args.n,
                edge_m=edge_m,
                vertex_deg=vertex_deg,
                alt_m=alt_m,
                speed_m_s=speed_m_s,
                land=args.land,
            )
        elif args.pattern == "three_d_n_pts":
            edge_m = _sample_tuple_spec(args.edge_m, args.n - 1, args.edge_m_range, mission_rng)
            vertex_deg = _sample_tuple_spec(args.vertex_deg, args.n - 1, args.vertex_deg_range, mission_rng)
            alt_m = _sample_tuple_spec(args.alt_m, args.n - 1, args.alt_m_range, mission_rng)
            takeoff_alt_m = _sample_float_spec(args.takeoff_alt_m, args.takeoff_alt_m_range, mission_rng)
            speed_m_s = _sample_tuple_spec(args.speed_m_s, args.n - 1, args.speed_m_s_range, mission_rng)

            mission = make_three_d_n_pts(
                home_position=args.home_lla,
                n=args.n,
                edge_m=edge_m,
                vertex_deg=vertex_deg,
                alt_m=alt_m,
                takeoff_alt_m=takeoff_alt_m,
                landing_alt_m=args.landing_alt_m,
                speed_m_s=speed_m_s,
                land=args.land,
            )
        else:
            raise ValueError(f"Unsupported pattern: {args.pattern}")

        write_scenario_yaml(
            run_dir,
            mission,
            wind_m_s=wind_m_s,
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
            vehicle_parameters=vehicle_parameters,
            seed=args.seed,
        )

    print(
        f"Generated {args.runs} runs "
        f"with pattern='{args.pattern}' "
        f"from run_{args.start_run_id:03d} "
        f"to run_{args.start_run_id + args.runs - 1:03d} "
        f"under {args.outdir}"
    )
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
