#!/usr/bin/env python3
"""Import QGroundControl .plan missions into runnable FIRE SITL scenarios.

Supports the existing TAKEOFF/WAYPOINT/CHANGE_SPEED/LAND command model and
relative-to-home altitudes. Rebase the route onto the checkout's Purdue world;
keep the original plan and all normalization decisions with each run.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tools.commander.flight_parameters import speed_defaults, write_flight_parameter_file
from tools.launcher.startup_parameters import collect_boot_parameters
from tools.scenario.create_params_sets import create_params_sets
from tools.commander.fc_parameter_profiles import normalize_profiles, resolve_profiles
from tools.scenario.scenario_generator import MissionSpec, ned_to_lla, write_scenario_yaml


SOURCE_AUTOPILOTS = {3: "ardupilot", 12: "px4"}
SUPPORTED_COMMANDS = {16, 21, 22, 178}


@dataclass
class ImportedMission:
    mission: MissionSpec
    report: dict
    source_bytes: bytes
    flight_parameters: dict


def number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def coordinate(value, label):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must be [latitude, longitude, altitude]")
    lat, lon, alt = [number(v, label) for v in value]
    if not -89 < lat < 89 or not -180 <= lon <= 180:
        raise ValueError(f"{label}: invalid or polar geographic coordinates")
    return lat, lon, alt


def lla_to_ned(point, home):
    """Inverse of scenario_generator.ned_to_lla; point altitude is relative."""
    lat0 = math.radians(home[0])
    denom = math.sqrt(1 - 6.69437999014e-3 * math.sin(lat0) ** 2)
    meridian = 6378137.0 * (1 - 6.69437999014e-3) / denom ** 3
    prime_vertical = 6378137.0 / denom
    north = math.radians(point[0] - home[0]) * (meridian + home[2])
    east = math.radians(point[1] - home[1]) * (prime_vertical + home[2]) * math.cos(lat0)
    if math.hypot(north, east) > 20_000:
        raise ValueError("Mission exceeds the 20 km local-coordinate conversion range")
    return north, east, -point[2]


def simulator_home():
    """The current launchers use locations.txt, independently of scenario home."""
    locations = ROOT / "tools/launcher/locations.txt"
    home = None
    for line in locations.read_text().splitlines():
        if line.strip().startswith("Purdue="):
            values = [float(v) for v in line.split("=", 1)[1].split(",")]
            if len(values) != 4 or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid Purdue entry in locations.txt")
            home = coordinate(values[:3], "Purdue home")
            break
    if home is None:
        raise ValueError(f"Purdue location missing in {locations}")
    for name in ("default_fire", "default_fire_px4vision"):
        path = ROOT / "gz/FIRE_moonshot_gazebo/worlds" / f"{name}.sdf"
        spherical = ET.parse(path).find(".//world/spherical_coordinates")
        if spherical is None:
            raise ValueError(f"{path}: world geographic origin is missing")
        origin = tuple(float(spherical.findtext(key)) for key in
                       ("latitude_deg", "longitude_deg", "elevation"))
        if any(abs(a - b) > 1e-7 for a, b in zip(home, origin)):
            raise ValueError(f"{path}: world origin and Purdue location disagree")
    return home


def discover_plans(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".plan":
            raise ValueError("Use a .plan input. KML alone does not specify a complete executable mission.")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input does not exist: {path}")
    plans = sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".plan"),
                   key=lambda p: p.name)
    if not plans:
        raise ValueError(f"No .plan files in {path}; KML-only reconstruction is not implemented")
    return plans


def _invalid_json_constant(value):
    raise ValueError(f"Nonstandard JSON number {value}")


def read_plan(path, target_home, *, autopilots=None, for_replay=False):
    """Read mission geometry; standalone imports use hoverSpeed, replay defers to log snapshots."""
    path = Path(path)
    source_bytes = path.read_bytes()
    data = json.loads(source_bytes, parse_constant=_invalid_json_constant)
    if not isinstance(data, dict) or data.get("fileType") != "Plan" or data.get("version") != 1:
        raise ValueError(f"{path}: expected QGroundControl Plan version 1")
    m = data.get("mission")
    if not isinstance(m, dict) or m.get("version") != 2:
        raise ValueError(f"{path}: expected mission version 2")
    if m.get("vehicleType") != 2 or m.get("firmwareType") not in SOURCE_AUTOPILOTS:
        raise ValueError(f"{path}: only ArduPilot/PX4 quadrotor plans are supported")
    for section, fields in (("geoFence", ("circles", "polygons")), ("rallyPoints", ("points",))):
        value = data.get(section, {})
        if not isinstance(value, dict) or any(value.get(field) for field in fields):
            raise ValueError(f"{path}: nonempty {section} cannot be reproduced by this importer")
    source_home = coordinate(m.get("plannedHomePosition"), "plannedHomePosition")
    target_home = coordinate(list(target_home), "SITL home")
    source_autopilot = SOURCE_AUTOPILOTS[m["firmwareType"]]
    selected = list(autopilots) if autopilots is not None else [source_autopilot]
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(SOURCE_AUTOPILOTS.values()):
        raise ValueError("Select distinct supported autopilots")
    items = m.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: mission.items must be a nonempty list")

    commands, positions, local_positions, speeds, mapping, changes = [], [], [], [], [], []
    takeoff_alt = None
    for index, item in enumerate(items):
        context = f"{path.name}: item {index}"
        if not isinstance(item, dict) or item.get("type") != "SimpleItem":
            raise ValueError(f"{context}: only SimpleItem missions are supported")
        cmd = item.get("command")
        if isinstance(cmd, bool) or not isinstance(cmd, int) or cmd not in SUPPORTED_COMMANDS:
            raise ValueError(f"{context}: unsupported command {cmd}")
        if item.get("autoContinue") is not True:
            raise ValueError(f"{context}: autoContinue=false is unsupported")
        params = item.get("params")
        if not isinstance(params, list) or len(params) != 7:
            raise ValueError(f"{context}: params must contain seven values")
        commands.append(cmd)
        mapping.append(dict(scenario_index=index, source_item_index=index, do_jump_id=item.get("doJumpId")))
        if cmd == 178:
            if item.get("frame") != 2:
                raise ValueError(f"{context}: CHANGE_SPEED requires MAV_FRAME_MISSION (2)")
            values = [number(v, context) for v in params]
            if values[0] not in (0, 1) or values[2] != -1 or any(values[3:]):
                raise ValueError(f"{context}: only horizontal CHANGE_SPEED without throttle changes is supported")
            speed_defaults(values[1], "px4", source=context)  # shared range in m/s
            if values[0] == 0:
                changes.append(dict(item=index, field="speed_type", original=0, effective=1,
                                    reason="Existing quadrotor commanders issue ground-speed commands"))
            speeds.append(values[1]); positions.append(None); local_positions.append(None)
            continue
        if item.get("frame") != 3:
            raise ValueError(f"{context}: only relative-to-home MAV_FRAME_GLOBAL_RELATIVE_ALT (3) is supported")
        if "AltitudeMode" in item and item["AltitudeMode"] != 1:
            raise ValueError(f"{context}: AltitudeMode conflicts with relative altitude")
        point = coordinate(params[4:7], context)
        if point[2] < 0 or (cmd != 21 and point[2] == 0):
            raise ValueError(f"{context}: flight altitude must be positive; LAND must use zero")
        if "Altitude" in item and not math.isclose(number(item["Altitude"], context), point[2], abs_tol=1e-6):
            raise ValueError(f"{context}: Altitude and params[6] disagree")
        prefix = [number(v, context) for v in params[:3]]
        if params[3] is not None or prefix[0] != 0 or prefix[2] != 0:
            raise ValueError(f"{context}: hold/pitch, pass/land mode and explicit yaw cannot be preserved by current commanders")
        if prefix[1] not in ((0, 1) if cmd == 16 else (0,)):
            raise ValueError(f"{context}: unsupported acceptance radius/reserved parameter")
        local = lla_to_ned(point, source_home)
        speeds.append(None)
        if cmd == 16:
            local_positions.append(local)
            positions.append(ned_to_lla(local, target_home))
            if prefix[1] != 1:
                changes.append(dict(item=index, field="acceptance_radius_m", original=prefix[1], effective=1.0,
                                    reason="Existing commander forces 1 m; PX4 overrides NAV_ACC_RAD, ArduCopter ignores this field"))
        else:
            if cmd == 22:
                if index != 0:
                    raise ValueError(f"{context}: TAKEOFF must be the first and only takeoff")
                takeoff_alt = point[2]
            elif index != len(items) - 1 or point[2] != 0:
                raise ValueError(f"{context}: LAND must be last, at relative altitude zero")
            positions.append(None); local_positions.append(None)
            if math.hypot(*local[:2]) > 0.01:
                changes.append(dict(item=index, field="horizontal_position", original=list(point[:2]),
                                    source_home_offset_m=list(local[:2]), effective="SITL home",
                                    reason="Existing commander uses home for TAKEOFF/LAND"))
    if commands[0] != 22 or commands[-1] != 21 or 16 not in commands:
        raise ValueError(f"{path}: require TAKEOFF, at least one WAYPOINT, and final LAND")

    speed_source = "flight log at mission start" if for_replay else "mission.hoverSpeed"
    speed_m_s = None
    flight = {}
    if not for_replay:
        speed_m_s = number(m.get("hoverSpeed"), f"{path}: mission.hoverSpeed")
        flight = {ap: speed_defaults(speed_m_s, ap, source=speed_source) for ap in selected}
    mission = MissionSpec(name=f"real_flight_{path.stem}", home_position=target_home,
                          command=commands, takeoff_alt_m=takeoff_alt, landing_alt_m=0,
                          waypoints_ned=local_positions, waypoints_lla=positions, speed_m_s=speeds)
    siblings = [str(p.resolve()) for p in sorted(path.parent.iterdir())
                if p.is_file() and p.stem == path.stem and p.suffix.lower() == ".kml"]
    report = dict(source_file=str(path.resolve()), source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  source_autopilot=source_autopilot, autopilots=selected,
                  source_home_lla=list(source_home), target_home_lla=list(target_home),
                  coordinate_transform="source-home local NED rebased to SITL Purdue; no rotation or scale",
                  altitude_reference="relative to home (MAV_FRAME_GLOBAL_RELATIVE_ALT)",
                  command_count=len(commands), waypoint_count=commands.count(16),
                  item_mapping=mapping, normalizations=changes,
                  default_speed_m_s=speed_m_s, default_speed_source=speed_source,
                  source_speed_metadata={key: m.get(key) for key in ("hoverSpeed", "cruiseSpeed")},
                  flight_parameters=flight, kml_sidecars=siblings, kml_used=False,
                  execution_notes=["Existing ArduPilot runner performs GUIDED takeoff before AUTO",
                                   "Explicit CHANGE_SPEED items are retained; they may override boot speed",
                                   "No wind or mission-only settling holds are added",
                                   "Planned speed is not evidence of the actual flight's commanded speed"])
    return ImportedMission(mission, report, source_bytes, flight)


def write_run(imported, run_dir, *, profiles=None, seed=0):
    run_dir.mkdir()
    write_scenario_yaml(
        run_dir, imported.mission, wind_m_s=(0, 0, 0),
        ardupilot_dir=str(ROOT / "ap/ardupilot"), ardupilot_vehicle="ArduCopter",
        ardupilot_frame="gazebo-px4vision", ardupilot_model="JSON", ardupilot_world="default_fire_px4vision",
        ardupilot_location="Purdue", ardupilot_mavproxy_outport=14551, ardupilot_connect_port=14550,
        px4_dir=str(ROOT / "ap/px4"), px4_vehicle=4006, px4_frame="gz_fire_px4vision",
        px4_world="default_fire", px4_location="Purdue", px4_qgc_outport=14550, px4_connect_port=14540,
        controller_gains=resolve_profiles(profiles or {}, seed, int(run_dir.name.split("_")[-1])), seed=seed)
    path = run_dir / "scenario.yaml"
    data = yaml.safe_load(path.read_text())
    selected = imported.report["autopilots"]
    data["autopilots"] = {ap: data["autopilots"][ap] for ap in selected}
    for ap, config in imported.flight_parameters.items():
        data["autopilots"][ap]["flight_parameters"] = write_flight_parameter_file(config, ap, run_dir)
    data["meta"]["mission_import"] = dict(source="source.plan", source_sha256=imported.report["source_sha256"],
                                          report="import_report.json")
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    (run_dir / "source.plan").write_bytes(imported.source_bytes)
    (run_dir / "import_report.json").write_text(json.dumps(imported.report, indent=2, allow_nan=False) + "\n")
    for ap in selected:
        collect_boot_parameters(data, ap, run_dir)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    gain_source = parser.add_mutually_exclusive_group()
    gain_source.add_argument("--flight-log", type=Path, help="Replay a native BIN/ULog timeline from a confirmed ground mission start; audit excluded pre-mission flights; always extract gains and flight settings from the log; incomplete missions produce warnings")
    parser.add_argument("--plan", type=Path, nargs="+", required=True, help="Explicit .plan file(s); matching candidates with --flight-log, otherwise standalone missions; directories are not accepted")
    parser.add_argument("--analyze-only", action="store_true", help="Print all logged intervals and plan candidates without generating a run")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--mission-segment", help="Compare one PX4 MISSION segment, e.g. flight_000:segment_007; prepare observed position/velocity/yaw for an in-air start; exclude setup and cleanup from comparison")
    selection.add_argument("--flight-id", action="append", help="Select an analyzed armed interval, e.g. flight_000 (repeatable); replay begins at its first observed mission on confirmed ground")
    parser.add_argument("--ground-wait-s", type=float, help="Override inter-flight ground waits; default preserves recorded disarm-to-arm delays")
    parser.add_argument("--outdir", type=Path, required=True, help="New output directory; --flight-log writes scenario.yaml directly here; never overwrite")
    parser.add_argument("--autopilots", nargs="+", choices=("ardupilot", "px4"), help="Default: source firmware per plan")
    gain_source.add_argument("--controller-gain-profile", type=Path, help="Plan-only: optional existing gain profile; omission keeps firmware gains; cannot combine with --flight-log")
    parser.add_argument('--parameter-groups', nargs='+', help='Flight-log capture groups (or all); default: legacy controller groups. Missing logged values fail.')
    parser.add_argument('--parameter-experiment', type=Path, help='Flight-log only YAML: group_parameters and parameters rules; baseline always comes from the selected log snapshot')
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic profile or replay experiment sampling")
    parser.add_argument("--start-run-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Validate all plans and print import report without writing")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.seed < 0 or args.start_run_id < 0:
            raise ValueError("seed and start-run-id must be non-negative")
        if args.ground_wait_s is not None and (not math.isfinite(args.ground_wait_s) or args.ground_wait_s < 0):
            raise ValueError("--ground-wait-s must be finite and nonnegative")
        if (args.analyze_only or args.flight_id or args.mission_segment) and not args.flight_log:
            raise ValueError('--analyze-only, --flight-id and --mission-segment require --flight-log')
        if (args.parameter_groups or args.parameter_experiment) and not args.flight_log:
            raise ValueError('--parameter-groups and --parameter-experiment require --flight-log')
        if args.flight_log:
            from tools.scenario.flight_replay import generate_replay
            return generate_replay(args)
        if args.ground_wait_s is not None:
            raise ValueError("--ground-wait-s requires --flight-log")
        if any(not p.is_file() for p in args.plan):
            raise ValueError("--plan requires explicit .plan files, not directories")
        plans = [plan for path in args.plan for plan in discover_plans(path)]
        if len(set(plans)) != len(plans):
            raise ValueError("Duplicate plan inputs")
        target_home = simulator_home()
        imported = [read_plan(path, target_home, autopilots=args.autopilots) for path in plans]
        profiles = normalize_profiles(create_params_sets(args.controller_gain_profile)) if args.controller_gain_profile else {}
        needed = set(ap for item in imported for ap in item.report["autopilots"])
        if profiles and needed - profiles.keys():
            raise ValueError(f"Gain profile missing selected autopilots: {sorted(needed - profiles.keys())}")
        report = dict(version=1, runs=[dict(run_dir=f"run_{args.start_run_id + i:03d}", **item.report)
                                       for i, item in enumerate(imported)])
        outdir = args.outdir.expanduser().resolve()
        if outdir.exists():
            raise ValueError(f"Output already exists: {outdir}; select a new directory")
        if args.dry_run:
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        outdir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".mission-import-", dir=outdir.parent) as temporary:
            staging = Path(temporary) / "dataset"
            staging.mkdir()
            for item, entry in zip(imported, report["runs"]):
                selected_profiles = {ap: profiles[ap] for ap in item.report["autopilots"] if ap in profiles}
                write_run(item, staging / entry["run_dir"], profiles=selected_profiles, seed=args.seed)
            (staging / "import_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            if outdir.exists():
                raise ValueError(f"Output appeared during import: {outdir}")
            staging.rename(outdir)
        for entry in report["runs"]:
            print(f"{entry['source_file']} -> {outdir / entry['run_dir'] / 'scenario.yaml'}")
            print(f"  {entry['waypoint_count']} waypoints; {len(entry['normalizations'])} normalized fields; "
                  f"speed={entry['default_speed_m_s']} m/s ({entry['default_speed_source']})")
        return 0
    except (OSError, ValueError, TypeError, ET.ParseError, yaml.YAMLError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
