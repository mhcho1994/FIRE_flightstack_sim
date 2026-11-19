"""Generate one persistent SITL session from matched, completed real flights."""
from __future__ import annotations
import csv
import json
import tempfile
from pathlib import Path

import yaml

from tools.scenario.flight_log import read_flight_log, sha256
from tools.scenario.mission_log_validation import mission_items
from tools.scenario.capture_flight_parameters import read_flight_parameters
from tools.commander.fc_parameter_profiles import normalize_profiles, resolve_profiles
from tools.commander.flight_parameters import parse_flight_parameters


def generate_replay(args):
    from tools.scenario.import_real_flight_mission import read_plan, simulator_home, write_run, lla_to_ned
    if args.controller_gain_profile:
        raise ValueError('--flight-log extracts time-specific gains; do not also supply --controller-gain-profile')
    plans = [p.resolve(strict=True) for p in args.input]
    if len(set(plans)) != len(plans) or any(not p.is_file() or p.suffix.lower() != '.plan' for p in plans):
        raise ValueError('Replay requires distinct explicit .plan files (KML has no complete command mission)')
    log = read_flight_log(args.flight_log)
    autopilot = log.autopilot
    digest = log.sha256
    if args.autopilots and args.autopilots != [autopilot]:
        raise ValueError(f'Flight log is {autopilot}; --autopilots must select the same firmware')
    home = simulator_home()
    imported = {str(p): read_plan(p, home, autopilots=[autopilot], use_firmware_speed=True) for p in plans}
    for plan in imported.values():
        if plan.report['source_autopilot'] != autopilot:
            raise ValueError(f"Plan firmware {plan.report['source_autopilot']} does not match flight log {autopilot}")
    flights, excluded = log.match_executions(plans)
    origin = flights[0]['home_lla']
    profiles, phases = [], []
    for index, flight in enumerate(flights):
        timestamp = flight['start_time_us']
        snapshot = f'time:{timestamp // 1000000}.{timestamp % 1000000:06d}'
        profile, capture = read_flight_parameters(args.flight_log, autopilot=autopilot, snapshot=snapshot)
        if capture['source_sha256'] != digest:
            raise ValueError('Flight log changed between mission analysis and parameter capture')
        gain_names = set(profile[autopilot]['baseline'])
        if any(timestamp < e['timestamp_us'] <= flight['end_time_us'] for e in capture['history']['changes']
               if e['parameter'] in gain_names and e['timestamp_us'] is not None):
            raise ValueError(f'{flight["plan"]}: gain changed during flight; dynamic gain replay is unsupported')
        schema = profile[autopilot]['parameter_schema']
        speed_name = ('MPC_XY_CRUISE' if autopilot == 'px4' else
                      'WP_SPD' if schema == 'arducopter-4.7' else 'WPNAV_SPEED')
        speed_scale = 100 if speed_name == 'WPNAV_SPEED' else 1
        flight_parameters = None
        if not args.use_firmware_speed:
            speed = (log.parameter_at(speed_name, timestamp) if args.speed_m_s is None
                     else args.speed_m_s * speed_scale)
            if args.speed_m_s is None and log.parameter_changed(speed_name, timestamp, flight['end_time_us'], speed):
                raise ValueError('Mission speed changed during flight; dynamic parameter replay is unsupported')
            flight_parameters = parse_flight_parameters(dict(values={speed_name: speed}, parameter_schema=schema,
                source='flight log at mission start' if args.speed_m_s is None else 'CLI --speed-m-s'), autopilot)
        profiles.append(normalize_profiles(profile))
        delay = 0.0 if index == 0 else (flight['arm_time_us'] - flights[index-1]['end_time_us']) / 1e6
        if args.ground_wait_s is not None and index:
            delay = args.ground_wait_s
        phases.append(dict(id=f'flight_{index:03d}', **flight,
            plan_sha256=sha256(flight['plan']), ground_wait_s=delay,
            timeout_s=max(180., 3 * (flight['end_time_us'] - timestamp) / 1e6),
            controller_gains=resolve_profiles(profiles[-1], args.seed, index)[autopilot],
            flight_parameters=flight_parameters,
            mission_items=[dict(item, source_sequence=seq) for seq, item in enumerate(
                mission_items(flight['plan'], origin, home, autopilot)) if autopilot != 'ardupilot' or seq],
            gain_capture={k: v for k, v in capture.items() if k != 'history'}))
    session = dict(version=1, autopilot=autopilot, source_log_format=log.log_format, source_log=str(Path(args.flight_log).resolve()),
        source_log_sha256=digest, source_origin_lla=origin, target_origin_lla=list(home),
        coordinate_transform='one shared horizontal NED origin; altitude remains relative to each flight home',
        phases=phases, excluded_armed_intervals=excluded,
        limitations=['No manual flight or midair mission replacement replay',
                     'Only controller gains and horizontal mission speed are copied; other firmware defaults remain',
                     'Airframe, wind, estimator state and ground handling may differ from the real flight'])
    for plan in plans:
        if sha256(plan) != imported[str(plan)].report['source_sha256']:
            raise ValueError(f'Plan changed during generation: {plan}')
    reference_positions = {}
    for phase in phases:
        positions = log.positions(phase)
        if len(positions) < 2:
            raise ValueError(f'{phase["id"]}: insufficient native trajectory samples for comparison')
        reference_positions[phase['id']] = positions
    outdir = args.outdir.expanduser().resolve()
    if outdir.exists():
        raise ValueError(f'Output already exists: {outdir}')
    if args.dry_run:
        print(json.dumps(session, indent=2, allow_nan=False))
        return 0
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.flight-replay-', dir=outdir.parent) as temporary:
        # Keep the numeric run id for gain sampling while publishing the session
        # directly at --outdir, regardless of the output directory's name.
        run = Path(temporary) / f'run_{args.start_run_id:03d}'
        first = imported[flights[0]['plan']]
        first.flight_parameters = {} if phases[0]['flight_parameters'] is None else {autopilot: phases[0]['flight_parameters']}
        write_run(first, run, profiles=profiles[0], seed=args.seed)
        replay_dir = run / 'replay'; replay_dir.mkdir()
        for index, plan in enumerate(plans):
            (replay_dir / f'source_{index:03d}.plan').write_bytes(plan.read_bytes())
        reference = replay_dir / 'reference_trajectory.csv'
        with reference.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['phase', 'boot_time_us', 'time_s', 'north_m', 'east_m', 'up_m'])
            for phase in phases:
                for timestamp, lat, lon, height in reference_positions[phase['id']]:
                    n, e, _ = lla_to_ned([lat, lon, height], origin)
                    writer.writerow([phase['id'], timestamp, (timestamp-phase['start_time_us'])/1e6,
                                     n, e, height])
        if sha256(args.flight_log) != digest:
            raise ValueError('Source log changed during generation')
        session.update(reference_trajectory='replay/reference_trajectory.csv', reference_sha256=sha256(reference))
        path = run / 'scenario.yaml'; data = yaml.safe_load(path.read_text())
        data['common']['scenario']['name'] = 'real_flight_session'
        data['mission_session'] = session
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        (run / 'replay_report.json').write_text(json.dumps(session, indent=2, allow_nan=False)+'\n')
        (run / 'import_report.json').write_text(json.dumps(dict(
            mode='flight_replay', report='replay_report.json',
            execution_source='scenario.yaml mission_session.phases[].mission_items',
            coordinate_transform=session['coordinate_transform'],
            normalizations=['Translate all mission coordinates using one actual-home horizontal origin',
                            'Keep relative altitude and unspecified yaw; preserve TAKEOFF/LAND positions',
                            'Use recorded horizontal mission speed unless explicitly overridden'],
            phase_count=len(phases)), indent=2)+'\n')
        if outdir.exists():
            raise ValueError(f'Output appeared during import: {outdir}')
        run.rename(outdir)
    print(f'{len(phases)} matched flights -> {outdir / "scenario.yaml"}')
    for phase in phases:
        print(f'  {phase["id"]}: {Path(phase["plan"]).name}, {phase["gain_capture"]["snapshot"]}, ground wait={phase["ground_wait_s"]:.3f}s')
    return 0
