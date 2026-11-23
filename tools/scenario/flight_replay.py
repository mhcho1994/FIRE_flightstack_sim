"""Generate timeline SITL sessions from native flight logs, regardless of completion."""
from __future__ import annotations
import csv
import json
import sys
import tempfile
from pathlib import Path

import yaml

from tools.scenario.flight_log import read_flight_log, sha256
from tools.scenario.mission_log_validation import mission_items
from tools.scenario.mission_timeline import select_replay_window
from tools.scenario.capture_flight_parameters import read_flight_parameters, select_snapshot
from tools.commander.fc_parameter_profiles import normalize_profiles, resolve_profiles
from tools.commander.flight_parameters import REPLAY_FLOAT_PARAMETERS, parse_flight_parameters


def generate_replay(args):
    from tools.scenario.import_real_flight_mission import read_plan, simulator_home, write_run, lla_to_ned
    if args.controller_gain_profile:
        raise ValueError('--controller-gain-profile is only available for plan-only conversion; --flight-log always extracts time-specific gains from the log')
    experiment = {}
    if getattr(args, 'parameter_experiment', None):
        experiment = yaml.safe_load(args.parameter_experiment.read_text())
        if not isinstance(experiment, dict) or set(experiment) - {'group_parameters', 'parameters'}:
            raise ValueError('Parameter experiment accepts only group_parameters and parameters; baseline must come from log')
    plans = [p.resolve(strict=True) for p in args.plan]
    if len(set(plans)) != len(plans) or any(not p.is_file() or p.suffix.lower() != '.plan' for p in plans):
        raise ValueError('Replay requires distinct explicit .plan files (KML has no complete command mission)')
    log = read_flight_log(args.flight_log)
    autopilot = log.autopilot
    digest = log.sha256
    if args.autopilots and args.autopilots != [autopilot]:
        raise ValueError(f'Flight log is {autopilot}; --autopilots must select the same firmware')
    home = simulator_home()
    imported = {str(p): read_plan(p, home, autopilots=[autopilot], for_replay=True) for p in plans}
    for plan in imported.values():
        if plan.report['source_autopilot'] != autopilot:
            raise ValueError(f"Plan firmware {plan.report['source_autopilot']} does not match flight log {autopilot}")
    if getattr(args, 'analyze_only', False):
        print(json.dumps(log.analyze(plans), indent=2, allow_nan=False))
        return 0
    if getattr(args, 'mission_segment', None):
        from tools.scenario.mission_segment import mission_segment_flights
        flights, excluded, audit = mission_segment_flights(log, plans, args.mission_segment)
    else:
        flights, excluded, audit = timeline_flights(log, plans, getattr(args, 'flight_id', None))
    origin = flights[0]['home_lla']
    profiles, phases = [], []
    for index, flight in enumerate(flights):
        timestamp = flight['start_time_us']
        snapshot = f'time:{timestamp // 1000000}.{timestamp % 1000000:06d}'
        history_names = ['MPC_XY_CRUISE', *REPLAY_FLOAT_PARAMETERS] if autopilot=='px4' else ['WP_SPD','WPNAV_SPEED']
        segment_parameters = ['MIS_TAKEOFF_ALT'] if flight.get('replay_scope') == 'mission_segment' else []
        history_names.extend(segment_parameters)
        profile, capture = read_flight_parameters(args.flight_log, autopilot=autopilot, snapshot=snapshot, groups=getattr(args, "parameter_groups", None), additional_parameters=history_names)
        if capture['source_sha256'] != digest:
            raise ValueError('Flight log changed between mission analysis and parameter capture')
        profile[autopilot].update(experiment)
        normalized_profile = normalize_profiles(profile)
        resolved = resolve_profiles(normalized_profile, args.seed, args.start_run_id + index)[autopilot]
        gain_names = set(profile[autopilot]['baseline'])
        schema = profile[autopilot]['parameter_schema']
        speed_name = ('MPC_XY_CRUISE' if autopilot == 'px4' else
                      'WP_SPD' if schema == 'arducopter-4.7' else 'WPNAV_SPEED')
        required = [speed_name, *segment_parameters]
        if autopilot == 'px4' and schema == 'px4-mc-v1':
            required.extend(REPLAY_FLOAT_PARAMETERS)
        # Gains and flight settings must use the same cutoff. Never substitute
        # plan metadata, firmware defaults or values first observed after it.
        snapshot_values = select_snapshot(capture['history'], snapshot)
        missing = sorted(set(required) - snapshot_values.keys())
        if missing:
            raise ValueError(f"{flight['id']}: missing required {autopilot} flight parameters at {snapshot} "
                             f"in {log.path.name}: {', '.join(missing)}. "
                             "No plan values or firmware defaults were substituted; supply a full log")
        flight_parameters = parse_flight_parameters(dict(
            values={name: resolved['values'].get(name, snapshot_values[name]) for name in required}, parameter_schema=schema,
            source='flight log at mission start'), autopilot)
        names = gain_names | set(flight_parameters['values'])
        for change in capture['history']['changes']:
            t = change['timestamp_us']
            before_end = t is not None and (t < flight['end_time_us'] if flight.get('replay_scope') == 'mission_segment' else t <= flight['end_time_us'])
            if change['parameter'] in resolved['sampling']['rules'] and t is not None and timestamp < t and before_end:
                raise ValueError(f"{change['parameter']}: variation conflicts with a recorded in-flight parameter change")
            if change['parameter'] in names and t is not None and timestamp < t and before_end:
                flight['events'].append(dict(id=f"parameter_{len(flight['events']):03d}",
                    action='parameters', time_s=(t-timestamp)/1e6, source_time_us=t,
                    values={change['parameter']: change['after']}))
        flight['events'].sort(key=lambda e:e['time_s'])
        profiles.append(normalized_profile)
        delay = 0.0 if index == 0 else (flight['arm_time_us'] - flights[index-1]['end_time_us']) / 1e6
        if args.ground_wait_s is not None and index:
            delay = args.ground_wait_s
        phases.append(dict(flight, id=flight.get('id', f'flight_{index:03d}'),
            plan_sha256=sha256(flight['plan']), ground_wait_s=delay,
            timeout_s=max(180., 3 * (flight['end_time_us'] - timestamp) / 1e6),
            controller_gains=resolved,
            flight_parameters=flight_parameters,
            mission_items=[dict(item, source_sequence=seq) for seq, item in enumerate(
                mission_items(flight['plan'], origin, home, autopilot)) if autopilot != 'ardupilot' or seq],
            gain_capture={k: v for k, v in capture.items() if k != 'history'}))
        phase = phases[-1]
        phase['replay_policy'] = 'recorded_time'
        phase['event_semantics'] = 'mode effects, not failure-cause injection'
        phase['mission_definitions'] = {path: [dict(item, source_sequence=seq) for seq,item in enumerate(
            mission_items(path, origin, home, autopilot)) if autopilot != 'ardupilot' or seq]
            for path in flight['mission_plans']}
    warnings = [warning for phase in phases for warning in phase['warnings']]
    warnings.extend(dict(code='unused_plan', plan=path,
        message=f'{Path(path).name}: no uniquely matched execution in this log; candidate retained in the analysis')
        for path in audit['unused_plans'])
    session = dict(version=2, autopilot=autopilot, source_log_format=log.log_format, source_log=str(Path(args.flight_log).resolve()),
        source_log_sha256=digest, source_origin_lla=origin, target_origin_lla=list(home),
        coordinate_transform='one shared horizontal NED origin; altitude remains relative to each flight home',
        phases=phases, excluded_armed_intervals=excluded, analysis=audit, warnings=warnings,
        limitations=['Manual control and unobserved initial states require a separate setup policy',
                     'Only explicitly listed parameters are copied; hardware, estimator and unlisted firmware defaults remain',
                     'Airframe, wind, estimator state and ground handling may differ from the real flight'])
    if any(p.get('initial_state', {}).get('kind') == 'in_air' for p in phases):
        from tools.scenario.mission_segment import IN_AIR_LIMITATIONS
        session['limitations'].extend(IN_AIR_LIMITATIONS)
    from tools.commander.mission_session import normalize_session
    normalize_session(dict(mission_session=session), autopilot)
    for plan in plans:
        if sha256(plan) != imported[str(plan)].report['source_sha256']:
            raise ValueError(f'Plan changed during generation: {plan}')
    reference_positions = {}
    for phase in phases:
        positions = log.positions(phase)
        if phase.get('replay_scope') == 'mission_segment':
            positions = [p for p in positions if p[0] < phase['end_time_us']]
            if (not positions or positions[0][0] - phase['start_time_us'] > 500_000
                    or phase['end_time_us'] - positions[-1][0] > 500_000
                    or any(b[0] - a[0] > 1_000_000 for a, b in zip(positions, positions[1:]))):
                raise ValueError('Selected segment has incomplete reference trajectory coverage')
        if len(positions) < 2:
            raise ValueError(f'{phase["id"]}: insufficient native trajectory samples for comparison')
        reference_positions[phase['id']] = positions
    outdir = args.outdir.expanduser().resolve()
    if outdir.exists():
        raise ValueError(f'Output already exists: {outdir}')
    for warning in warnings:
        print(f"WARNING [{warning['code']}]: {warning['message']}", file=sys.stderr)
    if args.dry_run:
        print(json.dumps(session, indent=2, allow_nan=False))
        return 0
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.flight-replay-', dir=outdir.parent) as temporary:
        # Keep the numeric run id for gain sampling while publishing the session
        # directly at --outdir, regardless of the output directory's name.
        run = Path(temporary) / f'run_{args.start_run_id:03d}'
        first = imported[flights[0]['plan']]
        first.flight_parameters = {autopilot: phases[0]['flight_parameters']}
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
        from tools.commander.mission_session import normalize_session
        normalize_session(data, autopilot)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        (run / 'replay_report.json').write_text(json.dumps(session, indent=2, allow_nan=False)+'\n')
        (run / 'import_report.json').write_text(json.dumps(dict(
            mode='flight_replay', report='replay_report.json',
            execution_source='scenario.yaml mission_session.phases[].mission_items',
            coordinate_transform=session['coordinate_transform'],
            normalizations=['Translate all mission coordinates using one actual-home horizontal origin',
                            'Keep relative altitude and unspecified yaw; preserve TAKEOFF/LAND positions',
                            'Require recorded flight parameters at each mission start and replay their changes'],
            phase_count=len(phases),
            phases=[dict(id=p['id'], replay_scope=p.get('replay_scope', 'flight'),
                         start_condition=p.get('initial_state', {}).get('kind', 'on_ground'),
                         initial_state=p.get('initial_state'),
                         start_source_sequence=p.get('start_source_sequence', 0),
                         comparison_window=p['replay_window']) for p in phases],
            warnings=warnings), indent=2)+'\n')
        if outdir.exists():
            raise ValueError(f'Output appeared during import: {outdir}')
        run.rename(outdir)
    print(f'{len(phases)} matched flights -> {outdir / "scenario.yaml"}')
    for phase in phases:
        print(f'  {phase["id"]}: {Path(phase["plan"]).name}, {phase["gain_capture"]["snapshot"]}, ground wait={phase["ground_wait_s"]:.3f}s')
    return 0


def timeline_flights(log, plans, selected_ids=None):
    """Compile supported continuous flights; preserve every other interval in audit."""
    audit = log.analyze(plans)
    chosen, excluded = [], []
    known = {f['id'] for f in audit['flights']}
    if selected_ids and set(selected_ids)-known:
        raise ValueError(f'Unknown --flight-id: {sorted(set(selected_ids)-known)}')
    for f in audit['flights']:
        segments = [s for s in f['segments'] if s['observations'] and s['mode']=='MISSION']
        if selected_ids and f['id'] not in selected_ids:
            excluded.append(dict(id=f['id'], reason='not selected')); continue
        if not segments:
            excluded.append(dict(id=f['id'], reason='no observed mission execution')); continue
        window = select_replay_window(f)
        f['replay_window'] = window
        problems = list(f['issues']) + window['issues']
        for s in segments:
            if s['match']['status'] != 'matched_observed':
                problems.append(f"{s['id']}: {s['match']['status']} plan candidates")
        first = segments[0]
        start = window['start_time_us']
        homes = [e['home'] for e in f['events'] if e['kind']=='home' and e['timestamp_us'] <= start]
        if homes:
            f['home_lla'] = homes[-1]
        if any(e['kind']=='home' and e['timestamp_us'] > start and e['home'] != f['home_lla'] for e in f['events']):
            problems.append('home changed during mission execution')
        if f['home_lla'] is None:
            problems.append('actual home unavailable')
        first_item = first['observations'][0]
        if first_item['sequence'] != 0 or first_item['command'] != 22:
            problems.append('initial TAKEOFF from item zero not observed; resumed midair initial state requires a setup policy')
        tail = [s for s in f['segments'] if s['end_time_us'] > start]
        if any(s['mode'] not in ('MISSION','HOLD','RTL','LAND') for s in tail):
            problems.append('manual/offboard segment has no supported input replay')
        if any(e['kind']=='home_invalid' and e['timestamp_us'] >= start for e in f['events']):
            problems.append('home becomes invalid during flight')
        if any(not p['valid'] for s in segments for p in s['progress']):
            problems.append('invalid mission_result; mission validity is unconfirmed')
        if problems:
            raise ValueError(f"{f['id']}: cannot compile timeline (matching or replay capability): {'; '.join(dict.fromkeys(problems))}. Use --analyze-only for the full audit")
        events, prior = [], first
        for s in tail:
            if s['id'] == first['id']:
                continue
            action = {'HOLD':'hold','RTL':'rtl','LAND':'land'}.get(s['mode'])
            extra = {}
            if s['mode']=='MISSION':
                if not s['observations']:
                    continue
                extra = dict(expected_source_sequence=s['source_sequence_start'])
                if s['plan'] != prior['plan'] or s['revision'] != prior['revision']:
                    action='replace_mission'; extra.update(plan=s['plan'], start_source_sequence=s['source_sequence_start'])
                elif s['boundary']=='sequence_rewind':
                    action='restart_mission'; extra['start_source_sequence']=s['source_sequence_start']
                else:
                    action='resume'
                prior = s
            if action:
                events.append(dict(id=f'event_{len(events):03d}', action=action,
                    time_s=(s['start_time_us']-start)/1e6, source_time_us=s['start_time_us'], **extra))
        last = tail[-1]
        if last['mode'] not in ('RTL','LAND') and not last.get('terminal_plan_land'):
            raise ValueError(f"{f['id']}: cannot compile timeline: no confirmed terminal plan LAND, RTL or LAND mode; an executable ending is required, not mission completion")
        warnings = []
        if any(e['airborne_intervals'] for e in window['excluded_intervals']):
            warnings.append(dict(code='pre_mission_flight_excluded', flight_id=f['id'],
                excluded_intervals=window['excluded_intervals'],
                message=f"{f['id']}: excluded flight before the first observed mission; "
                        f"replay begins on confirmed ground at time:{start / 1e6:.6f}; "
                        "the full armed interval remains in analysis"))
        for execution in f['mission_executions']:
            if execution['outcome'] != 'completed' or execution['mission_failure_observed']:
                outcome = execution['outcome']
                warnings.append(dict(code=('mission_failure_recorded' if execution['mission_failure_observed'] else
                        'mission_interrupted' if outcome=='interrupted' else 'mission_completion_unknown'),
                    flight_id=f['id'], mission_id=execution['id'], plan=execution['plan'],
                    source_mission_outcome=outcome, mission_failure_observed=execution['mission_failure_observed'],
                    observed_sequences=execution['observed_sequences'],
                    last_reached=execution['last_reached'], flight_terminal_mode=last['mode'],
                    message=f"{f['id']}/{execution['id']} ({Path(execution['plan']).name}): "
                        f"source mission outcome is {outcome}; "
                        + ("native failure was recorded; " if execution['mission_failure_observed'] else "")
                        + "replaying recorded effects without injecting failure causes or requiring full completion"))
        chosen.append(dict(id=f['id'], plan=first['plan'], arm_time_us=f['arm_time_us'], start_time_us=start,
            end_time_us=f['end_time_us'], home_lla=f['home_lla'], status='landed_disarmed',
            source_mission_outcome=f['mission_executions'][-1]['outcome'],
            mission_executions=f['mission_executions'], warnings=warnings,
            source_flight_outcome=f['outcome'], events=events, timeline_segments=tail,
            replay_window=window,
            completion='recorded_landing', match=first['match'],
            mission_plans=list(dict.fromkeys(s['plan'] for s in segments)),
            source_landing_time_us=window['airborne_intervals'][0]['end_time_us']))
    if not chosen:
        raise ValueError('No replayable mission flights; use --analyze-only to inspect all intervals')
    return chosen, excluded, audit
