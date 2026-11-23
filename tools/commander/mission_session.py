"""Offline mission session normalization shared by launchers and commanders.

Logical mission_items exclude ArduPilot's home slot. A native adapter adds that
slot after wind checkpoints, retaining scenario_index/source_sequence mappings.
"""
from __future__ import annotations

import copy
import math

EXECUTION_VERSION = 'mission-session-v2'


def session_block(data):
    if 'mission_session' in data and 'replay_session' in data:
        raise ValueError('Supply mission_session or replay_session, not both')
    return data.get('mission_session', data.get('replay_session'))


def phase_settings(data, autopilot, phase=None):
    settings = copy.deepcopy(data.get('autopilots', {}).get(autopilot, {}))
    if phase is None:
        session = session_block(data)
        if session is not None:
            if session.get('autopilot') != autopilot:
                raise ValueError('Mission session autopilot does not match launcher')
            if not session.get('phases'):
                raise ValueError('Mission session needs phases')
            phase = session['phases'][0]
    for key in ('controller_gains', 'flight_parameters'):
        if phase is not None and key in phase:
            settings[key] = copy.deepcopy(phase[key])
    return settings


def legacy_items(data):
    """Preserve legacy command geometry/defaults without a firmware home item."""
    mission = data.get('common', {}).get('scenario', {})
    home = mission.get('home_lla')
    if home is None or len(home) != 3:
        raise ValueError('common.scenario.home_lla is missing or invalid')
    points, speeds = mission.get('waypoints_lla', []), mission.get('speed_m_s', [])
    result = []
    for index, command in enumerate(mission.get('command', [])):
        if command is None:
            continue
        command = int(command)
        p, frame = [0., 0., 0., None, 0., 0., 0.], 3
        if command == 22:
            p[4:] = [*home[:2], mission.get('takeoff_alt_m', 10.)]
        elif command == 16:
            if index >= len(points) or points[index] is None:
                raise ValueError(f'Waypoint missing for command index {index}')
            p[1], p[4:] = 1., list(points[index])
        elif command == 21:
            p[4:] = [*home[:2], 0.]
        elif command == 178:
            if index >= len(speeds) or speeds[index] is None:
                raise ValueError(f'Speed missing for command index {index}')
            frame, p = 2, [1., float(speeds[index]), -1., 0., 0., 0., 0.]
        else:
            raise ValueError(f'Unsupported mission command: {command}')
        result.append(dict(command=command, frame=frame, autocontinue=1, scenario_index=index,
                           **{f'p{i+1}': value for i, value in enumerate(p)}))
    return result


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Invalid {name}')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'Invalid {name}')
    return value


def normalize_session(data, autopilot):
    """Return a detached, validated logical session from new or legacy YAML."""
    if autopilot not in ('ardupilot', 'px4'):
        raise ValueError('Unsupported mission autopilot')
    block = session_block(data)
    legacy_replay = 'replay_session' in data
    if block is None:
        mission = data.get('common', {}).get('scenario', {})
        block = dict(version=1, autopilot=autopilot, target_origin_lla=mission.get('home_lla'),
                     phases=[dict(id='flight_000', mission_items=legacy_items(data),
                                  takeoff_alt_m=mission.get('takeoff_alt_m', 10.))])
    session = copy.deepcopy(block)
    if session.get('version') not in (1, 2) or session.get('autopilot') != autopilot:
        raise ValueError('Unsupported mission session version/autopilot')
    phases = session.get('phases')
    if not isinstance(phases, list) or not phases or any(not isinstance(p, dict) for p in phases):
        raise ValueError('Mission session needs phases')
    ids = [p.get('id') for p in phases]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError('Mission phase IDs must be nonempty and unique')
    origin = session.get('target_origin_lla', data.get('common', {}).get('scenario', {}).get('home_lla'))
    if not isinstance(origin, (list, tuple)) or len(origin) != 3 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in origin):
        raise ValueError('Mission session needs a finite target_origin_lla')
    if not -89 < origin[0] < 89 or not -180 <= origin[1] <= 180:
        raise ValueError('Invalid mission origin coordinates')
    session['target_origin_lla'] = list(origin)
    from tools.commander.fc_parameter_registry import parameter_spec
    boot_reboot_values = None
    for phase in phases:
        effective_settings = phase_settings(data, autopilot, phase)
        reboot_values = {k: v for block in ('controller_gains', 'flight_parameters')
                         for k, v in (effective_settings.get(block) or {}).get('values', {}).items()
                         if parameter_spec(autopilot, k).get('rebootRequired')}
        if boot_reboot_values is None:
            boot_reboot_values = reboot_values
        elif any(k not in boot_reboot_values or boot_reboot_values[k] != v for k, v in reboot_values.items()):
            raise ValueError('Reboot-required parameters must stay fixed across session phases; use separate runs')
        for event in phase.get('events', []):
            if event.get('action') == 'parameters' and any(
                    parameter_spec(autopilot, k).get('rebootRequired') for k in event.get('values', {})):
                raise ValueError('Cannot replay an in-flight change to a reboot-required parameter')
        if legacy_replay:
            if phase.get('status') != 'landed_disarmed':
                raise ValueError('Legacy replay requires confirmed completed flights')
            native = phase['mission_items']
            if autopilot != 'ardupilot' or not native or native[0]['command'] != 16 or native[0]['frame'] != 0:
                raise ValueError('Legacy replay requires an ArduPilot home item')
            phase['mission_items'] = [dict(it, source_sequence=i) for i, it in enumerate(native) if i]
        settings = phase_settings(data, autopilot, phase)
        for key in ('controller_gains', 'flight_parameters'):
            phase[key] = settings.get(key)
        phase.setdefault('ground_wait_s', 0.)
        phase.setdefault('timeout_s', 300.)
        _number(phase['timeout_s'], 'timeout_s', positive=True)
        phase.setdefault('wall_timeout_s', max(60., 5 * phase['timeout_s']))
        phase.setdefault('completion', 'landed_disarmed')
        _number(phase['ground_wait_s'], 'ground_wait_s')
        for key in ('timeout_s', 'wall_timeout_s'):
            _number(phase[key], key, positive=True)
        if phase['completion'] not in (('landed_disarmed', 'recorded_landing', 'segment_window') if session['version']==2 else ('landed_disarmed',)):
            raise ValueError('Only landed_disarmed completion is currently supported')
        if session['version'] == 2:
            phase['_autopilot'] = autopilot
            validate_timeline_phase(phase)
            phase.pop('_autopilot')
        items = phase.get('mission_items', [])
        if not items or items[-1]['command'] != 21 or sum(it['command'] == 21 for it in items) != 1:
            raise ValueError('A phase must have exactly one final LAND')
        takeoffs = [i for i, it in enumerate(items) if it['command'] == 22]
        if takeoffs not in ([], [0]):
            raise ValueError('TAKEOFF must occur once at the start of a phase')
        if not takeoffs:
            if autopilot != 'ardupilot':
                raise ValueError('PX4 missions require an explicit TAKEOFF')
            _number(phase.get('takeoff_alt_m', 10.), 'takeoff_alt_m', positive=True)
        for index, item in enumerate(items):
            if item['command'] not in (16, 21, 22, 178):
                raise ValueError('Unsupported mission command')
            if item.get('autocontinue', 1) != 1:
                raise ValueError('Mission items must autocontinue')
            item.setdefault('autocontinue', 1)
            item.setdefault('scenario_index', index)
            for i in range(1, 8):
                value = item.get(f'p{i}')
                if i == 4 and value is None:
                    continue
                if not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError('Nonfinite mission parameter')
            if item['command'] in (16, 21, 22):
                if item['frame'] != 3:
                    raise ValueError('Session missions require relative-altitude frame 3')
                if not -89 < item['p5'] < 89 or not -180 <= item['p6'] <= 180:
                    raise ValueError('Invalid mission coordinates')
                if item['command'] != 21 and item['p7'] <= 0:
                    raise ValueError('Airborne mission altitudes must be positive')
            elif item['frame'] != 2 or item['p2'] <= 0:
                raise ValueError('Invalid mission speed command')
    return session


def comparison_session(data):
    """Expose the historical native numbering expected by BIN comparison tools."""
    block = session_block(data)
    if block is None or 'reference_trajectory' not in block:
        raise ValueError('Scenario has no real-flight comparison reference')
    session = normalize_session(data, block['autopilot'])
    if session['autopilot'] != 'ardupilot':
        raise ValueError('Native BIN comparison currently requires ArduPilot')
    for phase in session['phases']:
        phase['mission_items'] = [home_item(session['target_origin_lla'])] + phase['mission_items']
    return session


def home_item(origin):
    return dict(command=16, frame=0, autocontinue=1, scenario_index=None,
                p1=0., p2=0., p3=0., p4=0., p5=origin[0], p6=origin[1], p7=origin[2])


def validate_timeline_phase(phase):
    if phase.get('replay_policy') != 'recorded_time':
        raise ValueError('Timeline replay requires recorded_time policy')
    if phase.get('replay_scope') == 'mission_segment':
        validate_segment_phase(phase)
    elif phase.get('completion') == 'segment_window' or phase.get('initial_state', {}).get('kind') == 'in_air':
        raise ValueError('In-air setup and segment_window require explicit mission_segment scope')
    events = phase.get('events', [])
    if not isinstance(events, list):
        raise ValueError('Timeline events must be a list')
    previous = -1.
    ids = set()
    definitions = phase.get('mission_definitions', {})
    active_items = phase['mission_items']
    parameter_state = {key:copy.deepcopy(phase.get(key)) for key in ('controller_gains','flight_parameters')}
    for event in events:
        t = _number(event.get('time_s'), 'event time_s')
        if t < previous or t > phase['timeout_s']:
            raise ValueError('Timeline events must be sorted and inside the phase timeout')
        previous = t
        if not event.get('id') or event['id'] in ids:
            raise ValueError('Timeline event IDs must be unique')
        ids.add(event['id'])
        action = event.get('action')
        if phase.get('replay_scope') == 'mission_segment' and (action != 'parameters' or t >= phase['comparison_duration_s']):
            raise ValueError('Mission segment events may only change parameters inside the comparison window')
        if action not in ('hold','resume','rtl','land','restart_mission','replace_mission','parameters'):
            raise ValueError(f'Unsupported timeline action: {action}')
        if action == 'replace_mission' and event.get('plan') not in definitions:
            raise ValueError('Replacement mission definition is missing')
        if action in ('restart_mission', 'replace_mission'):
            sequence = event.get('start_source_sequence')
            items = definitions[event['plan']] if action == 'replace_mission' else active_items
            active_items = items
            if sequence not in [it.get('source_sequence') for it in items]:
                raise ValueError('Event start sequence is not in the mission')
        if action == 'parameters':
            known = {}
            for key in ('controller_gains','flight_parameters'):
                known.update((phase.get(key) or {}).get('values', {}))
            values = event.get('values')
            if not isinstance(values,dict) or not values or values.keys()-known.keys():
                raise ValueError('Parameter events may change only captured parameters')
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values.values()):
                raise ValueError('Parameter events must contain finite REAL32 values')
            from tools.commander.fc_parameter_profiles import parse_controller_parameters
            from tools.commander.flight_parameters import parse_flight_parameters
            for key, parser in [('controller_gains',parse_controller_parameters),('flight_parameters',parse_flight_parameters)]:
                config = parameter_state[key]
                if config is not None:
                    config = {k:v for k,v in config.items() if k not in ('parameter_file','parameter_file_sha256')}
                    config['values'].update({k:v for k,v in values.items() if k in config['values']})
                    parameter_state[key] = parser(config, phase['_autopilot'])
    # Validate every replacement using the existing mission geometry contract.
    for name, items in definitions.items():
        dummy = dict(version=1, autopilot=phase.get('_autopilot', 'px4'),
            target_origin_lla=[40.,-86.,0.], phases=[dict(id='validation',mission_items=items)])
        normalize_session(dict(mission_session=dummy), dummy['autopilot'])


def validate_segment_phase(phase):
    if phase.get('_autopilot') != 'px4' or phase.get('completion') != 'segment_window':
        raise ValueError('Mission segment comparison requires PX4 and segment_window completion')
    duration = _number(phase.get('comparison_duration_s'), 'comparison_duration_s', positive=True)
    if duration >= phase['timeout_s']:
        raise ValueError('Segment timeout must leave time for cleanup landing')
    start, end = phase.get('start_time_us'), phase.get('end_time_us')
    _number(start, 'source start time')
    _number(end, 'source end time', positive=True)
    if not math.isclose((end-start)/1e6, duration, abs_tol=1e-6):
        raise ValueError('Segment duration differs from source time window')
    sequence = phase.get('start_source_sequence')
    items = phase.get('mission_items', [])
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sum(it.get('source_sequence') == sequence for it in items) != 1:
        raise ValueError('Segment start sequence needs one source-to-upload mapping')
    state = phase.get('initial_state')
    if not isinstance(state, dict) or state.get('kind') not in ('on_ground', 'in_air'):
        raise ValueError('Segment needs an explicit on_ground or in_air initial state')
    if state.get('internal_state_restored') is not False or state.get('source_time_us') != start:
        raise ValueError('Segment initial state must disclose unrestored internal state and source time')
    if state['kind'] == 'on_ground':
        if sequence != 0:
            raise ValueError('On-ground segment must start at item zero')
        return
    for key, length in [('position_ned_m', 3), ('velocity_ned_m_s', 3),
                        ('attitude_quaternion_wxyz', 4), ('angular_velocity_body_rad_s', 3)]:
        vector = state.get(key)
        if (not isinstance(vector, (list, tuple)) or len(vector) != length
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector)):
            raise ValueError(f'Invalid initial_state.{key}')
    yaw = state.get('yaw_rad')
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)) or not math.isfinite(yaw):
        raise ValueError('Invalid initial_state.yaw_rad')
    if state['position_ned_m'][2] > -.5:
        raise ValueError('In-air initial height must be at least 0.5 m above home')
    setup = state.get('setup', {})
    if setup.get('policy') != 'position_velocity_yaw':
        raise ValueError('Unsupported in-air setup policy')
    for key in ('position_tolerance_m', 'velocity_tolerance_m_s', 'yaw_tolerance_rad', 'settle_time_s', 'timeout_s'):
        _number(setup.get(key), 'setup.' + key, positive=True)
