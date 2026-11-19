"""Offline mission session normalization shared by launchers and commanders.

Logical mission_items exclude ArduPilot's home slot. A native adapter adds that
slot after wind checkpoints, retaining scenario_index/source_sequence mappings.
"""
from __future__ import annotations

import copy
import math

EXECUTION_VERSION = 'mission-session-v1'


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
    if session.get('version') != 1 or session.get('autopilot') != autopilot:
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
    for phase in phases:
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
        if phase['completion'] != 'landed_disarmed':
            raise ValueError('Only landed_disarmed completion is currently supported')
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
