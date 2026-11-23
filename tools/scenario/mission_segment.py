"""Explicit autonomous windows, with observed (not hidden) start state."""
from __future__ import annotations

import math

from tools.scenario import px4_mission as px4

IN_AIR_LIMITATIONS = [
    'In-air start: setup flight matches observed position, velocity and yaw within reported tolerances.',
    'Controller integrators, filters, previous setpoints and EKF internal state are not restored from the source flight.',
    'Initial attitude and angular rates are recorded for audit, not imposed; initialization transients may differ.',
    'Setup flight and post-window landing are excluded from comparison; this is not full-flight input replay.',
]


def _sample(log, name, timestamp, fields):
    row = px4._latest(px4.topic(log, name, fields), timestamp)
    if row is None or timestamp - row['timestamp'] > 250_000:
        raise ValueError(f'{name}: no fresh sample at or before mission segment start (250 ms limit)')
    if not all(math.isfinite(row[k]) for k in fields):
        raise ValueError(f'{name}: nonfinite initial state')
    return row


def initial_state(log, timestamp, home, landed):
    from tools.scenario.import_real_flight_mission import lla_to_ned
    if landed:
        return dict(kind='on_ground', source_time_us=timestamp, internal_state_restored=False)
    gp = _sample(log, 'vehicle_global_position', timestamp, ('lat', 'lon', 'alt'))
    local = _sample(log, 'vehicle_local_position', timestamp, ('vx', 'vy', 'vz', 'heading'))
    attitude = _sample(log, 'vehicle_attitude', timestamp, tuple(f'q[{i}]' for i in range(4)))
    rates = _sample(log, 'vehicle_angular_velocity', timestamp, tuple(f'xyz[{i}]' for i in range(3)))
    for row, flags in ((gp, ('lat_lon_valid', 'alt_valid')),
                       (local, ('xy_valid', 'z_valid', 'v_xy_valid', 'v_z_valid', 'heading_good_for_control'))):
        if any(not row.get(k, True) for k in flags):
            raise ValueError('Invalid position, velocity or heading at mission segment start')
    height = gp['alt'] - home[2]
    if height < .5:
        raise ValueError('In-air setup requires at least 0.5 m observed height above home')
    return dict(kind='in_air', source_time_us=timestamp,
        source_position_lla=[gp['lat'], gp['lon'], gp['alt']],
        position_ned_m=list(lla_to_ned([gp['lat'], gp['lon'], height], home)),
        velocity_ned_m_s=[local[k] for k in ('vx', 'vy', 'vz')], yaw_rad=local['heading'],
        attitude_quaternion_wxyz=[attitude[f'q[{i}]'] for i in range(4)],
        angular_velocity_body_rad_s=[rates[f'xyz[{i}]'] for i in range(3)],
        sample_times_us={name: row['timestamp'] for name, row in (
            ('vehicle_global_position', gp), ('vehicle_local_position', local),
            ('vehicle_attitude', attitude), ('vehicle_angular_velocity', rates))},
        setup=dict(policy='position_velocity_yaw', position_tolerance_m=.5,
                   velocity_tolerance_m_s=.3, yaw_tolerance_rad=math.radians(5),
                   settle_time_s=1., timeout_s=120.),
        internal_state_restored=False, limitations=list(IN_AIR_LIMITATIONS))


def mission_segment_flights(log, plans, selector):
    if log.autopilot != 'px4':
        raise ValueError('--mission-segment currently supports PX4 ULog only')
    audit = log.analyze(plans)
    if audit['issues']:
        raise ValueError('Mission segment log issues: ' + '; '.join(audit['issues']))
    candidates = [(f, s) for f in audit['flights'] for s in f['segments']
                  if f"{f['id']}:{s['id']}" == selector]
    if len(candidates) != 1:
        raise ValueError(f'Unknown --mission-segment {selector}; use --analyze-only for flight:segment IDs')
    flight, segment = candidates[0]
    if (segment['mode'] != 'MISSION' or not segment.get('observations')
            or segment.get('match', {}).get('status') != 'matched_observed'
            or not segment['match']['observed_sequences']):
        raise ValueError('Selected segment needs uniquely matched canonical mission targets')
    start, end = segment['start_time_us'], segment['end_time_us']
    if start < audit['start_time_us'] or end is None or end <= start:
        raise ValueError('Selected mission segment has no observed time window')
    if not segment['progress'] or any(not p['valid'] for p in segment['progress']):
        raise ValueError('Selected segment has no confirmed valid mission_result')
    home_row = px4._latest(px4.topic(log.data, 'home_position'), start)
    land = px4._latest(px4.topic(log.data, 'vehicle_land_detected'), start)
    if not home_row or not home_row['valid_hpos'] or not home_row['valid_alt'] or not land:
        raise ValueError('Mission segment start needs confirmed home and landed state')
    home = [home_row[k] for k in ('lat', 'lon', 'alt')]
    if any(e['kind'] == 'home_invalid' or (e['kind'] == 'home' and e['home'] != home)
           for e in flight['events'] if start <= e['timestamp_us'] < end):
        raise ValueError('Home changes or becomes invalid in the selected mission segment')
    state = initial_state(log.data, start, home, bool(land['landed']))
    sequence = segment['source_sequence_start']
    if state['kind'] == 'on_ground' and (sequence != 0 or segment['observations'][0]['command'] != 22):
        raise ValueError('On-ground segment requires the initial TAKEOFF from item zero')
    warnings = [dict(code='mission_segment_only', flight_id=flight['id'], segment_id=segment['id'],
        message=f'{selector}: compare only the selected MISSION window; exclude other segments, setup and cleanup')]
    if state['kind'] == 'in_air':
        warnings.append(dict(code='in_air_initialization', message=' '.join(IN_AIR_LIMITATIONS)))
    phase = dict(id=flight['id'] + '_' + segment['id'], plan=segment['plan'],
        source_flight_id=flight['id'], source_segment_id=segment['id'],
        arm_time_us=flight['arm_time_us'], start_time_us=start, end_time_us=end,
        home_lla=home, events=[], warnings=warnings, match=segment['match'],
        timeline_segments=[segment], mission_executions=flight['mission_executions'],
        mission_plans=[segment['plan']], source_mission_outcome=segment['mission_outcome'],
        source_flight_outcome=flight['outcome'], replay_scope='mission_segment',
        start_source_sequence=sequence, initial_state=state, completion='segment_window',
        comparison_duration_s=(end-start)/1e6,
        replay_window=dict(start_time_us=start, end_time_us=end, end_exclusive=True,
            start_basis='explicitly selected MISSION segment', initial_landed=bool(land['landed']),
            excluded_segment_ids=[s['id'] for s in flight['segments'] if s['id'] != segment['id']]))
    excluded = [dict(id=f['id'], reason='not selected') for f in audit['flights'] if f is not flight]
    return [phase], excluded, audit
