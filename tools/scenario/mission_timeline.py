"""Firmware-independent audit of armed intervals and mission execution segments.

Decoders supply timestamped facts. Analysis never discards an interrupted or
unmatched interval, and never treats mission item selection as item completion.
"""
from __future__ import annotations

from copy import deepcopy


# Correlate adjacent native log writes only; this never certifies a landing.
# ArduPilot's disarm routine can emit LAND_COMPLETE even without a touchdown.
POST_DISARM_EVENT_WINDOW_US = 10_000


def build_timeline(events, *, start_us, end_us, issues=()):
    events = sorted(deepcopy(events), key=lambda e: e['timestamp_us'])
    state = dict(armed=False, mode=None, landed=None, home=None, revision=None, revision_event=None)
    flights, active, segment = [], None, None
    pending_disarm = None

    def close_segment(t):
        nonlocal segment
        if segment is not None:
            segment['end_time_us'] = t
            segment = None

    def open_segment(t, reason):
        nonlocal segment
        close_segment(t)
        segment = dict(id=f"segment_{len(active['segments']):03d}", start_time_us=t,
                       end_time_us=None, mode=state['mode'], revision=state['revision'],
                       boundary=reason, observations=[], progress=[])
        active['segments'].append(segment)

    for event in events:
        t, kind = event['timestamp_us'], event['kind']
        old = deepcopy(state)
        if kind == 'state':
            state.update({k: event[k] for k in ('armed', 'mode') if k in event})
        elif kind == 'land':
            state['landed'] = event['landed']
        elif kind == 'home':
            state['home'] = event['home']
        elif kind == 'revision':
            state['revision'] = event['revision']
            state['revision_event'] = event
        if state['armed'] and active is None:
            pending_disarm = None
            active = dict(id=f'flight_{len(flights):03d}', arm_time_us=event.get('armed_time_us') or t,
                          start_observed=t >= start_us, end_time_us=None, disarm_time_us=None, home_lla=state['home'],
                          initial_revision=state['revision_event'],
                          segments=[], events=[], initial_landed=state['landed'], airborne_intervals=[], outcome='unknown', issues=list(issues))
            flights.append(active)
            open_segment(max(t, start_us), 'arm')
        if active is None:
            # Preserve the native post-disarm reset next to its armed interval,
            # but do not close an airborne interval or turn it into a touchdown.
            if (kind == 'land' and event.get('evidence') == 'EV.LAND_COMPLETE'
                    and pending_disarm is not None
                    and 0 <= t - pending_disarm['disarm_time_us'] <= POST_DISARM_EVENT_WINDOW_US):
                pending_disarm['post_disarm_land_events'] = [event]
            if kind == 'land':
                pending_disarm = None
            continue
        active['events'].append(event)
        if kind == 'home':
            if active['home_lla'] is None:
                active['home_lla'] = state['home']
        if not state['armed']:
            close_segment(t)
            active['end_time_us'] = t
            active['disarm_time_us'] = t
            air = active['airborne_intervals']
            active['outcome'] = 'landed_disarmed' if air and air[-1]['end_time_us'] is not None and state['landed'] else 'disarmed_landing_unconfirmed'
            pending_disarm, active = active, None
            continue
        if kind == 'state' and state['mode'] != old['mode']:
            open_segment(t, 'mode_change')
        elif kind == 'revision' and state['revision'] != old['revision']:
            open_segment(t, 'mission_revision_change')
        if kind == 'land':
            air = active['airborne_intervals']
            if not state['landed'] and (not air or air[-1]['end_time_us'] is not None):
                air.append(dict(start_time_us=t, end_time_us=None))
            elif state['landed'] and air and air[-1]['end_time_us'] is None:
                air[-1]['end_time_us'] = t
        if kind == 'item' and state['mode'] == 'MISSION' and t >= start_us:
            # A backward selection is a boundary to audit, not proof of a new plan.
            if segment['observations'] and event['sequence'] < segment['observations'][-1]['sequence']:
                open_segment(t, 'sequence_rewind')
            segment['observations'].append(event)
        if kind == 'progress' and state['mode'] == 'MISSION' and t >= start_us:
            segment['progress'].append(event)
    if active is not None:
        close_segment(end_us)
        active['issues'].append('disarm not observed')
    return dict(version=1, start_time_us=start_us, end_time_us=end_us, flights=flights,
                issues=list(issues))


def analyze_log(log, plans):
    from tools.scenario.mission_matching import match_timeline
    decoded = log.adapter.decode_events(log.data)
    timeline = build_timeline(**decoded)
    timeline.update(autopilot=log.autopilot, source_log=str(log.path), source_log_sha256=log.sha256)
    audit = match_timeline(timeline, plans, log.autopilot)
    for flight in audit['flights']:
        flight['replay_window'] = select_replay_window(flight)
    return audit


def select_replay_window(flight):
    """Select the first observed mission through disarm, after confirmed ground.

    The armed interval remains the audit/selection unit. Only a prefix before
    any observed mission may be excluded; interventions and later missions are
    never removed just because their mode or outcome is unsupported.
    """
    missions = [s for s in flight['segments'] if s['mode'] == 'MISSION' and s['observations']]
    if not missions:
        return None
    start, end = missions[0]['start_time_us'], flight['end_time_us']
    land_events = [e for e in flight['events'] if e['kind'] == 'land' and e['timestamp_us'] <= start]
    grounded = land_events[-1]['landed'] if land_events else flight['initial_landed']
    prior_air = [a for a in flight['airborne_intervals']
                 if a['end_time_us'] is not None and a['end_time_us'] <= start]
    air = [a for a in flight['airborne_intervals'] if a not in prior_air]
    issues = []
    if any(s['mode'] == 'MISSION' and s['end_time_us'] <= start for s in flight['segments']):
        issues.append('earlier MISSION segment has no observed execution; cannot discard a possible mission attempt')
    if grounded is not True:
        issues.append('ground state at mission start is unconfirmed')
    if flight['outcome'] != 'landed_disarmed':
        issues.append('landing before disarm is not confirmed')
    if len(air) != 1 or air[0]['start_time_us'] < start:
        issues.append('one ground takeoff within the selected mission window is required; '
                      'midair setup or repeated takeoffs after mission start are unsupported')
    excluded = []
    if start > flight['arm_time_us']:
        excluded.append(dict(start_time_us=flight['arm_time_us'], end_time_us=start,
            end_exclusive=True, reason='before_first_observed_mission',
            segment_ids=[s['id'] for s in flight['segments'] if s['end_time_us'] <= start],
            airborne_intervals=deepcopy(prior_air)))
    return dict(start_time_us=start, end_time_us=end, source_arm_time_us=flight['arm_time_us'],
                start_basis='first observed MISSION entry', initial_landed=grounded,
                ground_evidence=deepcopy(land_events[-1]) if land_events else None,
                airborne_intervals=deepcopy(air), excluded_intervals=excluded, issues=issues)
