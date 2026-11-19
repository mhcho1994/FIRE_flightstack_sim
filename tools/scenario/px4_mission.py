"""Analyze observed PX4 mission executions; generate only complete ground-to-ground flights.

ULog records navigator work items, not a complete uploaded mission snapshot.
Match every supported plan item against observed execution and double-precision
position setpoints. Internal climb/landing work items are not separate commands.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct

from tools.scenario.flight_log import detect_log_format, sha256
from tools.scenario.mission_log_validation import horizontal_distance, require_matching_plans

AUTO_MISSION, ARMED = 3, 2
MODE_NAMES = {2: 'POSCTL', 3: 'AUTO_MISSION', 4: 'Hold', 5: 'RTL', 18: 'LAND'}


def read_log(path):
    try:
        from pyulog import ULog
    except ImportError as exc:
        raise ValueError('Reading ULog files requires pyulog') from exc
    path = Path(path)
    if detect_log_format(path) != 'ulog':
        raise ValueError('PX4 mission analysis requires ULog')
    digest = sha256(path)
    # Full decoding is needed for pyulog's parameter-change timestamps.
    try:
        log = ULog(str(path))
    except (KeyError, IndexError, ValueError, struct.error) as exc:
        raise ValueError(f'Cannot decode ULog: {exc}') from exc
    if log.file_corruption:
        raise ValueError('ULog file corruption prevents mission validation')
    if sha256(path) != digest:
        raise ValueError('Flight log changed while reading')
    return log, digest


def validate_identity(log):
    from tools.scenario.capture_flight_parameters import _identify_log_firmware
    return _identify_log_firmware(dict(firmware=log.msg_info_dict), 'ulog', 'px4')


def topic(log, name, fields=(), *, optional=False):
    try:
        data = log.get_dataset(name, 0).data
    except (KeyError, IndexError):
        if optional:
            return []
        raise ValueError(f'ULog missing required topic: {name}') from None
    missing = {'timestamp', *fields} - data.keys()
    if missing:
        raise ValueError(f'ULog {name} missing required fields: {sorted(missing)}')
    rows = [{key: values[i].item() for key, values in data.items()}
            for i in range(len(data['timestamp']))]
    # Cached initial states can predate the ULog header. Keep them as state,
    # but never mistake their timestamps for a new mission execution.
    return sorted(rows, key=lambda row: row['timestamp'])


def _latest(rows, timestamp):
    return next((r for r in reversed(rows) if r['timestamp'] <= timestamp), None)


def _inside(rows, start, end):
    return [r for r in rows if start <= r['timestamp'] <= end]


def _flights(status):
    result, active = [], None
    for row in status:
        if row['arming_state'] == ARMED and active is None:
            armed_time = int(row.get('armed_time', 0))
            active = dict(arm_time_us=armed_time or row['timestamp'],
                          arm_time_confirmed=bool(armed_time), end_time_us=None)
        elif row['arming_state'] != ARMED and active is not None:
            active['end_time_us'] = row['timestamp']
            result.append(active)
            active = None
    if active is not None:
        result.append(active)
    return result


def _float32_close(actual, expected):
    # MISSION_ITEM_INT quantizes to 1e-7 degrees before float32 logging.
    rounded = struct.unpack('<f', struct.pack('<f', expected))[0]
    ulp = 2. ** (math.frexp(abs(rounded))[1] - 24) if rounded else 1e-7
    return math.isfinite(actual) and abs(actual - rounded) <= ulp + 1e-7


def _target_matches(row, item, home, triplets):
    p = item['params']
    if not (_float32_close(row['latitude'], p[4]) and _float32_close(row['longitude'], p[5])):
        return False
    altitude = row['altitude'] - (0 if row['altitude_is_relative'] else home[2])
    if not math.isclose(altitude, p[6], abs_tol=.05):
        return False
    # The float32 coordinates alone can differ by tens of centimetres. Require
    # an associated current setpoint with double precision as independent evidence.
    for sp in triplets:
        if abs(sp['timestamp'] - row['timestamp']) > 250_000 or not sp.get('current.valid'):
            continue
        if (horizontal_distance([sp['current.lat'], sp['current.lon']], p[4:6]) <= .2
                and math.isclose(sp['current.alt'] - home[2], p[6], abs_tol=.05)):
            return True
    return False


def compare_plan(items, native, results, triplets, commands, home):
    """Return mismatches; comparisons refer to original zero-based PX4 indices."""
    count = len(items)
    if any(r['seq_total'] != count for r in results):
        return ['mission item count differs']
    current = [r['seq_current'] for r in results]
    reached = [r['seq_reached'] for r in results]
    if (any(not 0 <= v <= count for v in current) or current != sorted(current)
            or any(not -1 <= v < count for v in reached) or reached != sorted(reached)):
        return ['mission_result progression is invalid or restarted']
    navigation_indices = {i for i, item in enumerate(items) if item['command'] in (16, 21, 22)}
    if not navigation_indices.issubset(set(current)):
        return ['mission_result does not confirm every navigation item']
    sequence = [r['sequence_current'] for r in native if r['nav_cmd'] != 0]
    dedup = [seq for i, seq in enumerate(sequence) if i == 0 or seq != sequence[i-1]]
    if dedup != list(range(count)):
        return [f'incomplete/skipped/restarted mission sequence: {dedup}']
    diffs = []
    for seq, item in enumerate(items):
        cmd, p = item['command'], item['params']
        rows = [r for r in native if r['sequence_current'] == seq and r['nav_cmd'] != 0]
        if cmd == 178:
            # navigator_mission_item does not preserve the seven DO parameters.
            # Match the actual command issued by Navigator at the same instant.
            observed = [c for c in commands if c['command'] == 178 and not c.get('from_external', False)
                        and any(abs(c['timestamp'] - r['timestamp']) <= 250_000
                                and r['nav_cmd'] == 178 for r in rows)]
            if not any(all(math.isclose(c[f'param{i+1}'], value, abs_tol=1e-5)
                           for i, value in enumerate(p)) for c in observed):
                diffs.append(f'item {seq}: CHANGE_SPEED parameters not confirmed')
            continue
        allowed = {22, 16} if cmd == 22 else ({21, 16} if cmd == 21 else {16})
        if cmd not in (16, 21, 22) or any(r['nav_cmd'] not in allowed for r in rows):
            diffs.append(f'item {seq}: command differs')
            continue
        # TAKEOFF is followed by a waypoint work item at its planned location.
        # LAND can be preceded by a horizontal approach at the current altitude.
        canonical = [r for r in rows if r['nav_cmd'] == (16 if cmd == 22 else cmd)
                     and r['altitude_is_relative']]
        if not canonical or not all(_target_matches(r, item, home, triplets)
                   and math.isclose(r.get('time_inside', math.nan), p[0], abs_tol=1e-5)
                   and (cmd != 16 or math.isclose(r.get('acceptance_radius', math.nan), p[1], abs_tol=1e-5))
                   and r.get('autocontinue', False) for r in canonical):
            diffs.append(f'item {seq}: target/altitude/parameters not confirmed')
        if cmd == 22 and not any(r['nav_cmd'] == 22 for r in rows):
            diffs.append(f'item {seq}: initial TAKEOFF not confirmed')
    return diffs


def _analyze_flight(flight, topics, log_start, log_end):
    start_bound = max(flight['arm_time_us'], log_start)
    end = flight['end_time_us'] or log_end
    status, native = topics['vehicle_status'], topics['navigator_mission_item']
    flown = [r for r in _inside(native, start_bound, end)
             if (s := _latest(status, r['timestamp'])) and s['arming_state'] == ARMED
             and s['nav_state'] == AUTO_MISSION and r['nav_cmd'] != 0]
    if not flown:
        raise ValueError('no fresh AUTO_MISSION execution markers')
    start = flown[0]['timestamp']
    flight['start_time_us'] = start
    if flown[0]['sequence_current'] != 0 or flown[0]['nav_cmd'] != 22:
        raise ValueError('mission start/TAKEOFF not observed; log starts mid-mission or mission was resumed')
    if flight['end_time_us'] is None:
        raise ValueError('disarm not observed; flight outcome is unknown')
    land = topics['vehicle_land_detected']
    initial_ground = _latest(land, start)
    airborne = [r for r in _inside(land, start, end) if not r['landed']]
    landed = [r for r in _inside(land, start, end) if r['landed']
              and airborne and r['timestamp'] > airborne[0]['timestamp']]
    last_land = _latest(land, end)
    flight['flight_outcome'] = ('landed_disarmed' if landed and last_land['landed'] else 'landing_unconfirmed')
    changes = []
    for row in _inside(status, start, end):
        if row['arming_state'] == ARMED and row['nav_state'] != AUTO_MISSION:
            if not changes or changes[-1]['nav_state'] != row['nav_state']:
                changes.append(dict(timestamp_us=row['timestamp'], nav_state=row['nav_state'],
                                    mode=MODE_NAMES.get(row['nav_state'], str(row['nav_state']))))
    if changes:
        flight['mode_changes'] = changes
        raise ValueError('interrupted AUTO_MISSION: ' + ' -> '.join(r['mode'] for r in changes)
                         + f'; {flight["flight_outcome"]}; interrupted replay is unsupported')
    if not initial_ground or not initial_ground['landed'] or not airborne or not landed or not last_land['landed']:
        raise ValueError('ground takeoff, airborne, landing and disarm sequence not confirmed')
    homes = topics['home_position']
    h = _latest(homes, start)
    if not h or not h['valid_hpos'] or not h['valid_alt']:
        raise ValueError('actual home unavailable at mission start')
    home = [h['lat'], h['lon'], h['alt']]
    if not all(math.isfinite(x) for x in home):
        raise ValueError('nonfinite home position')
    if any(not r['valid_hpos'] or not r['valid_alt'] or [r['lat'], r['lon'], r['alt']] != home
           for r in _inside(homes, start, end)):
        raise ValueError('home changed during flight')
    all_results = topics['mission_result']
    initial = _latest(all_results, start)
    results = _inside(all_results, start, end)
    if initial:
        results.insert(0, initial)
    if not results or any(not r['valid'] or r['failure'] for r in results):
        raise ValueError('valid mission_result unavailable or mission failure recorded')
    if len({r['mission_id'] for r in results}) != 1:
        raise ValueError('mission_id changed during flight')
    count = results[0]['seq_total']
    finishes = [r for r in results if r['timestamp'] >= start and r['finished']
                and r['seq_reached'] == count - 1]
    final_land = [r for r in flown if r['sequence_current'] == count - 1 and r['nav_cmd'] == 21]
    if final_land:
        finishes = [r for r in finishes if r['timestamp'] >= final_land[0]['timestamp']]
    if not final_land or not finishes or not any(r['timestamp'] >= final_land[0]['timestamp'] for r in landed):
        raise ValueError('final plan LAND, mission completion and subsequent landing not confirmed')
    flight.update(home_lla=home, mission_id=results[0]['mission_id'],
                  landing_time_us=next(r['timestamp'] for r in landed if r['timestamp'] > airborne[-1]['timestamp']), mission_finished_time_us=finishes[0]['timestamp'])
    return flown, results, home


def match_executions(plans, log):
    validate_identity(log)
    if log.dropouts:
        raise ValueError('ULog contains dropouts; complete mission progression cannot be confirmed')
    fields = {
        'vehicle_status': ('arming_state', 'nav_state'),
        'vehicle_land_detected': ('landed',),
        'home_position': ('lat', 'lon', 'alt', 'valid_hpos', 'valid_alt'),
        'mission_result': ('mission_id', 'seq_current', 'seq_reached', 'seq_total', 'valid', 'finished', 'failure'),
        'navigator_mission_item': ('sequence_current', 'nav_cmd', 'latitude', 'longitude', 'altitude',
                                   'altitude_is_relative', 'autocontinue', 'time_inside', 'acceptance_radius'),
        'position_setpoint_triplet': ('current.valid', 'current.lat', 'current.lon', 'current.alt'),
    }
    topics = {name: topic(log, name, required) for name, required in fields.items()}
    commands = topic(log, 'vehicle_command', optional=True)
    plans = {str(Path(p).resolve()): json.loads(Path(p).read_text())['mission']['items'] for p in plans}
    selected, excluded = [], []
    for flight in _flights(topics['vehicle_status']):
        try:
            native, results, home = _analyze_flight(flight, topics, log.start_timestamp, log.last_timestamp)
            triplets = _inside(topics['position_setpoint_triplet'], flight['start_time_us'], flight['end_time_us'])
            differences = {path: compare_plan(items, native, results, triplets, commands, home)
                           for path, items in plans.items()}
            matches = [path for path, diffs in differences.items() if not diffs]
            if not matches:
                flight['plan_differences'] = differences
                details = '; '.join(f'{Path(p).name}: {", ".join(d)}' for p, d in differences.items())
                raise ValueError(f'plan execution does not match: {details}')
            if len(matches) > 1:
                raise ValueError(f'ambiguous matching plans: {matches}')
            path = matches[0]
            flight.update(plan=path, executed_sequence=list(range(len(plans[path]))), status='landed_disarmed',
                          match=dict(basis='all plan items observed in PX4 execution; not an uploaded mission snapshot',
                                     horizontal_tolerance_m=.2, altitude_tolerance_m=.05,
                                     coordinate_evidence='float32 navigator items plus double position setpoints',
                                     frame_evidence='altitude_is_relative; navigator frame may be overwritten by origin',
                                     unspecified_yaw='plan null is unspecified'))
            selected.append(flight)
        except ValueError as exc:
            if 'ambiguous matching plans' in str(exc):
                raise
            excluded.append(dict(flight, reason=str(exc)))
    require_matching_plans(plans, selected, excluded)
    return selected, excluded


def parameter_at(log, name, timestamp):
    value = log.initial_parameters.get(name)
    for t, key, new_value in log.changed_parameters:
        if key == name and t <= timestamp:
            value = new_value
    if value is None or not math.isfinite(value):
        raise ValueError(f'{name}: no finite logged value at {timestamp}')
    return float(value)


def parameter_changed(log, name, start, end, value):
    return any(key == name and start < t <= end and new_value != value
               for t, key, new_value in log.changed_parameters)


def positions(log, flight):
    rows = _inside(topic(log, 'vehicle_global_position', ('lat', 'lon', 'alt')),
                   flight['start_time_us'], flight['end_time_us'])
    for counter in ('lat_lon_reset_counter', 'alt_reset_counter'):
        if len({r[counter] for r in rows if counter in r}) > 1:
            raise ValueError(f'PX4 global position {counter} changed during flight')
    result = []
    for row in rows:
        if not row.get('lat_lon_valid', True) or not row.get('alt_valid', True):
            continue
        point = (row['timestamp'], row['lat'], row['lon'], row['alt'] - flight['home_lla'][2])
        if not all(math.isfinite(x) for x in point):
            raise ValueError('Nonfinite PX4 global position')
        if result and point[0] == result[-1][0]:
            if point != result[-1]:
                raise ValueError('Conflicting PX4 global position timestamps')
            continue
        result.append(point)
    return result
