"""Decode native PX4 mission evidence for common timeline analysis.

ULog records observed work items rather than a complete uploaded mission snapshot.
Internal climb and landing work items remain distinct from canonical plan targets.
"""
from __future__ import annotations

import math
from pathlib import Path
import struct

from tools.scenario.flight_log import detect_log_format, sha256

AUTO_MISSION, ARMED = 3, 2


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


def parameter_at(log, name, timestamp):
    value = log.initial_parameters.get(name)
    for t, key, new_value in log.changed_parameters:
        if key == name and t <= timestamp:
            value = new_value
    if value is None or not math.isfinite(value):
        raise ValueError(f'{name}: no finite logged value at {timestamp}')
    return float(value)


def unique_state_publications(rows, fields, label):
    """Resolve a repeated publication only with distinct increasing sample clocks.

    PX4 lockstep can publish consecutive estimator samples at the same clock
    tick. Keep the latest sample at that tick; conflicting same-sample states
    or missing sample-clock evidence remain errors.
    """
    result = []
    for row in rows:
        values = tuple(row[f] for f in fields)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f'Nonfinite {label} estimate')
        if result and row['timestamp'] == result[-1]['timestamp']:
            previous = result[-1]
            same = values == tuple(previous[f] for f in fields)
            ordered = 0 < previous.get('timestamp_sample', 0) < row.get('timestamp_sample', 0) <= row['timestamp']
            if not same and not ordered:
                raise ValueError(f'Conflicting {label} timestamps')
            result[-1] = row
        else:
            result.append(row)
    return result


def positions(log, flight):
    rows = _inside(topic(log, 'vehicle_global_position', ('lat', 'lon', 'alt')),
                   flight['start_time_us'], flight['end_time_us'])
    for counter in ('lat_lon_reset_counter', 'alt_reset_counter'):
        if len({r[counter] for r in rows if counter in r}) > 1:
            raise ValueError(f'PX4 global position {counter} changed during flight')
    rows = unique_state_publications([r for r in rows if r.get('lat_lon_valid', True) and r.get('alt_valid', True)],
                                     ('lat', 'lon', 'alt'), 'PX4 global position')
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


def decode_events(log):
    """Expose native evidence without requiring a completed mission."""
    events, issues = [], []
    if log.dropouts:
        issues.append('ULog contains dropouts')
    names = ('vehicle_status', 'vehicle_land_detected', 'home_position', 'mission_result',
             'navigator_mission_item', 'position_setpoint_triplet', 'vehicle_command')
    rows = {n: topic(log, n, optional=True) for n in names}
    for n in names[:-1]:
        if not rows[n]:
            issues.append(f'missing topic: {n}')
    def add(row, kind, **data):
        events.append(dict(timestamp_us=row['timestamp'], kind=kind, **data))
    for r in rows['vehicle_status']:
        add(r, 'state', armed=r['arming_state'] == ARMED, armed_time_us=r.get('armed_time', 0),
            mode={3: 'MISSION', 4: 'HOLD', 5: 'RTL', 18: 'LAND'}.get(r['nav_state'], f"MANUAL:{r['nav_state']}"))
    for r in rows['vehicle_land_detected']:
        add(r, 'land', landed=bool(r['landed']))
    for r in rows['home_position']:
        if r['valid_hpos'] and r['valid_alt']:
            add(r, 'home', home=[r['lat'], r['lon'], r['alt']])
        else:
            add(r, 'home_invalid')
    last_revision = None
    for r in rows['mission_result']:
        revision = f"px4:{r['mission_id']}:{r['seq_total']}"
        if revision != last_revision:
            add(r, 'revision', revision=revision, count=r['seq_total'], valid=bool(r['valid']))
            last_revision = revision
        add(r, 'progress', current=r['seq_current'], reached=r['seq_reached'],
            finished=bool(r['finished']), failure=bool(r['failure']), valid=bool(r['valid']),
            count=r['seq_total'], revision=revision)
    for r in rows['navigator_mission_item']:
        if not r['nav_cmd']:
            continue
        h = _latest(rows['home_position'], r['timestamp'])
        # Retain internal work items, but only compare canonical navigation targets.
        canonical = bool(r['altitude_is_relative'] and r['nav_cmd'] in (16, 21))
        params = [r['time_inside'], r['acceptance_radius'], 0., None,
                  r['latitude'], r['longitude'], r['altitude']]
        confirmed = False
        if h and canonical:
            for sp in rows['position_setpoint_triplet']:
                if abs(sp['timestamp'] - r['timestamp']) <= 250_000 and sp.get('current.valid'):
                    if (_float32_close(r['latitude'], sp['current.lat']) and
                        _float32_close(r['longitude'], sp['current.lon']) and
                        abs(sp['current.alt'] - h['alt'] - r['altitude']) <= .05):
                        params[4:6] = [sp['current.lat'], sp['current.lon']]
                        confirmed = True
                        break
        if r['nav_cmd'] == 178:
            issued = [c for c in rows['vehicle_command'] if c['command'] == 178
                      and not c.get('from_external', False) and abs(c['timestamp']-r['timestamp']) <= 250_000]
            canonical, confirmed = True, bool(issued)
            if issued:
                params = [issued[0][f'param{i}'] for i in range(1, 8)]
        add(r, 'item', sequence=r['sequence_current'], source_sequence=r['sequence_current'],
            command=r['nav_cmd'], params=params, canonical=canonical, target_confirmed=confirmed,
            autocontinue=bool(r['autocontinue']), home=[h['lat'], h['lon'], h['alt']] if h else None)
    # Stable sorting places cached state before fresh observations at equal timestamps.
    return dict(events=events, start_us=log.start_timestamp, end_us=log.last_timestamp, issues=issues)
