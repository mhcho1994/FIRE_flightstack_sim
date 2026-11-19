"""Match explicit plans to complete ArduCopter executions, never by filename.

CMD snapshots describe stored missions; MISE starts items. ARM and landed events
bound flights. An interrupted or ambiguous execution is deliberately not replayed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tools.scenario.flight_log import detect_log_format, sha256
from tools.scenario.mission_log_validation import horizontal_distance, require_matching_plans


def read_log(path):
    from pymavlink import DFReader
    path = Path(path).resolve(strict=True)
    if detect_log_format(path) != 'bin':
        raise ValueError('ArduPilot mission analysis requires a BIN log, not ULog')
    before = sha256(path)
    records = []
    log = DFReader.DFReader_binary(str(path))
    try:
        while True:
            msg = log.recv_match(type=['ARM', 'MODE', 'CMD', 'MISE', 'EV', 'ORGN', 'POS', 'PARM', 'MSG', 'VER'], strict=True)
            if msg is None:
                break
            records.append(msg.to_dict())
    finally:
        log.close()
    if sha256(path) != before:
        raise ValueError('Flight log changed while reading')
    return records, before


def compare_plan(items, commands, horizontal_tolerance_m=0.2, altitude_tolerance_m=0.05):
    """Compare stored values, allowing coordinate quantization and unspecified yaw."""
    rows = [row for row in commands if row['CNum'] != 0]
    if [r['CNum'] for r in rows] != list(range(1, len(items) + 1)):
        return ['mission item count/sequence differs']
    diffs = []
    for index, (item, row) in enumerate(zip(items, rows), 1):
        if row['CId'] != item['command'] or row['Frame'] != item['frame']:
            diffs.append(f'item {index}: command/frame differs')
            continue
        p = item['params']
        fields = ('Lat', 'Lng', 'Alt', 'Prm1', 'Prm2', 'Prm3')
        if any(not math.isfinite(row[k]) for k in fields):
            diffs.append(f'item {index}: nonfinite logged mission value')
            continue
        if item['command'] in (16, 21, 22):
            if horizontal_distance((row['Lat'], row['Lng']), p[4:6]) > horizontal_tolerance_m:
                diffs.append(f'item {index}: horizontal position differs')
            if abs(row['Alt'] - p[6]) > altitude_tolerance_m:
                diffs.append(f'item {index}: altitude differs')
        for j in range(4):
            if p[j] is not None and not math.isclose(row[f'Prm{j+1}'], p[j], abs_tol=1e-5):
                diffs.append(f'item {index}: parameter {j+1} differs')
    return diffs


def match_executions(plans, records):
    """Return complete selected flights and an audit of other armed intervals."""
    plans = {str(Path(path).resolve()): json.loads(Path(path).read_text())['mission']['items'] for path in plans}
    snapshots, pending, flights, active = [], [], [], None
    for row in records:
        kind = row['mavpackettype']
        if kind == 'CMD':
            if row['CNum'] == 0:
                pending = []
            pending.append(row)
            if len(pending) == row['CTot'] and [r['CNum'] for r in pending] == list(range(row['CTot'])):
                snapshots.append(list(pending))
                pending = []
        if kind == 'ARM':
            if row['ArmState'] and active is None:
                active = dict(arm_time_us=row['TimeUS'], rows=[])
            elif not row['ArmState'] and active is not None:
                active['end_time_us'] = row['TimeUS']
                flights.append(active)
                active = None
        if active is not None:
            active['rows'].append(row)
    if active:
        active['end_time_us'] = None
        flights.append(active)
    selected, excluded = [], []
    for flight in flights:
        starts = [r for r in flight['rows'] if r['mavpackettype'] == 'MISE']
        if not starts:
            excluded.append(dict(arm_time_us=flight['arm_time_us'], reason='no mission execution'))
            continue
        start = starts[0]['TimeUS']
        candidates = [s for s in snapshots if s[-1]['TimeUS'] <= start]
        if not candidates:
            raise ValueError(f'Mission at {start}: no complete CMD snapshot')
        commands = candidates[-1]
        matches = [path for path, items in plans.items() if not compare_plan(items, commands)]
        if not matches:
            excluded.append(dict(arm_time_us=flight['arm_time_us'], reason='mission does not match supplied plans'))
            continue
        if len(matches) != 1:
            raise ValueError(f'Mission at {start}: ambiguous matching plans {matches}')
        path = matches[0]
        last = len(plans[path])
        rows = flight['rows']
        modes = [r for r in records if r['mavpackettype'] == 'MODE' and r['TimeUS'] <= start]
        if not modes or modes[-1]['ModeNum'] != 3:
            raise ValueError(f'{Path(path).name} at {start}: AUTO execution not confirmed')
        if (flight['end_time_us'] is None or starts[-1]['CNum'] != last or starts[-1]['CId'] != 21
                or not any(r['mavpackettype'] == 'EV' and r['Id'] == 18 and r['TimeUS'] >= starts[-1]['TimeUS'] for r in rows)):
            raise ValueError(f'{Path(path).name} at {start}: no confirmed final LAND, landed event and disarm; interrupted replay is unsupported')
        seq = [r['CNum'] for r in starts]
        dedup = [v for i, v in enumerate(seq) if i == 0 or v != seq[i-1]]
        if dedup != list(range(1, last + 1)):
            raise ValueError(f'{Path(path).name} at {start}: skipped/restarted mission items {dedup}')
        if any(r['mavpackettype'] == 'MODE' and r['TimeUS'] >= start and r['ModeNum'] != 3 for r in rows):
            raise ValueError(f'{Path(path).name} at {start}: interrupted AUTO mode')
        by_seq = {r['CNum']: r for r in commands}
        for r in starts:
            original = by_seq.get(r['CNum'], {})
            if any(r.get(k) != original.get(k) for k in ('CTot', 'CId', 'Frame', 'Lat', 'Lng', 'Alt', 'Prm1', 'Prm2', 'Prm3')):
                raise ValueError(f'{Path(path).name} at {start}: mission changed during flight')
        homes = [r for r in records if r['mavpackettype'] == 'ORGN' and r['Type'] == 1 and r['TimeUS'] <= start]
        if not homes:
            raise ValueError(f'Mission at {start}: actual home is unavailable')
        h = homes[-1]
        selected.append(dict(plan=path, arm_time_us=flight['arm_time_us'], start_time_us=start,
            end_time_us=flight['end_time_us'], home_lla=[h['Lat'], h['Lng'], h['Alt']],
            executed_sequence=dedup, status='landed_disarmed',
            match=dict(horizontal_tolerance_m=0.2, altitude_tolerance_m=0.05,
                       unspecified_yaw='plan null is unspecified; firmware stored value is not compared')))
    require_matching_plans(plans, selected, excluded)
    return selected, excluded


def parameter_at(records, name, timestamp):
    rows = [r for r in records if r['mavpackettype'] == 'PARM' and r['Name'] == name and r['TimeUS'] <= timestamp]
    if not rows:
        raise ValueError(f'{name}: no logged value at {timestamp}')
    return float(rows[-1]['Value'])



def validate_identity(records):
    from tools.scenario.capture_flight_parameters import _identify_log_firmware
    history = dict(firmware=[r['Message'] for r in records if r['mavpackettype'] == 'MSG'],
                   firmware_versions=[r for r in records if r['mavpackettype'] == 'VER'])
    return _identify_log_firmware(history, 'bin', 'ardupilot')


def parameter_changed(records, name, start, end, value):
    return any(r['mavpackettype'] == 'PARM' and r['Name'] == name
               and start < r['TimeUS'] <= end and r['Value'] != value for r in records)


def positions(records, flight):
    return [(r['TimeUS'], r['Lat'], r['Lng'], r['RelHomeAlt']) for r in records
            if r['mavpackettype'] == 'POS'
            and flight['start_time_us'] <= r['TimeUS'] <= flight['end_time_us']]
