"""Decode ArduPilot CMD snapshots, MISE starts and flight-state evidence.

The common timeline layer matches plans and classifies completed or interrupted
executions; a stored CMD snapshot does not prove that its items were executed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from tools.scenario.flight_log import detect_log_format, sha256
from tools.scenario.mission_log_validation import horizontal_distance


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


def positions(records, flight):
    return [(r['TimeUS'], r['Lat'], r['Lng'], r['RelHomeAlt']) for r in records
            if r['mavpackettype'] == 'POS'
            and flight['start_time_us'] <= r['TimeUS'] <= flight['end_time_us']]


def decode_events(records):
    """Translate native BIN facts; CMD snapshots and MISE starts remain distinct."""
    import hashlib
    events, pending = [], []
    last_revision = None
    armed = False
    for r in records:
        if 'TimeUS' not in r:
            continue
        t, kind = r['TimeUS'], r['mavpackettype']
        def add(kind, **data):
            events.append(dict(timestamp_us=t, kind=kind, **data))
        if kind == 'ARM':
            armed = bool(r['ArmState'])
            add('state', armed=armed, armed_time_us=t if armed else 0)
        elif kind == 'MODE':
            number = r['ModeNum']
            native_mode = {0: 'STABILIZE', 1: 'ACRO', 2: 'ALT_HOLD', 3: 'AUTO',
                           4: 'GUIDED', 5: 'LOITER', 6: 'RTL', 9: 'LAND', 16: 'POSHOLD'}.get(number, f'MODE:{number}')
            add('state', mode={3: 'MISSION', 5: 'HOLD', 6: 'RTL', 9: 'LAND'}.get(number, native_mode),
                native_mode=native_mode, mode_number=number)
        elif kind == 'ORGN' and r['Type'] == 1:
            add('home', home=[r['Lat'], r['Lng'], r['Alt']])
        elif kind == 'EV' and r['Id'] in (18, 28):
            # Native land-detector transitions, not height above home. In
            # particular, a positive POS altitude must never undo LAND_COMPLETE.
            add('land', landed=r['Id'] == 18,
                evidence='EV.LAND_COMPLETE' if r['Id'] == 18 else 'EV.NOT_LANDED')
        elif kind == 'CMD':
            if r['CNum'] == 0:
                pending = []
            pending.append(r)
            if len(pending) == r['CTot'] and [p['CNum'] for p in pending] == list(range(r['CTot'])):
                commands = [{k: v for k,v in p.items() if k not in ('TimeUS','mavpackettype')} for p in pending[1:]]
                revision = 'ap:' + hashlib.sha256(json.dumps(commands, sort_keys=True).encode()).hexdigest()[:16]
                if revision != last_revision:
                    add('revision', revision=revision, count=len(commands), snapshot=commands)
                    last_revision = revision
                pending = []
        elif kind == 'MISE' and r['CNum'] > 0:
            add('item', sequence=r['CNum']-1, source_sequence=r['CNum'], command=r['CId'],
                params=[r.get(f'Prm{i}', 0.) for i in range(1,5)] + [r['Lat'],r['Lng'],r['Alt']],
                frame=r['Frame'], canonical=True, target_confirmed=True, autocontinue=True)
    times = [e['timestamp_us'] for e in events]
    return dict(events=events, start_us=min(times, default=0), end_us=max(times, default=0), issues=[])
