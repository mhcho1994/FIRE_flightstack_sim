"""Shared mission validation and compatibility entry points for BIN callers."""
from __future__ import annotations

import json
import math
from pathlib import Path

from tools.scenario.flight_log import sha256


def require_matching_plans(plans, selected, excluded=()):
    """Every explicit plan must have a confirmed, complete execution."""
    missing = {str(Path(p).resolve()) for p in plans} - {s['plan'] for s in selected}
    if missing:
        reasons = '; '.join(dict.fromkeys(r['reason'] for r in excluded if r.get('reason')))
        detail = f'. Observed flights: {reasons}' if reasons else ''
        raise ValueError(f'No complete matching execution for: {sorted(missing)}{detail}')


def read_log(path):
    """Legacy BIN interface used by the native ArduPilot comparison tools."""
    from tools.scenario.ardupilot_mission import read_log as read_bin
    return read_bin(path)


def compare_plan(items, commands, *args, **kwargs):
    from tools.scenario.ardupilot_mission import compare_plan as compare_bin
    return compare_bin(items, commands, *args, **kwargs)


def match_executions(plans, records):
    from tools.scenario.ardupilot_mission import match_executions as match_bin
    return match_bin(plans, records)


def parameter_at(records, name, timestamp):
    from tools.scenario.ardupilot_mission import parameter_at as bin_parameter_at
    return bin_parameter_at(records, name, timestamp)


def horizontal_distance(a, b):
    return 6371000 * math.hypot(math.radians(a[0] - b[0]),
        math.cos(math.radians(b[0])) * math.radians(a[1] - b[1]))


def mission_items(plan, source_home, target_home, autopilot):
    from tools.scenario.import_real_flight_mission import lla_to_ned
    from tools.scenario.scenario_generator import ned_to_lla
    result = []
    if autopilot == 'ardupilot':
        # ArduPilot reserves index zero for home. Preserve source CNum indexing.
        result.append(dict(command=16, frame=0, autocontinue=1,
            p1=0, p2=0, p3=0, p4=0, p5=target_home[0], p6=target_home[1], p7=target_home[2]))
    for item in json.loads(Path(plan).read_text())['mission']['items']:
        p = list(item['params'])
        if item['command'] in (16, 21, 22):
            p[4:7] = ned_to_lla(lla_to_ned(p[4:7], source_home), target_home)
        result.append(dict(command=item['command'], frame=item['frame'], autocontinue=1,
                           **{f'p{i+1}': value for i, value in enumerate(p)}))
    return result
