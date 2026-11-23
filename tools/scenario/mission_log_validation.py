"""Shared mission geometry and native BIN access for comparison tools."""
from __future__ import annotations

import json
import math
from pathlib import Path


def read_log(path):
    """Legacy BIN interface used by the native ArduPilot comparison tools."""
    from tools.scenario.ardupilot_mission import read_log as read_bin
    return read_bin(path)


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
