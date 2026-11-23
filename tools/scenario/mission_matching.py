"""Match candidate plans to observed mission revisions, including partial runs."""
from __future__ import annotations

import json
import math
from pathlib import Path

from tools.scenario.mission_log_validation import horizontal_distance


def item_matches(observation, item, autopilot):
    cmd, p, q = item['command'], item['params'], observation['params']
    if any(v is not None and (not isinstance(v,(int,float)) or not math.isfinite(v)) for v in p+q):
        return False
    allowed = {22, 16} if autopilot == 'px4' and cmd == 22 else {cmd}
    if observation['command'] not in allowed or not observation['target_confirmed']:
        return False
    if not observation['autocontinue'] or not item.get('autoContinue', True):
        return False
    if autopilot == 'ardupilot' and observation.get('frame') != item['frame']:
        return False
    if cmd == 178:
        return all(a is None or math.isclose(a, b, abs_tol=1e-5) for a,b in zip(p,q))
    if cmd not in (16, 21, 22) or horizontal_distance(p[4:6], q[4:6]) > .2 or abs(p[6]-q[6]) > .05:
        return False
    # PX4 navigator does not preserve all uploaded DO/yaw fields.
    indices = range(4) if autopilot == 'ardupilot' else ([0, 1] if cmd == 16 else [0])
    return all(p[i] is None or math.isclose(p[i], q[i], abs_tol=1e-5) for i in indices)


def match_timeline(timeline, paths, autopilot):
    plans = {str(Path(p).resolve()): json.loads(Path(p).read_text())['mission']['items'] for p in paths}
    used = set()
    for flight in timeline['flights']:
        groups = {}
        for seg in flight['segments']:
            if seg['mode'] == 'MISSION' and seg['observations']:
                groups.setdefault(seg['revision'], []).append(seg)
        for revision, segments in groups.items():
            observations = [o for s in segments for o in s['observations'] if o['canonical']]
            revision_events = [e for e in flight['events'] if e['kind'] == 'revision' and e['revision'] == revision]
            cached = flight.get('initial_revision')
            if cached and cached['revision'] == revision:
                revision_events.insert(0, cached)
            # Cached revisions can precede arm: progress rows also carry the count.
            counts = {e['count'] for e in revision_events}
            counts.update(p['count'] for s in segments for p in s['progress'])
            candidates = []
            differences = {}
            for path, items in plans.items():
                bad = []
                if counts and counts != {len(items)}:
                    bad.append('mission item count differs')
                if not observations:
                    bad.append('no canonical target evidence')
                for o in observations:
                    seq = o['sequence']
                    if not 0 <= seq < len(items) or not item_matches(o, items[seq], autopilot):
                        bad.append(f'item {seq}: command/target/parameters differ or are unconfirmed')
                if autopilot == 'ardupilot':
                    from tools.scenario.ardupilot_mission import compare_plan
                    for event in revision_events:
                        bad.extend(compare_plan(items, event['snapshot']))
                differences[path] = list(dict.fromkeys(bad))
                if not bad:
                    candidates.append(path)
            coverage = sorted({o['sequence'] for o in observations})
            for seg in segments:
                seg['match'] = dict(candidates=candidates, observed_sequences=sorted({o['sequence'] for o in seg['observations'] if o['canonical']}),
                    revision_coverage=coverage, status='matched_observed' if len(candidates)==1 else 'ambiguous' if candidates else 'unmatched',
                    basis='observed execution targets; unobserved plan items are not verified', differences=differences)
                if len(candidates)==1:
                    seg['plan'] = candidates[0]
                    used.add(candidates[0])
                seg['source_sequence_start'] = seg['observations'][0]['source_sequence']
                reached = [r['reached'] for r in seg['progress'] if r['reached'] >= 0]
                seg['last_reached'] = reached[-1] if reached else None
                # Completion and match coverage are independent. A selected LAND
                # must be the final item and precede landing in this segment;
                # stale finished flags must never certify a later execution.
                canonical = [o for o in seg['observations'] if o['canonical']]
                last = canonical[-1] if canonical else None
                # The LAND target must belong to the same airborne interval as
                # its touchdown; a later flight in this arm interval is unrelated.
                landings = [a['end_time_us'] for a in flight['airborne_intervals']
                            if last and a['end_time_us'] is not None
                            and a['start_time_us'] <= last['timestamp_us'] <= a['end_time_us']
                            and a['end_time_us'] <= seg['end_time_us']]
                landed_at = landings[0] if len(landings) == 1 else None
                terminal_land = bool(last and last['target_confirmed'] and last['command']==21
                    and counts == {last['sequence']+1} and landed_at is not None
                    and last['timestamp_us'] <= landed_at <= seg['end_time_us'])
                seg['terminal_plan_land'] = terminal_land
                seg['landing_time_us'] = landed_at if terminal_land else None
                seg['mission_failure_observed'] = any(r['failure'] for r in seg['progress'])
                finished = terminal_land and (autopilot == 'ardupilot' or any(
                    r.get('finished') and r['reached']==r['count']-1 and r['valid'] and not r['failure']
                    and r['timestamp_us'] >= last['timestamp_us'] for r in seg['progress']))
                seg['mission_outcome'] = ('completed' if finished else 'interrupted' if seg['mission_failure_observed'] else 'unknown'
                    if terminal_land or (seg is flight['segments'][-1] and
                        (flight['end_time_us'] is None or (last and last['command'] == 21)))
                    else 'interrupted')
        flight['mission_segments'] = [s['id'] for s in flight['segments'] if s['mode']=='MISSION' and s['observations']]
        flight['mission_executions'] = mission_executions(flight)
    timeline['unused_plans'] = sorted(set(plans)-used)
    return timeline


def mission_executions(flight):
    """Join Hold/resume segments; preserve separate replacements and restarts."""
    executions = []
    for seg in flight['segments']:
        if seg['mode'] != 'MISSION' or not seg['observations']:
            continue
        if (not executions or seg['revision'] != executions[-1]['revision']
                or seg.get('plan') != executions[-1]['plan'] or seg['boundary']=='sequence_rewind'):
            executions.append(dict(id=f'mission_{len(executions):03d}', revision=seg['revision'],
                plan=seg.get('plan'), start_time_us=seg['start_time_us'], segment_ids=[],
                observed_sequences=[], last_reached=None, outcome='interrupted', mission_failure_observed=False))
        execution = executions[-1]
        execution['segment_ids'].append(seg['id'])
        execution['mission_failure_observed'] |= seg['mission_failure_observed']
        execution['end_time_us'] = seg['end_time_us']
        execution['observed_sequences'] = sorted(set(execution['observed_sequences'])
            | set(seg['match']['observed_sequences']))
        if seg['last_reached'] is not None:
            execution['last_reached'] = seg['last_reached']
        if execution['outcome'] != 'completed':
            execution['outcome'] = seg['mission_outcome']
    return executions
