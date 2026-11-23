"""Compare real/SITL positions and logged velocities using shared N/E origins.

Time zero is the mission-start marker in each source. No trajectory rotation,
translation fitting, time warping, extrapolation or cross-flight interpolation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

def read_trajectories(path):
    result = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            result.setdefault(row['phase'], []).append([float(row[k]) for k in ('time_s', 'north_m', 'east_m', 'up_m')])
    cleaned = {}
    for phase, rows in result.items():
        array = np.asarray(rows, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError(f'{phase}: nonfinite trajectory')
        array = array[np.argsort(array[:, 0], kind='stable')]
        # Retain the last sample at repeated timestamps.
        keep = np.r_[array[1:, 0] != array[:-1, 0], True]
        cleaned[phase] = array[keep]
    return cleaned


def compare_phase(real, sitl):
    if len(real) < 2 or len(sitl) < 2:
        raise ValueError('At least two trajectory samples per source are required')
    start, end = max(real[0, 0], sitl[0, 0]), min(real[-1, 0], sitl[-1, 0])
    times = real[(real[:, 0] >= start) & (real[:, 0] <= end), 0]
    if len(times) < 2 or end <= start:
        raise ValueError('No common time interval for trajectory comparison')
    ref = np.column_stack([np.interp(times, real[:, 0], real[:, i]) for i in range(1, 4)])
    sim = np.column_stack([np.interp(times, sitl[:, 0], sitl[:, i]) for i in range(1, 4)])
    delta = sim-ref
    metrics = dict(real_duration_s=float(real[-1, 0]), sitl_duration_s=float(sitl[-1, 0]),
        overlap_start_s=float(start), overlap_end_s=float(end), samples=len(times),
        position_rmse_m=float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
        horizontal_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:, :2]**2, axis=1)))),
        vertical_rmse_m=float(np.sqrt(np.mean(delta[:, 2]**2))),
        max_position_error_m=float(np.max(np.linalg.norm(delta, axis=1))))
    return metrics, np.column_stack([times, ref, sim, delta])



def bin_trajectories(logs, session, result):
    """Use native MISE/POS/ARM timestamps after the launcher has flushed logs."""
    from tools.launcher.attempt_logs import flight_logs
    from tools.scenario.mission_log_validation import read_log
    from tools.scenario.import_real_flight_mission import lla_to_ned
    records, sources, seen, by_digest = [], [], set(), {}
    for path in flight_logs(logs, collected_only=True):
        rows, digest = read_log(path)
        if digest in seen:
            continue
        seen.add(digest)
        records.extend(rows)
        by_digest[digest] = rows
        sources.append(dict(path=str(path), sha256=digest))
    if not records:
        raise ValueError('No collected SITL BIN logs; finish the launcher to collect native position/velocity logs')
    trajectories, markers = {}, {}
    for phase in result['phases']:
        lower = phase['sitl_start_time_us']
        upper = phase['sitl_end_time_us'] + 3_000_000
        starts = [r['TimeUS'] for r in records if r['mavpackettype'] == 'MISE'
                  and r['CNum'] == 1 and r['CId'] == 22 and lower <= r['TimeUS'] <= lower + 5_000_000]
        if not starts:
            raise ValueError(f'{phase["id"]}: no native mission-start marker near the runtime start')
        start = min(starts)
        ends = [r['TimeUS'] for r in records if r['mavpackettype'] == 'ARM' and not r['ArmState']
                and start < r['TimeUS'] <= upper]
        if not ends:
            raise ValueError(f'{phase["id"]}: no native disarm marker')
        end = min(ends)
        rows = []
        for r in records:
            if r['mavpackettype'] == 'POS' and start <= r['TimeUS'] <= end:
                n, e, _ = lla_to_ned([r['Lat'],r['Lng'],r['RelHomeAlt']], session['target_origin_lla'])
                rows.append([(r['TimeUS']-start)/1e6,n,e,r['RelHomeAlt']])
        rows.sort(key=lambda r:r[0])
        if len(rows)<2 or not np.isfinite(rows).all():
            raise ValueError(f'{phase["id"]}: insufficient or invalid native POS samples')
        times=[r[0] for r in rows]
        if len(times)!=len(set(times)):
            raise ValueError('Overlapping SITL logs have ambiguous boot timestamps')
        trajectories[phase['id']]=np.asarray(rows)
        matching = [digest for digest, rows in by_digest.items()
                    if any(r['mavpackettype'] == 'MISE' and r['TimeUS'] == start for r in rows)
                    and any(r['mavpackettype'] == 'ARM' and not r['ArmState'] and r['TimeUS'] == end for r in rows)]
        if len(matching) != 1:
            raise ValueError('Native BIN flight markers do not identify one source log')
        markers[phase['id']]=dict(start_time_us=start,end_time_us=end,source_sha256=matching[0])
    return trajectories, dict(kind='BIN POS', sources=sources, markers=markers)


def compare_run(run_dir, output=None, sitl_source="auto"):
    run_dir = Path(run_dir).resolve()
    scenario_path = run_dir/'scenario.yaml'
    scenario_bytes = scenario_path.read_bytes()
    from tools.commander.mission_session import comparison_session, normalize_session, session_block
    data = yaml.safe_load(scenario_bytes)
    autopilot = session_block(data)['autopilot']
    session = normalize_session(data, autopilot) if autopilot=='px4' else comparison_session(data)
    if sitl_source == 'auto':
        sitl_source = 'ulog' if autopilot=='px4' else 'bin'
    if (sitl_source=='ulog' and autopilot!='px4') or (sitl_source=='bin' and autopilot!='ardupilot'):
        raise ValueError('Native SITL source does not match scenario autopilot')
    reference = run_dir/session['reference_trajectory']
    if hashlib.sha256(reference.read_bytes()).hexdigest() != session['reference_sha256']:
        raise ValueError('Reference trajectory SHA256 mismatch')
    logs = run_dir/('px4_logs' if autopilot=='px4' else 'ardu_logs')
    result_path = logs/'session_result.json'
    if not result_path.exists():
        result_path = logs/'replay_result.json'
    result = json.loads(result_path.read_text())
    if result['status'] != 'completed' or result['scenario_sha256'] != hashlib.sha256(scenario_bytes).hexdigest():
        raise ValueError('SITL replay is incomplete or belongs to a different scenario')
    real = read_trajectories(reference)
    if sitl_source == 'bin':
        sitl, source_info = bin_trajectories(logs, session, result)
    elif sitl_source == 'ulog':
        sitl, source_info = ulog_trajectories(logs, session, result)
    elif sitl_source == 'telemetry':
        sitl = read_trajectories(logs/('session_trajectory.csv' if result_path.name == 'session_result.json' else 'replay_trajectory.csv'))
        for p in session['phases']:
            if p.get('replay_scope') == 'mission_segment':
                a = sitl[p['id']]
                sitl[p['id']] = a[(a[:, 0] >= 0) & (a[:, 0] < p['comparison_duration_s'])]
        source_info = dict(kind='MAVLink GLOBAL_POSITION_INT', time_zero='last position timestamp before AUTO request')
    else:
        raise ValueError('sitl_source must be auto, bin, ulog or telemetry')
    ids = [p['id'] for p in session['phases']]
    if set(real) != set(ids) or set(sitl) != set(ids) or [p['id'] for p in result['phases']] != ids:
        raise ValueError('Reference/SITL phase mapping is incomplete')
    from tools.plotting.flight_replay_state import session_velocities, write_state_comparison
    velocity_info = source_info
    if sitl_source == 'telemetry':
        # The runtime CSV stores positions only. Read native velocities on the
        # same request clock, with each ULog still bound to its selected flight.
        native_reader = ulog_trajectories if autopilot == 'px4' else bin_trajectories
        _, velocity_info = native_reader(logs, session, result)
        for phase, runtime in zip(session['phases'], result['phases']):
            bounds = velocity_info['markers'][phase['id']]
            bounds['start_time_us'] = runtime['sitl_start_time_us']
            if phase.get('replay_scope') == 'mission_segment':
                bounds['end_time_us'] = bounds['start_time_us'] + round(phase['comparison_duration_s'] * 1e6)
                bounds['end_exclusive'] = True
        velocity_info['time_zero'] = source_info['time_zero']
    velocities, velocity_summary = session_velocities(session, velocity_info)
    output = Path(output).resolve() if output else run_dir/'comparison'
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    summary = dict(alignment='mission start per phase; fixed shared horizontal origin; height above flight home',
        source_log_sha256=session['source_log_sha256'], scenario_sha256=result['scenario_sha256'], phases=[],
        limitations=session['limitations'], sitl_source=source_info)
    paired_by_phase = {}
    for phase in session['phases']:
        key = phase['id']; r, s = real[key], sitl[key]
        metrics, paired = compare_phase(r, s)
        paired_by_phase[key] = paired
        runtime = next(p for p in result['phases'] if p['id'] == key)
        summary['phases'].append(dict(id=key, plan=Path(phase['plan']).name,
            replay_scope=phase.get('replay_scope', 'flight'), initial_state=phase.get('initial_state'),
            initialization=runtime.get('initialization'), comparison_duration_s=phase.get('comparison_duration_s'), **metrics))
        np.savetxt(output/f'{key}_paired.csv', paired, delimiter=',', comments='',
            header='time_s,real_n,real_e,real_up,sitl_n,sitl_e,sitl_up,error_n,error_e,error_up')
    summary['state_comparison'] = write_state_comparison(output, session, real, sitl, paired_by_phase, velocities)
    summary['state_comparison']['velocity'] = velocity_summary
    if autopilot=='px4':
        summary['trajectory_3d'] = write_timeline_details(output, session, real, sitl)
    else:
        from tools.plotting.flight_replay_details import write_detail_plots
        summary['trajectory_3d'] = write_detail_plots(output, session, real, sitl, velocities)
    from tools.plotting.flight_replay_diagnostics import write_diagnostic_plots
    summary['diagnostics'] = write_diagnostic_plots(output, session, velocity_info, velocity_summary)
    if autopilot == 'px4':
        from tools.plotting.flight_replay_progress import write_progress_plots
        summary['progress_analysis'] = write_progress_plots(output, session, result, velocity_summary, real, sitl)
    summary['runtime_events'] = {p['id']:dict(events=p.get('events',[]),divergences=p.get('divergences',[])) for p in result['phases']}
    (output/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    # Remove only the superseded generated figures after the new outputs succeed.
    for stem in ('trajectory_comparison', 'speed_comparison'):
        for suffix in ('.png', '.svg'):
            (output / (stem + suffix)).unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))
    return summary



def ulog_trajectories(logs, session, result):
    """Locate each phase in exactly one native boot clock; never join log files."""
    from tools.launcher.attempt_logs import flight_logs
    from tools.scenario import px4_mission as px4
    from tools.scenario.import_real_flight_mission import lla_to_ned
    decoded, seen = [], set()
    for path in flight_logs(logs, collected_only=True):
        log,digest = px4.read_log(path)
        if digest in seen:
            continue
        seen.add(digest)
        decoded.append((path,log,digest,px4.topic(log,'vehicle_status')))
    trajectories, markers, sources = {}, {}, []
    for phase in result['phases']:
        definition = next((p for p in session.get('phases', []) if p['id'] == phase['id']), {})
        segment = definition.get('replay_scope') == 'mission_segment'
        matches = []
        for path,log,digest,status in decoded:
            for flight in px4._flights(status):
                modes = [r for r in status if r['arming_state']==2 and r['nav_state']==3
                         and r['timestamp']>=max(flight['arm_time_us'],log.start_timestamp)
                         and (flight['end_time_us'] is None or r['timestamp']<flight['end_time_us'])]
                if not modes or flight['end_time_us'] is None:
                    continue
                start=modes[0]['timestamp'];end=flight['end_time_us']
                if segment:
                    entries = [r for i, r in enumerate(status) if r['arming_state'] == 2 and r['nav_state'] == 3
                               and (i == 0 or status[i-1]['arming_state'] != 2 or status[i-1]['nav_state'] != 3)
                               and flight['arm_time_us'] <= r['timestamp'] < flight['end_time_us']
                               and -250_000 <= r['timestamp']-phase['sitl_start_time_us'] <= 5_000_000]
                    if len(entries) != 1:
                        continue
                    start = entries[0]['timestamp']
                    end = start + round(definition['comparison_duration_s']*1e6)
                    if end > flight['end_time_us'] or any(
                            r['arming_state'] != 2 or r['nav_state'] != 3 for r in status if start <= r['timestamp'] < end):
                        raise ValueError('SITL left MISSION before the selected comparison window ended')
                    items = [r for r in px4.topic(log, 'navigator_mission_item')
                             if start <= r['timestamp'] <= start+500_000]
                    if not items or items[0]['sequence_current'] != definition['start_source_sequence']:
                        raise ValueError('Native initial mission sequence differs from the selected segment')
                if abs(start-phase['sitl_start_time_us'])>5_000_000 or abs(flight['end_time_us']-phase['sitl_end_time_us'])>3_000_000:
                    continue
                h=px4._latest(px4.topic(log,'home_position'),start)
                if not h or not h['valid_hpos'] or not h['valid_alt']:
                    raise ValueError('SITL home unavailable at mission start')
                observed=dict(start_time_us=start,end_time_us=end,home_lla=[h['lat'],h['lon'],h['alt']])
                positions=px4.positions(log,observed)
                if segment:
                    positions = [p for p in positions if p[0] < end]
                    if not positions or positions[0][0]-start > 500_000 or end-positions[-1][0] > 500_000:
                        raise ValueError('Incomplete native coverage of the selected comparison window')
                rows=[]
                for t,lat,lon,up in positions:
                    n,e,_=lla_to_ned([lat,lon,up],session['target_origin_lla'])
                    rows.append([(t-start)/1e6,n,e,up])
                if len(rows)<2 or any(b[0]-a[0]>1. for a,b in zip(rows,rows[1:])):
                    raise ValueError('Insufficient native positions or >1 s position gap')
                matches.append((rows,path,digest,start,end,log))
        if len(matches)!=1:
            raise ValueError(f"{phase['id']}: expected one native ULog flight, found {len(matches)}")
        rows,path,digest,start,end,log=matches[0]
        trajectories[phase['id']]=np.asarray(rows)
        markers[phase['id']]=dict(start_time_us=start,end_time_us=end,source_sha256=digest,end_exclusive=segment,
            runtime_anchor_offset_s=(phase['sitl_start_time_us']-start)/1e6)
        duplicate_counts = {}
        for topic_name in ('vehicle_global_position', 'vehicle_local_position'):
            try:
                data = log.get_dataset(topic_name).data
            except (KeyError, IndexError):
                continue
            stamps = data['timestamp']
            selected = stamps[(stamps >= start) & (stamps <= end) & ((stamps < end) if segment else True)]
            duplicate_counts[topic_name] = int(np.count_nonzero(np.diff(selected) == 0))
        sources.append(dict(path=str(path),sha256=digest,dropouts=len(log.dropouts),
                            duplicate_state_publications=duplicate_counts))
    return trajectories,dict(kind='ULog vehicle_global_position',sources=sources,markers=markers,
        time_zero='selected native AUTO_MISSION transition; fixed-duration windows exclude setup and cleanup',
        duplicate_timestamp_policy='Latest of increasing timestamp_sample values at a repeated publication timestamp; ambiguous conflicts rejected')


def write_timeline_details(output, session, real, sitl):
    import matplotlib.pyplot as plt
    from tools.scenario.import_real_flight_mission import lla_to_ned
    ids=[p['id'] for p in session['phases']]
    fig=plt.figure(figsize=(9,7*len(ids)),layout='constrained')
    for index,phase in enumerate(session['phases']):
        ax=fig.add_subplot(len(ids),1,index+1,projection='3d')
        for label,arr in [('Real',real[phase['id']]),('SITL',sitl[phase['id']])]:
            ax.plot(arr[:,2],arr[:,1],arr[:,3],label=label,lw=1.8)
        definitions=phase.get('mission_definitions') or {phase['plan']:phase['mission_items']}
        for name,items in definitions.items():
            pts=[]
            labels=[]
            for item in items:
                if item['command'] not in (16,21,22):continue
                n,e,_=lla_to_ned([item['p5'],item['p6'],item['p7']],session['target_origin_lla'])
                pts.append([e,n,item['p7']])
                point=np.array([e,n,item['p7']])
                nearby=next((group for group in labels if np.linalg.norm(group[0]-point)<2.),None)
                if nearby is None:
                    labels.append([point,[str(item.get('source_sequence',''))]])
                else:
                    nearby[1].append(str(item.get('source_sequence','')))
            for point,sequences in labels:
                ax.text(point[0],point[1],point[2]+.6,' / '.join(sequences),fontsize=8)
            pts=np.asarray(pts)
            ax.plot(pts[:,0],pts[:,1],pts[:,2],'o--',color='gray',alpha=.55,label=f'Plan: {Path(name).name}')
        ax.set(xlabel='East [m]',ylabel='North [m]',zlabel='Height above home [m]',title=phase['id']+' — full plan includes unexecuted items')
        ax.legend();ax.view_init(elev=27,azim=-60)
    fig.savefig(output/'trajectory_3d.png',dpi=180);fig.savefig(output/'trajectory_3d.svg');plt.close(fig)
    return dict(trajectory='trajectory_3d.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sitl-source', choices=('auto','bin','ulog','telemetry'), default='auto', help='Position timing: native BIN/ULog by autopilot, or runtime MAVLink; logged velocities require native logs in all modes')
    args = parser.parse_args()
    compare_run(args.run_dir, args.output, args.sitl_source)


if __name__ == '__main__':
    main()
