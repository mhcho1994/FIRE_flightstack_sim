"""Compare matched real/SITL trajectories per flight, using shared N/E origins.

Time zero is the mission-start marker in each source. No trajectory rotation,
translation fitting, time warping, extrapolation or cross-flight interpolation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


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
    records, sources, seen = [], [], set()
    for path in flight_logs(logs, collected_only=True):
        rows, digest = read_log(path)
        if digest in seen:
            continue
        seen.add(digest)
        records.extend(rows)
        sources.append(dict(path=str(path), sha256=digest))
    if not records:
        raise ValueError('No collected SITL BIN logs; finish the launcher or explicitly select --sitl-source telemetry')
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
        markers[phase['id']]=dict(start_time_us=start,end_time_us=end)
    return trajectories, dict(kind='BIN POS', sources=sources, markers=markers)


def compare_run(run_dir, output=None, sitl_source="bin", *, details=False):
    if details and sitl_source != 'bin':
        raise ValueError('--details requires --sitl-source bin')
    run_dir = Path(run_dir).resolve()
    scenario_path = run_dir/'scenario.yaml'
    scenario_bytes = scenario_path.read_bytes()
    from tools.commander.mission_session import comparison_session
    session = comparison_session(yaml.safe_load(scenario_bytes))
    reference = run_dir/session['reference_trajectory']
    if hashlib.sha256(reference.read_bytes()).hexdigest() != session['reference_sha256']:
        raise ValueError('Reference trajectory SHA256 mismatch')
    logs = run_dir/'ardu_logs'
    result_path = logs/'session_result.json'
    if not result_path.exists():
        result_path = logs/'replay_result.json'
    result = json.loads(result_path.read_text())
    if result['status'] != 'completed' or result['scenario_sha256'] != hashlib.sha256(scenario_bytes).hexdigest():
        raise ValueError('SITL replay is incomplete or belongs to a different scenario')
    real = read_trajectories(reference)
    if sitl_source == 'bin':
        sitl, source_info = bin_trajectories(logs, session, result)
    elif sitl_source == 'telemetry':
        sitl = read_trajectories(logs/('session_trajectory.csv' if result_path.name == 'session_result.json' else 'replay_trajectory.csv'))
        source_info = dict(kind='MAVLink GLOBAL_POSITION_INT', time_zero='last position timestamp before AUTO request')
    else:
        raise ValueError('sitl_source must be bin or telemetry')
    ids = [p['id'] for p in session['phases']]
    if set(real) != set(ids) or set(sitl) != set(ids) or [p['id'] for p in result['phases']] != ids:
        raise ValueError('Reference/SITL phase mapping is incomplete')
    output = Path(output).resolve() if output else run_dir/'comparison'
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(ids), 3, figsize=(15, 4*len(ids)), squeeze=False, constrained_layout=True)
    summary = dict(alignment='mission start per phase; fixed shared horizontal origin; height above flight home',
        source_log_sha256=session['source_log_sha256'], scenario_sha256=result['scenario_sha256'], phases=[],
        limitations=session['limitations'], sitl_source=source_info)
    for i, phase in enumerate(session['phases']):
        key = phase['id']; r, s = real[key], sitl[key]
        metrics, paired = compare_phase(r, s)
        summary['phases'].append(dict(id=key, plan=Path(phase['plan']).name, **metrics))
        np.savetxt(output/f'{key}_paired.csv', paired, delimiter=',', comments='',
            header='time_s,real_n,real_e,real_up,sitl_n,sitl_e,sitl_up,error_n,error_e,error_up')
        ax = axes[i, 0]
        ax.plot(r[:, 2], r[:, 1], label='Real'); ax.plot(s[:, 2], s[:, 1], label='SITL')
        ax.set(xlabel='East [m]', ylabel='North [m]', title=f'{key}: {Path(phase["plan"]).name}')
        ax.set_aspect('equal', adjustable='datalim')
        ax = axes[i, 1]
        ax.plot(r[:, 0], r[:, 3], label='Real'); ax.plot(s[:, 0], s[:, 3], label='SITL')
        ax.set(xlabel='Time from mission start [s]', ylabel='Height above home [m]')
        ax = axes[i, 2]
        ax.plot(paired[:, 0], np.linalg.norm(paired[:, -3:], axis=1), label='Position error')
        ax.set(xlabel='Time from mission start [s]', ylabel='Position error [m]',
               title=f'RMSE {metrics["position_rmse_m"]:.2f} m; overlap only')
        for ax in axes[i]:
            ax.grid(alpha=.3); ax.legend()
    fig.savefig(output/'trajectory_comparison.png', dpi=160)
    fig.savefig(output/'trajectory_comparison.svg')
    plt.close(fig)
    if details:
        from tools.plotting.flight_replay_details import write_detail_plots
        summary['details'] = write_detail_plots(output, session, real, sitl, source_info)
    (output/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sitl-source', choices=('bin','telemetry'), default='bin', help='Native BIN MISE/POS timing (default), or runtime MAVLink samples')
    parser.add_argument('--details', action='store_true', help='Also save 3D waypoint and velocity plots; requires native BIN PSCN/PSCE/PSCD logs')
    args = parser.parse_args()
    compare_run(args.run_dir, args.output, args.sitl_source, details=args.details)


if __name__ == '__main__':
    main()
