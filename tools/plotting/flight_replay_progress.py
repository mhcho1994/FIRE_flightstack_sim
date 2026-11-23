"""PX4 mission progress and controller reference diagnostics for replay comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.scenario.import_real_flight_mission import lla_to_ned
from tools.scenario.px4_mission import parameter_at, read_log, validate_identity

PARAMETERS = ('MPC_XY_CRUISE', 'NAV_ACC_RAD', 'NAV_MC_ALT_RAD', 'MPC_XY_TRAJ_P',
              'MPC_ACC_DOWN_MAX', 'MPC_THR_HOVER', 'THR_MDL_FAC')
TARGET_HEADER = ('timestamp_us,time_s,nav_state,logged_seq_current,logged_seq_reached,'
                 'mission_valid,setpoint_type,valid,latitude_deg,longitude_deg,'
                 'altitude_amsl_m,north_m,east_m,height_above_home_m')


def dataset(log, name):
    try:
        return log.get_dataset(name, 0).data
    except (KeyError, IndexError):
        return {}


def latest(data, timestamp):
    """No future backfill when a topic starts after the requested time."""
    index = int(np.searchsorted(data.get('timestamp', []), timestamp, side='right')) - 1
    return index if index >= 0 else None


def value(data, field, index):
    return float(data[field][index]) if index is not None and field in data else np.nan


def window(data, bounds, *, carry=False):
    stamps = np.asarray(data.get('timestamp', []))
    start, end = bounds['start_time_us'], bounds['end_time_us']
    inside = (stamps >= start) & (stamps <= end)
    if bounds.get('end_exclusive'):
        inside &= stamps < end
    indices = np.flatnonzero(inside).tolist()
    prior = latest(data, start)
    if carry and prior is not None and (not indices or prior < indices[0]):
        indices.insert(0, prior)
    return indices


def changes(data, fields, bounds):
    events, previous = [], None
    for i in window(data, bounds, carry=True):
        vals = [value(data, f, i) for f in fields]
        vals = [v if np.isfinite(v) else None for v in vals]
        if vals != previous:
            stamp = int(data['timestamp'][i])
            events.append(dict(time_s=max(0, stamp - bounds['start_time_us']) / 1e6,
                               source_time_us=stamp, **dict(zip(fields, vals))))
            previous = vals
    return events


def active_targets(log, bounds, origin, home_alt):
    sp, mode, mission = [dataset(log, name) for name in
                        ('position_setpoint_triplet', 'vehicle_status', 'mission_result')]
    rows = []
    for i in window(sp, bounds, carry=True):
        stamp = int(sp['timestamp'][i])
        effective = max(stamp, bounds['start_time_us'])
        si, mi = latest(mode, effective), latest(mission, effective)
        lat, lon, alt = [value(sp, 'current.' + f, i) for f in ('lat', 'lon', 'alt')]
        valid = value(sp, 'current.valid', i) == 1
        n, e = (lla_to_ned([lat, lon, 0], origin)[:2]
                if valid and np.isfinite([lat, lon]).all() else (np.nan, np.nan))
        rows.append([stamp, (effective - bounds['start_time_us']) / 1e6,
                     value(mode, 'nav_state', si), value(mission, 'seq_current', mi),
                     value(mission, 'seq_reached', mi), value(mission, 'valid', mi),
                     value(sp, 'current.type', i), int(valid), lat, lon, alt,
                     n, e, alt - home_alt if valid else np.nan])
    return np.asarray(rows, dtype=float).reshape(-1, 14)


def local_reference(data, local, bounds, origin, home_alt):
    """Translate local NED using the last observed estimator reference per sample."""
    rows = []
    for i in window(data, bounds):
        stamp = data['timestamp'][i]
        li = latest(local, stamp)
        lat, lon, alt = [value(local, f, li) for f in ('ref_lat', 'ref_lon', 'ref_alt')]
        n, e = (lla_to_ned([lat, lon, 0], origin)[:2]
                if np.isfinite([lat, lon]).all() else (np.nan, np.nan))
        if 'xy_global' in local and value(local, 'xy_global', li) != 1:
            n, e = np.nan, np.nan
        if 'z_global' in local and value(local, 'z_global', li) != 1:
            alt = np.nan
        rows.append([(int(stamp) - bounds['start_time_us']) / 1e6,
                     value(data, 'position[0]', i) + n, value(data, 'position[1]', i) + e,
                     alt - home_alt - value(data, 'position[2]', i)])
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def selected_logs(info):
    """Bind phases to hashes; overlapping boot clocks never identify a log."""
    logs, sources = {}, {}
    for source in info['sources']:
        digest = source['sha256']
        if digest not in logs:
            log, actual = read_log(source['path'])
            if actual != digest:
                raise ValueError(f"Progress source SHA256 mismatch: {source['path']}")
            logs[digest], sources[digest] = log, source
    selected = {}
    for key, bounds in info['markers'].items():
        digest = bounds.get('source_sha256')
        if digest is None and len(logs) == 1:
            digest = next(iter(logs))
        if digest not in logs:
            raise ValueError(f'{key}: progress requires one identified native ULog')
        selected[key] = (logs[digest], sources[digest], bounds)
    return selected


def write_progress_plots(output, session, runtime, velocity_info, real, sitl):
    """Optional PX4 topics supplement, never replace, the state comparison."""
    import matplotlib.pyplot as plt

    output = Path(output)
    selected = {name: selected_logs(velocity_info[name]) for name in ('real', 'sitl')}
    report = dict(
        alignment='Same phase time origins and bounds as state comparison; no time rescaling',
        caveats=[
            'Mission indices can be retained during HOLD/RTL; they do not identify the active navigation goal.',
            'Topic joins use the last known sample, never future observations; publications are asynchronous.',
            'Trajectory positions are moving references, not mission waypoint destinations.',
            'Local NED references are translated using logged estimator reference coordinates.',
            'LAND altitude is retained in CSV but hidden in the plot because landing ignores that target altitude.',
            'Parameter/build differences are observations, not isolated causal explanations.',
            'Missing optional topics or reference fields remain unavailable; no reconstructed setpoints.'],
        phases=[])
    phases = session['phases']
    fig, axes = plt.subplots(4 * len(phases), 2, figsize=(16, 12 * len(phases)),
                             squeeze=False, layout='constrained')
    for pi, phase in enumerate(phases):
        key = phase['id']
        execution = next(p for p in runtime['phases'] if p['id'] == key)
        entry = dict(id=key, mission_item_map=execution.get('mission_item_map'),
                     runtime_divergences=execution.get('divergences', []), sources={})
        for col, name in enumerate(('real', 'sitl')):
            log, source, bounds = selected[name][key]
            start, end = bounds['start_time_us'], bounds['end_time_us']
            duration = (end - start) / 1e6
            origin = session['source_origin_lla' if name == 'real' else 'target_origin_lla']
            home = dataset(log, 'home_position')
            home_alt = value(home, 'alt', latest(home, start))
            lp, ts, cs = [dataset(log, t) for t in
                          ('vehicle_local_position', 'trajectory_setpoint', 'vehicle_local_position_setpoint')]
            modes = changes(dataset(log, 'vehicle_status'), ['nav_state'], bounds)
            progress = changes(dataset(log, 'mission_result'), ['seq_current', 'seq_reached', 'valid'], bounds)
            targets = active_targets(log, bounds, origin, home_alt)
            smooth = local_reference(ts, lp, bounds, origin, home_alt)
            csv_name = f'{key}_{name}_active_target.csv'
            np.savetxt(output / csv_name, targets, delimiter=',', comments='', header=TARGET_HEADER)
            reference_name = f'{key}_{name}_trajectory_reference.csv'
            np.savetxt(output / reference_name, smooth, delimiter=',', comments='',
                       header='time_s,north_m,east_m,height_above_home_m')
            params = {}
            for parameter in PARAMETERS:
                try:
                    params[parameter] = parameter_at(log, parameter, start)
                except ValueError:
                    params[parameter] = None
            identity = validate_identity(log)
            identity['branch'] = log.msg_info_dict.get('ver_sw_branch')
            source_report = dict(**source, bounds=bounds, firmware=identity,
                                 modes=modes, mission_progress=progress,
                                 parameters_at_start=params, active_target_csv=csv_name,
                                 trajectory_reference_csv=reference_name,
                                 navigator_samples=len(targets), trajectory_samples=len(smooth),
                                 translated_position_available=bool(np.isfinite(smooth[:, 1:]).any()),
                                 horizontal_speed={})
            entry['sources'][name] = source_report
            actual = (real if name == 'real' else sitl)[key]
            axcol = axes[pi * 4:pi * 4 + 4, col]
            plotted = targets.copy()
            if len(plotted):
                plotted = np.vstack([plotted, plotted[-1]])
                plotted[-1, 1] = duration
                plotted[plotted[:, 6] == 4, 13] = np.nan
            for j, ax in enumerate(axcol[:3]):
                ax.plot(actual[:, 0], actual[:, j + 1], color='#1871ad', label='Actual')
                if len(plotted):
                    ax.step(plotted[:, 1], plotted[:, j + 11], where='post',
                            color='#dc701b', label='Navigator target (including HOLD/RTL)')
                if len(smooth):
                    ax.plot(smooth[:, 0], smooth[:, j + 1], '--', color='#43964c',
                            label='Trajectory reference')
                ax.set_ylabel(('North [m]', 'East [m]', 'Height above home [m]')[j])
                if not len(plotted) or not np.isfinite(smooth[:, j + 1]).any():
                    ax.text(.99, .02, 'Some setpoint data unavailable', ha='right',
                            transform=ax.transAxes, fontsize=8)
            speed_csvs = []
            for data, fields, label, color, style, stem in (
                (lp, ('vx', 'vy'), 'Actual', '#1871ad', '-', 'actual'),
                (ts, ('velocity[0]', 'velocity[1]'), 'Trajectory feedforward', '#43964c', '--', 'trajectory'),
                (cs, ('vx', 'vy'), 'Controller velocity target', '#dc701b', ':', 'controller')):
                indices = window(data, bounds)
                values = np.array([[(int(data['timestamp'][i]) - start) / 1e6,
                                    value(data, fields[0], i), value(data, fields[1], i)]
                                   for i in indices], dtype=float).reshape(-1, 3)
                available = bool(np.isfinite(values[:, 1:]).all(axis=1).any())
                source_report['horizontal_speed'][stem] = dict(available=available, samples=len(values))
                if available:
                    axcol[3].plot(values[:, 0], np.hypot(values[:, 1], values[:, 2]),
                                  label=label, color=color, linestyle=style)
                csv = f'{key}_{name}_{stem}_horizontal_velocity.csv'
                np.savetxt(output / csv, values, delimiter=',', comments='', header='time_s,v_north_m_s,v_east_m_s')
                speed_csvs.append(csv)
            source_report['horizontal_velocity_csvs'] = speed_csvs
            axcol[3].set(ylabel='Horizontal speed [m/s]', xlabel='Time from comparison phase start [s]')
            if not all(v['available'] for v in source_report['horizontal_speed'].values()):
                axcol[3].text(.99, .02, 'Some velocity setpoint data unavailable', ha='right',
                              transform=axcol[3].transAxes, fontsize=8)
            axcol[0].set_title(f'{key}: {name.upper()}')
            for ax in axcol:
                ax.grid(alpha=.25)
                ax.set_xlim(0, max(duration, .001))
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=7, loc='upper left')
                for i, event in enumerate(modes):
                    if event['nav_state'] == 4:
                        stop = modes[i + 1]['time_s'] if i + 1 < len(modes) else duration
                        ax.axvspan(event['time_s'], stop, color='gray', alpha=.13)
                    if event['nav_state'] == 5:
                        ax.axvline(event['time_s'], color='#904cac', ls=':', alpha=.6)
        # Matching axes make side-by-side comparison meaningful without changing time.
        for j in range(4):
            pair = axes[pi * 4 + j]
            lower = min(ax.get_ylim()[0] for ax in pair)
            upper = max(ax.get_ylim()[1] for ax in pair)
            for ax in pair:
                ax.set_ylim(lower, upper)
        report['phases'].append(entry)
    fig.suptitle('Actual state, Navigator target, trajectory reference and controller velocity target\n'
                 'Gray: HOLD; purple line: RTL entry')
    for ext in ('png', 'svg'):
        fig.savefig(output / f'setpoint_tracking.{ext}', dpi=160)
    plt.close(fig)
    (output / 'progress_analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return dict(figure='setpoint_tracking.png', vector_figure='setpoint_tracking.svg',
                report='progress_analysis.json', phases=report['phases'])
