"""Native actuator commands and observed mode/mission-item timelines."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np

from tools.plotting.flight_replay_details import COLORS

PX4_MODES = {0: 'MANUAL', 1: 'ALTCTL', 2: 'POSCTL', 3: 'MISSION', 4: 'HOLD',
             5: 'RTL', 6: 'POSITION_SLOW', 8: 'ALTITUDE_CRUISE', 10: 'ACRO',
             12: 'DESCEND', 13: 'TERMINATION', 14: 'OFFBOARD', 15: 'STAB',
             17: 'TAKEOFF', 18: 'LAND', 19: 'FOLLOW_TARGET', 20: 'PRECLAND',
             21: 'ORBIT', 22: 'VTOL_TAKEOFF'}
AP_MODES = {0: 'STABILIZE', 1: 'ACRO', 2: 'ALT_HOLD', 3: 'AUTO', 4: 'GUIDED',
            5: 'LOITER', 6: 'RTL', 7: 'CIRCLE', 9: 'LAND', 16: 'POSHOLD',
            17: 'BRAKE', 18: 'THROW', 20: 'GUIDED_NOGPS', 21: 'SMART_RTL'}


def step_series(times, values, bounds):
    """Hold observed discrete states; preserve their source timestamps in CSV.

    The last known state at the window start is carried in. Never backfill from
    a future observation. NaN means explicitly invalid/unknown, not item zero.
    """
    start, end = bounds['start_time_us'], bounds['end_time_us']
    exclusive = bounds.get('end_exclusive', False)
    rows = sorted(zip(times, values), key=lambda r: r[0])
    before = [r for r in rows if r[0] <= start]
    selected = ([(start, before[-1][1], before[-1][0])] if before else [])
    selected += [(t, v, t) for t, v in rows if start < t <= end and (not exclusive or t < end)]
    compact = []
    for t, v, observed in selected:
        if compact and t == compact[-1][0]:
            compact[-1] = (t, v, observed)
        elif not compact or not np.array_equal(v, compact[-1][1], equal_nan=True):
            compact.append((t, v, observed))
    if compact and compact[-1][0] < end:
        compact.append((end, compact[-1][1], compact[-1][2]))
    result = np.asarray(compact, dtype=float).reshape(-1, 3)
    result[:, 0] = (result[:, 0] - start) / 1e6
    return result


def command_stream(data, bounds, fields, *, topic, instance=0, units, omit_zero=False):
    times = np.asarray(data['timestamp'])
    mask = (times >= bounds['start_time_us']) & (times <= bounds['end_time_us'])
    if bounds.get('end_exclusive'):
        mask &= times < bounds['end_time_us']
    if mask.sum() < 2:
        return None
    columns, values = [], []
    for field in fields:
        v = np.asarray(data[field], dtype=float)[mask]
        finite = np.isfinite(v)
        if finite.any() and (not omit_zero or np.any(v[finite] != 0)):
            columns.append(field)
            values.append(np.where(finite, v, np.nan))
    if not columns:
        return None
    array = np.column_stack([(times[mask].astype(float) - bounds['start_time_us']) / 1e6, *values])
    array = array[np.argsort(array[:, 0], kind='stable')]
    return dict(topic=topic, instance=instance, units=units, columns=columns, values=array)


def ulog_diagnostics(log, bounds):
    from tools.scenario.px4_mission import topic

    commands = []
    # Normalized thrust commands have consistent semantics across PWM, DShot,
    # UAVCAN and simulation. Raw actuator_outputs are device-specific units.
    for name, prefix, units in [('actuator_motors', 'control', 'Normalized thrust [-1, 1]'),
                               ('actuator_outputs', 'output', 'Native output units')]:
        for dataset in sorted(log.data_list, key=lambda d: d.multi_id):
            if dataset.name != name:
                continue
            fields = sorted((f for f in dataset.data if re.fullmatch(prefix + r'\[\d+\]', f)),
                            key=lambda f: int(f.split('[')[1][:-1]))
            data = dict(dataset.data)
            if name == 'actuator_outputs' and 'noutputs' in data:
                for field in fields:
                    channel = int(field.split('[')[1][:-1])
                    data[field] = np.where(data['noutputs'] > channel, data[field], np.nan)
            stream = command_stream(data, bounds, fields, topic=name, instance=dataset.multi_id,
                                    units=units, omit_zero=name == 'actuator_outputs')
            if stream:
                commands.append(stream)
        if commands:
            break
    status = topic(log, 'vehicle_status', optional=True)
    mission = topic(log, 'mission_result', optional=True)
    mode = step_series([r['timestamp'] for r in status], [r.get('nav_state', np.nan) for r in status], bounds)
    waypoint = step_series([r['timestamp'] for r in mission],
        [r.get('seq_current', np.nan) if r.get('valid', False) and r.get('seq_current', -1) >= 0
         else np.nan for r in mission], bounds)
    return dict(commands=commands, mode=mode, waypoint=waypoint, mode_names=PX4_MODES,
                mode_source='vehicle_status.nav_state', waypoint_source='mission_result.seq_current',
                waypoint_notes='Native zero-based mission item, including non-waypoint items; '
                               'retained during HOLD/RTL; not proof the item is executing')


def bin_diagnostics(records, bounds):
    mode = [r for r in records if r['mavpackettype'] == 'MODE']
    mission = [r for r in records if r['mavpackettype'] == 'MISE']
    commands = []
    outputs = [r for r in records if r['mavpackettype'] == 'RCOU']
    if outputs:
        fields = sorted({f for r in outputs for f in r if re.fullmatch(r'C\d+', f)}, key=lambda f: int(f[1:]))
        data = {f: np.asarray([r.get(f, np.nan) for r in outputs]) for f in fields}
        data['timestamp'] = np.asarray([r['TimeUS'] for r in outputs])
        stream = command_stream(data, bounds, fields, topic='RCOU', units='Output PWM [us]', omit_zero=True)
        if stream:
            commands.append(stream)
    return dict(commands=commands,
        mode=step_series([r['TimeUS'] for r in mode], [r.get('ModeNum', np.nan) for r in mode], bounds),
        waypoint=step_series([r['TimeUS'] for r in mission],
                             [r['CNum'] if r.get('CNum', 0) > 0 else np.nan for r in mission], bounds),
        mode_names=AP_MODES, mode_source='MODE.ModeNum', waypoint_source='MISE.CNum',
        waypoint_notes='Native mission item number including non-waypoint items; home is item 0. '
                       'Last started item is retained outside AUTO, not proof it is executing')


def read_diagnostics(sources, markers, kind):
    decoded = {}
    for source in sources:
        digest = source['sha256']
        if digest in decoded:
            continue
        if kind == 'ULog vehicle_global_position':
            from tools.scenario.px4_mission import read_log
            log, actual = read_log(source['path'])
        else:
            from pymavlink import DFReader
            path = Path(source['path'])
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"Diagnostic source SHA256 mismatch: {path}")
            reader = DFReader.DFReader_binary(str(path))
            log = []
            try:
                while (message := reader.recv_match(type=['MODE', 'MISE', 'RCOU'], strict=True)) is not None:
                    log.append(message.to_dict())
            finally:
                reader.close()
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Diagnostic source SHA256 mismatch: {source['path']}")
        decoded[digest] = log
    result = {}
    for key, bounds in markers.items():
        digest = bounds.get('source_sha256')
        if digest is None and len(decoded) == 1:
            digest = next(iter(decoded))
        if digest not in decoded:
            raise ValueError(f'{key}: diagnostics require one identified native flight log')
        decode = ulog_diagnostics if kind == 'ULog vehicle_global_position' else bin_diagnostics
        result[key] = decode(decoded[digest], bounds)
    return result


def write_diagnostic_plots(output, session, source_info, velocity_info):
    """Always emit figures; explicitly annotate optional missing log topics."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output = Path(output)
    diagnostics = {name: read_diagnostics(velocity_info[name]['sources'], velocity_info[name]['markers'],
                                          source_info['kind']) for name in ('real', 'sitl')}
    phases = session['phases']
    commands_fig, command_axes = plt.subplots(len(phases), 2, figsize=(15, 4 * len(phases)),
                                              squeeze=False, layout='constrained')
    timeline_fig, timeline_axes = plt.subplots(2 * len(phases), 1, figsize=(15, 6 * len(phases)),
                                               squeeze=False, layout='constrained')
    summary = dict(command_figure='command_comparison.png', timeline_figure='mission_timeline.png',
                   alignment='Same phase time origins and bounds as state comparison; observed log states',
                   state_sampling='Step hold, including the last state known at time zero; no future backfill. '
                                  'CSV source_time_us identifies the original observation for held endpoints.',
                   command_notes='Normalized motor thrust preferred for PX4. Raw fallback uses device-specific units; '
                                 'no PWM conversion or assumption that channel indices match physical motors. '
                                 'NaN commands remain gaps; entirely inactive channels are omitted.',
                   phases=[])
    for index, phase in enumerate(phases):
        key = phase['id']
        entry = dict(id=key)
        duration = max((velocity_info[name]['markers'][key]['end_time_us'] -
                        velocity_info[name]['markers'][key]['start_time_us']) / 1e6 for name in ('real', 'sitl'))
        mode_ax, waypoint_ax = timeline_axes[2 * index:2 * index + 2, 0]
        modes = {}
        for col, name in enumerate(('real', 'sitl')):
            label = 'Real' if name == 'real' else 'SITL'
            item = diagnostics[name][key]
            ax = command_axes[index, col]
            command_info = []
            for stream in item['commands']:
                values = stream['values']
                for j, field in enumerate(stream['columns']):
                    ax.plot(values[:, 0], values[:, j + 1], lw=.85,
                            label=f"{stream['instance']}:{field}")
                filename = f"{key}_{name}_{stream['topic']}_{stream['instance']}.csv"
                np.savetxt(output / filename, values, delimiter=',', comments='',
                           header=','.join(['time_s', *stream['columns']]))
                command_info.append({k: v for k, v in stream.items() if k != 'values'} |
                                    dict(csv=filename, samples=len(values)))
            if command_info:
                ax.set_ylabel(command_info[0]['units'])
                ax.legend(ncol=2, fontsize=8)
                topic_names = ', '.join(sorted({s['topic'] for s in command_info}))
            else:
                topic_names = 'command log unavailable'
                ax.text(.5, .5, 'No usable command trace in this window', transform=ax.transAxes, ha='center')
            ax.set(title=f'{key} | {label} | {topic_names}', xlabel='Time from mission start [s]',
                   xlim=(0, duration))
            ax.grid(alpha=.25)
            entry[name] = dict(commands=command_info, mode_source=item['mode_source'],
                               waypoint_source=item['waypoint_source'], waypoint_notes=item['waypoint_notes'])
            for field, target in [('mode', mode_ax), ('waypoint', waypoint_ax)]:
                series = item[field]
                available = bool(len(series) and np.isfinite(series[:, 1]).any())
                entry[name][field + '_available'] = available
                if len(series):
                    np.savetxt(output / f'{key}_{name}_{field}.csv', series, delimiter=',', comments='',
                               header=f'time_s,{field}_id,source_time_us')
                if available:
                    target.step(series[:, 0], series[:, 1], where='post', label=label, color=COLORS[name],
                                ls='-' if name == 'real' else '--', lw=1.8)
                else:
                    target.plot([], [], label=f'{label}: log unavailable', color=COLORS[name])
                if field == 'mode':
                    for value in series[:, 1]:
                        if np.isfinite(value):
                            modes[int(value)] = item['mode_names'].get(int(value), f'MODE_{int(value)}')
            # No inferred mode/waypoint is substituted from the planned event schedule.
        # Matching units need matching scales for meaningful visual comparison.
        streams = [stream for name in ('real', 'sitl') for stream in diagnostics[name][key]['commands']]
        units = {stream['units'] for stream in streams}
        if len(units) == 1 and streams:
            values = np.concatenate([stream['values'][:, 1:].ravel() for stream in streams])
            values = values[np.isfinite(values)]
            if 'Normalized thrust [-1, 1]' in units:
                limits = (-1.05 if values.min() < 0 else -.05, 1.05)
            else:
                padding = max(float(np.ptp(values)) * .05, 1.)
                limits = (values.min() - padding, values.max() + padding)
            for ax in command_axes[index]:
                ax.set_ylim(*limits)
        ticks = sorted(modes)
        if ticks:
            mode_ax.set_yticks(ticks, [f'{modes[t]} ({t})' for t in ticks])
        mode_ax.set(title=f'{key}: observed flight mode', ylabel='Mode')
        waypoint_ax.set(title='Current mission item (retained outside mission mode)',
                        ylabel='Native item number')
        waypoint_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        for ax in (mode_ax, waypoint_ax):
            ax.set(xlabel='Time from mission start [s]', xlim=(0, duration))
            ax.grid(alpha=.25)
            ax.legend()
        summary['phases'].append(entry)
    commands_fig.suptitle('Real flight / SITL: logged actuator commands')
    timeline_fig.suptitle('Real flight / SITL: observed mode and mission progress')
    for fig, stem in [(commands_fig, 'command_comparison'), (timeline_fig, 'mission_timeline')]:
        fig.savefig(output / (stem + '.png'), dpi=160)
        fig.savefig(output / (stem + '.svg'))
        plt.close(fig)
    return summary
