"""Native BIN velocity and 3D waypoint plots for completed replay sessions."""
from __future__ import annotations

import csv
import hashlib
import textwrap
from pathlib import Path

import numpy as np


COLORS = {'real': '#1871ad', 'sitl': '#dc701b'}
VELOCITY_HEADER = 'time_s,north_m_s,east_m_s,up_m_s,desired_north_m_s,desired_east_m_s,desired_up_m_s'


def mission_waypoints(phase, origin):
    """Return sequence, command, N/E/up; exclude home and in-place TAKEOFF."""
    from tools.scenario.import_real_flight_mission import lla_to_ned

    points = []
    for sequence, item in enumerate(phase['mission_items']):
        if sequence == 0 or item['command'] not in (16, 21):
            continue
        if item['frame'] not in (3, 6):
            raise ValueError('Waypoint plots require altitude relative to home')
        north, east, _ = lla_to_ned([item['p5'], item['p6'], item['p7']], origin)
        points.append([sequence, item['command'], north, east, item['p7']])
    if not points:
        raise ValueError('No navigation waypoints to plot')
    return np.asarray(points, dtype=float)


def align_velocity_axes(axes, start_us, end_us):
    """Interpolate E/D onto N timestamps strictly inside this flight's overlap.

    Inputs are TimeUS, measured velocity, desired velocity for each N/E/D axis.
    Output is seconds since MISE and measured/desired N/E/up velocity in m/s.
    """
    cropped = []
    for axis in axes:
        rows = np.asarray(axis, dtype=float).reshape(-1, 3)
        rows = rows[(rows[:, 0] >= start_us) & (rows[:, 0] <= end_us)]
        rows = rows[np.argsort(rows[:, 0], kind='stable')]
        if len(rows) < 2 or not np.isfinite(rows).all():
            raise ValueError('Insufficient or nonfinite PSC velocity samples in flight')
        if np.any(np.diff(rows[:, 0]) <= 0):
            raise ValueError('Ambiguous duplicate PSC velocity timestamps')
        cropped.append(rows)
    lower = max(rows[0, 0] for rows in cropped)
    upper = min(rows[-1, 0] for rows in cropped)
    times = cropped[0][:, 0]
    times = times[(times >= lower) & (times <= upper)]
    if len(times) < 2:
        raise ValueError('No common PSC velocity interval')
    measured = [np.interp(times, rows[:, 0], rows[:, 1]) for rows in cropped]
    desired = [np.interp(times, rows[:, 0], rows[:, 2]) for rows in cropped]
    measured[2] *= -1
    desired[2] *= -1
    return np.column_stack([(times - start_us) / 1e6, *measured, *desired])


def read_velocities(sources, markers):
    """Read controller-used velocity estimates, not an arbitrary EKF lane."""
    from pymavlink import DFReader

    fields = {'PSCN': ('VN', 'DVN'), 'PSCE': ('VE', 'DVE'), 'PSCD': ('VD', 'DVD')}
    axes = {kind: [] for kind in fields}
    for source in sources:
        path = Path(source['path'])
        digest = source['sha256']
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Velocity source SHA256 mismatch: {path}')
        log = DFReader.DFReader_binary(str(path))
        try:
            while (message := log.recv_match(type=list(fields), strict=True)) is not None:
                measured, desired = fields[message.get_type()]
                axes[message.get_type()].append([
                    message.TimeUS, getattr(message, measured), getattr(message, desired)])
        finally:
            log.close()
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Velocity source changed while reading: {path}')
    return {key: align_velocity_axes(list(axes.values()), bounds['start_time_us'], bounds['end_time_us'])
            for key, bounds in markers.items()}


def _waypoint_labels(points):
    # The first and last waypoint can be almost coincident. Keep both IDs legible.
    groups = []
    for point in points:
        label = ('L' if point[1] == 21 else 'W') + str(int(point[0]))
        nearby = next((group for group in groups if np.linalg.norm(point[2:] - group[0][2:]) < 1), None)
        if nearby is None:
            groups.append((point, [label]))
        else:
            nearby[1].append(label)
    return groups


def _plot_3d(ax, real, sitl, points, bounds, title):
    from matplotlib.ticker import MaxNLocator

    ax.plot(real[:, 2], real[:, 1], real[:, 3], color=COLORS['real'], lw=1.9, label='Real')
    ax.plot(sitl[:, 2], sitl[:, 1], sitl[:, 3], color=COLORS['sitl'], lw=1.7, ls='--', label='SITL')
    ax.plot(points[:, 3], points[:, 2], points[:, 4], color='#686868', lw=1.2, ls=':', label='Waypoint order')
    waypoints, land = points[points[:, 1] == 16], points[points[:, 1] == 21]
    ax.scatter(waypoints[:, 3], waypoints[:, 2], waypoints[:, 4], s=42, marker='D',
               facecolors='white', edgecolors='#263442', depthshade=False, label='Waypoint')
    ax.scatter(land[:, 3], land[:, 2], land[:, 4], s=65, marker='v', color='#263442',
               depthshade=False, label='LAND')
    for point, labels in _waypoint_labels(points):
        ax.text(point[3] + 1, point[2] + 1, point[4] + .8, ' / '.join(labels), fontsize=9, color='#263442')
    ax.set(xlabel='East [m]', ylabel='North [m]', zlabel='Height above home [m]', title=title,
           xlim=bounds[0], ylim=bounds[1], zlim=bounds[2])
    # Preserve metric proportions; use identical bounds/view for every mission.
    ax.set_box_aspect([high - low for low, high in bounds])
    ax.view_init(elev=27, azim=-58)
    ax.set_proj_type('ortho')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
    ax.tick_params(labelsize=8)


def write_detail_plots(output, session, real, sitl, source_info):
    """Write separate 3D and velocity figures plus the plotted numerical data."""
    if source_info['kind'] != 'BIN POS':
        raise ValueError('Detailed plots require --sitl-source bin and native PSC velocity logs')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(output)
    phases = session['phases']
    real_markers = {phase['id']: {key: phase[key] for key in ('start_time_us', 'end_time_us')} for phase in phases}
    velocities = {
        'real': read_velocities([dict(path=session['source_log'], sha256=session['source_log_sha256'])], real_markers),
        'sitl': read_velocities(source_info['sources'], source_info['markers']),
    }
    waypoints = {phase['id']: mission_waypoints(phase, session['target_origin_lla']) for phase in phases}
    all_positions = np.concatenate([rows[:, 1:4] for rows in [*real.values(), *sitl.values()]] +
                                   [rows[:, 2:5] for rows in waypoints.values()])
    minimum, maximum = all_positions.min(axis=0), all_positions.max(axis=0)
    padding = np.maximum((maximum - minimum) * .08, [3., 3., 1.5])
    bounds = [(minimum[i] - padding[i], maximum[i] + padding[i]) for i in (1, 0, 2)]
    caption = ('W / L labels are mission item numbers. Dotted lines show waypoint order. '
               'TAKEOFF #1 climbs in place; its stored latitude/longitude is not a travel waypoint.\n'
               'Shared horizontal origin; altitude relative to each flight home. Equal metric axis scale; no fitted alignment.')
    fig = plt.figure(figsize=(6.6 * len(phases), 7.2))
    handles = None
    summary = dict(velocity_source='PSCN.VN / PSCE.VE / PSCD.VD; controller-used estimates; vertical sign changed to up',
                   desired_velocity_source='PSCN.DVN / PSCE.DVE / PSCD.DVD; trajectory desired velocity',
                   velocity_alignment='First TAKEOFF MISE per flight; E/D interpolated onto N within overlap only; no smoothing',
                   waypoint_notes=caption, phases=[])
    for index, phase in enumerate(phases):
        key = phase['id']; r, s, points = real[key], sitl[key], waypoints[key]
        title = f'Mission {index + 1} | {Path(phase["plan"]).name}\nReal {r[-1, 0]:.1f} s / SITL {s[-1, 0]:.1f} s'
        ax = fig.add_subplot(1, len(phases), index + 1, projection='3d')
        _plot_3d(ax, r, s, points, bounds, title)
        handles = ax.get_legend_handles_labels()
        single = plt.figure(figsize=(9, 7.6))
        single_ax = single.add_subplot(111, projection='3d')
        _plot_3d(single_ax, r, s, points, bounds, '')
        single.suptitle(title, fontsize=14, y=.98)
        single.legend(*handles, loc='upper center', bbox_to_anchor=(.5, .90), ncol=5, fontsize=9)
        single.text(.5, .025, textwrap.fill(caption, width=130), ha='center', fontsize=8)
        single.subplots_adjust(left=.02, right=.97, bottom=.12, top=.84)
        single.savefig(output / f'{key}_trajectory_3d.png', dpi=180)
        single.savefig(output / f'{key}_trajectory_3d.svg')
        plt.close(single)
        with (output / f'{key}_waypoints.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['mission_item', 'command', 'north_m', 'east_m', 'height_above_home_m'])
            writer.writerows([[int(row[0]), int(row[1]), *row[2:]] for row in points])
        entry = dict(id=key, plan=Path(phase['plan']).name)
        for name, trajectory in [('real', r), ('sitl', s)]:
            np.savetxt(output / f'{key}_{name}_trajectory.csv', trajectory, delimiter=',', comments='',
                       header='time_s,north_m,east_m,height_above_home_m')
            velocity = velocities[name][key]
            np.savetxt(output / f'{key}_{name}_velocity.csv', velocity, delimiter=',', comments='', header=VELOCITY_HEADER)
            entry[name] = dict(duration_s=float(trajectory[-1, 0]),
                               max_horizontal_speed_m_s=float(np.linalg.norm(velocity[:, 1:3], axis=1).max()),
                               max_3d_speed_m_s=float(np.linalg.norm(velocity[:, 1:4], axis=1).max()))
        summary['phases'].append(entry)
    fig.suptitle('Real flight / SITL: 3D mission trajectories and waypoints', fontsize=17, y=.97)
    fig.legend(*handles, loc='upper center', bbox_to_anchor=(.5, .92), ncol=5, fontsize=11)
    fig.text(.5, .035, caption, ha='center', fontsize=10)
    fig.subplots_adjust(left=.015, right=.985, bottom=.16, top=.76, wspace=.02)
    fig.savefig(output / 'trajectory_3d_comparison.png', dpi=180)
    fig.savefig(output / 'trajectory_3d_comparison.svg')
    plt.close(fig)

    speed_fig, axes = plt.subplots(len(phases), 3, figsize=(16, 3.5 * len(phases)), squeeze=False)
    last_time = max(rows[-1, 0] for by_phase in velocities.values() for rows in by_phase.values())
    for index, phase in enumerate(phases):
        key = phase['id']
        for name in ('real', 'sitl'):
            rows = velocities[name][key]
            for offset, style, width, alpha in [(1, '-', 1.7, 1.), (4, '--', 1., .65)]:
                components = rows[:, offset:offset + 3]
                columns = [np.linalg.norm(components[:, :2], axis=1), components[:, 2], np.linalg.norm(components, axis=1)]
                for ax, values in zip(axes[index], columns):
                    ax.plot(rows[:, 0], values, color=COLORS[name], ls=style, lw=width, alpha=alpha)
        for column, (title, ylabel) in enumerate([('Horizontal speed', 'Speed [m/s]'),
                                                ('Vertical velocity (up +)', 'Velocity [m/s]'),
                                                ('3D speed', 'Speed [m/s]')]):
            ax = axes[index, column]
            ax.set(title=f'Mission {index + 1} | {Path(phase["plan"]).name} | {title}',
                   xlabel='Time since mission start [s]', ylabel=ylabel, xlim=(0, last_time + 1))
            ax.grid(alpha=.22)
            if column == 1:
                ax.axhline(0, color='#68737d', lw=.7)
        # Same vertical limits across missions are set below.
    for column in range(3):
        lower = min(ax.get_ylim()[0] for ax in axes[:, column])
        upper = max(ax.get_ylim()[1] for ax in axes[:, column])
        for ax in axes[:, column]:
            ax.set_ylim(lower if column == 1 else 0, upper)
    legend = [Line2D([], [], color=COLORS[name], ls=style, lw=1.7, label=f'{label} {kind}')
              for kind, style in [('measured', '-'), ('desired', '--')]
              for name, label in [('real', 'Real'), ('sitl', 'SITL')]]
    speed_fig.suptitle('Real flight / SITL: velocity comparison', fontsize=17, y=.99)
    speed_fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .955), ncol=4, fontsize=11)
    speed_fig.text(.5, .015, 'Native PSC velocity logs; vertical positive upward. Each flight starts at its first TAKEOFF MISE. '
                   'Each source retains its full duration; no time rescaling.', ha='center', fontsize=10)
    speed_fig.tight_layout(rect=(0, .045, 1, .905))
    speed_fig.savefig(output / 'speed_comparison.png', dpi=180)
    speed_fig.savefig(output / 'speed_comparison.svg')
    plt.close(speed_fig)
    return summary
