"""State comparison figures using native logged velocity estimates."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.plotting.flight_replay_details import COLORS, VELOCITY_HEADER, read_velocities


def ulog_velocity(log, start_us, end_us, *, end_exclusive=False):
    """Read NED estimates at publication timestamps, exporting vertical up."""
    from tools.scenario.px4_mission import topic, unique_state_publications

    rows = topic(log, 'vehicle_local_position', ('vx', 'vy', 'vz', 'v_xy_valid', 'v_z_valid'))
    rows = [r for r in rows if start_us <= r['timestamp'] <= end_us
            and (not end_exclusive or r['timestamp'] < end_us)]
    if len(rows) < 2:
        raise ValueError('Insufficient native vehicle_local_position velocity samples in comparison window')
    if any(not r['v_xy_valid'] or not r['v_z_valid'] for r in rows):
        raise ValueError('Invalid vehicle_local_position velocity estimate in comparison window')
    rows = unique_state_publications(rows, ('vx', 'vy', 'vz'), 'native velocity')
    values = np.asarray([[r['timestamp'], r['vx'], r['vy'], -r['vz']] for r in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite native velocity estimate')
    # Repeated identical publications are harmless; conflicting states are not.
    duplicate = np.diff(values[:, 0]) == 0
    if np.any(np.any(values[1:, 1:] != values[:-1, 1:], axis=1) & duplicate):
        raise ValueError('Conflicting native velocity timestamps')
    values = values[np.r_[~duplicate, True]]
    if len(values) < 2 or np.any(np.diff(values[:, 0]) > 1_000_000):
        raise ValueError('Insufficient native velocity samples or >1 s velocity gap')
    if values[0, 0] - start_us > 1_000_000 or end_us - values[-1, 0] > 1_000_000:
        raise ValueError('Incomplete native velocity coverage of comparison window')
    values[:, 0] = (values[:, 0] - start_us) / 1e6
    return values


def read_ulog_velocities(sources, markers):
    """Keep each selected flight in its own log's boot clock."""
    from tools.scenario.px4_mission import read_log

    decoded = {}
    for source in sources:
        if source['sha256'] in decoded:
            continue
        log, digest = read_log(source['path'])
        if digest != source['sha256']:
            raise ValueError(f"Velocity source SHA256 mismatch: {source['path']}")
        decoded[digest] = log
    result = {}
    for key, bounds in markers.items():
        digest = bounds.get('source_sha256')
        if digest is None and len(decoded) == 1:
            digest = next(iter(decoded))
        if digest not in decoded:
            raise ValueError(f'{key}: velocity source does not identify one native ULog')
        result[key] = ulog_velocity(decoded[digest], bounds['start_time_us'], bounds['end_time_us'],
                                    end_exclusive=bounds.get('end_exclusive', False))
    return result


def session_velocities(session, source_info):
    """Use the same source and time bounds as the compared trajectories."""
    real_markers = {p['id']: dict(start_time_us=p['start_time_us'], end_time_us=p['end_time_us'],
                                end_exclusive=p.get('replay_scope') == 'mission_segment')
                    for p in session['phases']}
    sources = [dict(path=session['source_log'], sha256=session['source_log_sha256'])]
    if source_info['kind'] == 'ULog vehicle_global_position':
        reader = read_ulog_velocities
        fields = 'vehicle_local_position.vx / vy / vz (NED)'
        alignment = 'Native publication timestamps; no velocity interpolation or smoothing'
    elif source_info['kind'] == 'BIN POS':
        reader = read_velocities
        fields = 'PSCN.VN / PSCE.VE / PSCD.VD (NED)'
        alignment = 'E/D interpolated onto N timestamps within each flight overlap only; no smoothing'
    else:
        raise ValueError('State comparison requires native ULog or BIN velocity sources')
    velocities = dict(real=reader(sources, real_markers),
                      sitl=reader(source_info['sources'], source_info['markers']))
    info = dict(basis='native logged velocity estimates; no position differentiation',
                fields=fields, output_frame='north/east/up; up = -down',
                horizontal_speed='sqrt(v_north^2 + v_east^2)',
                vertical_velocity='-v_down (up positive)',
                speed_3d='sqrt(v_north^2 + v_east^2 + v_down^2)',
                alignment=alignment,
                real=dict(sources=sources, markers=real_markers),
                sitl=dict(sources=source_info['sources'], markers=source_info['markers'],
                          time_zero=source_info.get('time_zero', 'native first TAKEOFF MISE')))
    return velocities, info


def speed_components(velocity):
    """Time, horizontal speed, signed upward velocity, and 3D speed."""
    return np.column_stack([velocity[:, 0], np.linalg.norm(velocity[:, 1:3], axis=1),
                            velocity[:, 3], np.linalg.norm(velocity[:, 1:4], axis=1)])


def write_state_comparison(output, session, real, sitl, paired, velocities):
    """Write a two-row, three-column state panel for each phase."""
    import matplotlib.pyplot as plt

    output = Path(output)
    phases = session['phases']
    fig, axes = plt.subplots(2 * len(phases), 3, figsize=(16, 8 * len(phases)),
                             squeeze=False, layout='constrained')
    for index, phase in enumerate(phases):
        key = phase['id']
        position_axes, velocity_axes = axes[2 * index:2 * index + 2]
        for name, positions in [('real', real[key]), ('sitl', sitl[key])]:
            style = dict(color=COLORS[name], label='Real' if name == 'real' else 'SITL',
                         lw=1.5, ls='-' if name == 'real' else '--')
            position_axes[0].plot(positions[:, 2], positions[:, 1], **style)
            position_axes[1].plot(positions[:, 0], positions[:, 3], **style)
            velocity = velocities[name][key]
            header = VELOCITY_HEADER if velocity.shape[1] == 7 else 'time_s,north_m_s,east_m_s,up_m_s'
            np.savetxt(output / f'{key}_{name}_velocity.csv', velocity, delimiter=',', comments='', header=header)
            speeds = speed_components(velocity)
            np.savetxt(output / f'{key}_{name}_speed.csv', speeds, delimiter=',', comments='',
                       header='time_s,horizontal_speed_m_s,vertical_up_speed_m_s,speed_3d_m_s')
            for j, ax in enumerate(velocity_axes):
                ax.plot(speeds[:, 0], speeds[:, j + 1], **style)
        position_axes[0].set(xlabel='East [m]', ylabel='North [m]',
                             title=f'{key}: horizontal position')
        position_axes[0].set_aspect('equal', adjustable='box')
        errors = paired[key]
        position_axes[1].set(title='Altitude', ylabel='Height above home [m]')
        position_axes[2].plot(errors[:, 0], np.linalg.norm(errors[:, 7:10], axis=1),
                              color='#7851a9', label='||SITL - Real||')
        position_axes[2].set(title='3D position difference', ylabel='Position error [m]', ylim=(0, None))
        for ax, title, ylabel in zip(velocity_axes,
                ('Horizontal speed', 'Vertical velocity (up positive)', '3D speed'),
                ('Speed [m/s]', 'Velocity [m/s]', 'Speed [m/s]')):
            ax.set(title=title, ylabel=ylabel)
        velocity_axes[0].set_ylim(bottom=0)
        velocity_axes[1].axhline(0, color='gray', lw=.7)
        velocity_axes[2].set_ylim(bottom=0)
        for ax in [*position_axes[1:], *velocity_axes]:
            ax.set_xlabel('Time from mission start [s]')
            for event in phase.get('events', []):
                ax.axvline(event['time_s'], color='gray', alpha=.35, ls=':')
        for ax in [*position_axes, *velocity_axes]:
            ax.grid(alpha=.25)
            ax.legend()
    fig.suptitle('Real flight / SITL: state comparison\n'
                 'Native logged velocity; position errors use overlapping times only')
    fig.savefig(output / 'state_comparison.png', dpi=160)
    fig.savefig(output / 'state_comparison.svg')
    plt.close(fig)
    return dict(figure='state_comparison.png', vector_figure='state_comparison.svg',
                layout='2 rows x 3 columns per phase',
                altitude='Real and SITL height above their respective flight home; native trajectory samples',
                position_difference='Euclidean norm of SITL - real position; overlap only',
                event_lines='Source event schedule (dotted)')
