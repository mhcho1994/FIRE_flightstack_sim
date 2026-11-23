"""Native velocity, phase boundaries, and state plot regression coverage."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.tools.scenario.test_px4_mission import FORMATS, fixture, native_log, save
from tools.plotting.compare_flight_replay import compare_run
from tools.plotting.flight_replay_state import read_ulog_velocities, speed_components, ulog_velocity
from tools.scenario.px4_mission import read_log


@pytest.fixture
def flight_data(monkeypatch):
    monkeypatch.setitem(FORMATS, 'vehicle_local_position',
        'uint64_t timestamp;float vx;float vy;float vz;bool v_xy_valid;bool v_z_valid;')
    plan, data = fixture()
    base = data['vehicle_global_position'][0]
    data['vehicle_global_position'] = [dict(base, timestamp=t) for t in range(2_100_000, 10_000_000, 100_000)]
    data['vehicle_local_position'] = [
        dict(timestamp=t, vx=3., vy=4., vz=12., v_xy_valid=True, v_z_valid=True)
        for t in range(1_900_000, 10_100_000, 100_000)]
    return plan, data


def test_logged_velocity_preserves_ned_values_and_excludes_setup_and_cleanup(tmp_path, flight_data):
    _, path = save(tmp_path, *flight_data)
    log, _ = read_log(path)
    values = ulog_velocity(log, 4_000_000, 5_500_000, end_exclusive=True)
    # Positions in this log are stationary: differentiating them would give zero.
    np.testing.assert_allclose(values[:, 1:], np.tile([3, 4, -12], (15, 1)))
    np.testing.assert_allclose(speed_components(values)[:, 1:], np.tile([5, -12, 13], (15, 1)))
    assert values[0, 0] == 0
    assert values[-1, 0] == pytest.approx(1.4)


@pytest.mark.parametrize('mutation, message', [
    ('invalid', 'Invalid'), ('nonfinite', 'Nonfinite'), ('duplicate', 'Conflicting'),
    ('gap', 'velocity gap'), ('missing', 'missing required topic'),
])
def test_bad_native_velocity_is_not_replaced_by_position_derivatives(tmp_path, flight_data, mutation, message):
    plan, data = flight_data
    rows = data['vehicle_local_position']
    if mutation == 'invalid':
        rows[8]['v_xy_valid'] = False
    elif mutation == 'nonfinite':
        rows[8]['vz'] = float('nan')
    elif mutation == 'duplicate':
        rows.append(dict(rows[8], vx=99.))
    elif mutation == 'gap':
        data['vehicle_local_position'] = [r for r in rows if not 3_000_000 < r['timestamp'] < 5_000_000]
    else:
        del data['vehicle_local_position']
    _, path = save(tmp_path, plan, data)
    log, _ = read_log(path)
    with pytest.raises(ValueError, match=message):
        ulog_velocity(log, 2_000_000, 10_000_000)


def test_velocity_sources_never_join_overlapping_boot_clocks(tmp_path, flight_data):
    plan, data = flight_data
    _, a = save(tmp_path, plan, data)
    b = tmp_path/'second.ulg'
    data['vehicle_local_position'] = [dict(r, vx=6., vy=8., vz=-24.) for r in data['vehicle_local_position']]
    native_log(b, data)
    sources = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (a, b)]
    markers = {str(i): dict(start_time_us=2_000_000, end_time_us=10_000_000, source_sha256=s['sha256'])
               for i, s in enumerate(sources)}
    velocities = read_ulog_velocities(sources, markers)
    np.testing.assert_allclose(velocities['0'][0], [0, 3, 4, -12])
    np.testing.assert_allclose(velocities['1'][0], [0, 6, 8, 24])
    with pytest.raises(ValueError, match='one native ULog'):
        read_ulog_velocities(sources, {'ambiguous': dict(start_time_us=2_000_000, end_time_us=10_000_000)})
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        read_ulog_velocities([dict(sources[0], sha256='wrong')], markers)


@pytest.mark.parametrize('source', ['auto', 'telemetry'])
def test_compare_run_writes_state_grid_with_native_speeds(tmp_path, flight_data, monkeypatch, source):
    from matplotlib.figure import Figure
    from tools.commander.mission_session import normalize_session
    from tools.scenario.import_real_flight_mission import main

    plan, data = flight_data
    plan_path, log_path = save(tmp_path, plan, data)
    run = tmp_path/'run'
    assert main(['--plan', str(plan_path), '--flight-log', str(log_path), '--outdir', str(run)]) == 0
    scenario = (run/'scenario.yaml').read_bytes()
    session = normalize_session(yaml.safe_load(scenario), 'px4')
    phase = session['phases'][0]
    key = phase['id']
    sitl_data = copy.deepcopy(data)
    for row in sitl_data['vehicle_global_position']:
        row.update(lat=session['target_origin_lla'][0], lon=session['target_origin_lla'][1])
        row['alt'] += 1.
    sitl_data['vehicle_local_position'] = [
        dict(r, vx=6., vy=8., vz=-24.) for r in data['vehicle_local_position']]
    logs = run/'px4_logs'
    logs.mkdir()
    native_log(logs/'sitl.ulg', sitl_data)
    result = dict(status='completed', scenario_sha256=hashlib.sha256(scenario).hexdigest(),
                  phases=[dict(id=key, sitl_start_time_us=2_500_000, sitl_end_time_us=10_000_000)])
    (logs/'session_result.json').write_text(json.dumps(result))
    (logs/'session_trajectory.csv').write_text(
        'phase,time_s,north_m,east_m,up_m\n' +
        ''.join(f'{key},{t / 10},0,0,3\n' for t in range(75)))
    output = tmp_path/'comparison'
    output.mkdir()
    old_plots = [output/f'{stem}.{ext}' for stem in ('trajectory_comparison', 'speed_comparison')
                 for ext in ('png', 'svg')]
    for path in old_plots:
        path.write_text('superseded plot')
    captured = {}
    original_save = Figure.savefig

    def capture_figure(fig, filename, *args, **kwargs):
        if Path(filename).name == 'state_comparison.png':
            captured['axes'] = fig.axes
        return original_save(fig, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capture_figure)
    report = compare_run(run, output, sitl_source=source)
    axes = captured['axes']
    assert len(axes) == 6
    assert axes[0].get_subplotspec().get_gridspec().get_geometry() == (2, 3)
    # Plot each source's height above home, not the difference between them.
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), 2.)
    np.testing.assert_allclose(axes[1].lines[1].get_ydata(), 3.)
    assert [line.get_label() for line in axes[1].lines[:2]] == ['Real', 'SITL']
    for ax, real_value, sitl_value in zip(axes[3:], (5., -12., 13.), (10., 24., 26.)):
        np.testing.assert_allclose(ax.lines[0].get_ydata(), real_value)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), sitl_value)
    assert (output/'state_comparison.svg').is_file()
    assert (output/'trajectory_3d.png').is_file()
    assert (output/'command_comparison.png').is_file()
    assert (output/'mission_timeline.png').is_file()
    assert (output/'setpoint_tracking.svg').is_file()
    progress = json.loads((output/'progress_analysis.json').read_text())
    source_progress = progress['phases'][0]['sources']['sitl']
    assert not source_progress['horizontal_speed']['controller']['available']
    assert not source_progress['translated_position_available']
    assert source_progress['bounds']['start_time_us'] == (2_000_000 if source == 'auto' else 2_500_000)
    assert report['diagnostics']['phases'][0]['real']['mode_available']
    assert report['diagnostics']['phases'][0]['real']['waypoint_available']
    # Optional command topics are absent from this fixture, without losing the other plots.
    assert report['diagnostics']['phases'][0]['real']['commands'] == []
    assert not any(path.exists() for path in old_plots)
    velocity = report['state_comparison']['velocity']
    assert 'vehicle_local_position' in velocity['fields']
    assert velocity['sitl']['markers'][key]['start_time_us'] == (2_000_000 if source == 'auto' else 2_500_000)
    exported = np.loadtxt(output/f'{key}_sitl_speed.csv', delimiter=',', skiprows=1)
    np.testing.assert_allclose(exported[0], [0, 10, 24, 26])


def test_lockstep_duplicate_publications_require_ordered_sample_evidence():
    from tools.scenario.px4_mission import unique_state_publications
    rows = [dict(timestamp=20, timestamp_sample=12, vx=1.),
            dict(timestamp=20, timestamp_sample=20, vx=2.),
            dict(timestamp=30, timestamp_sample=28, vx=3.)]
    clean = unique_state_publications(rows, ('vx',), 'native velocity')
    assert [(r['timestamp'], r['vx']) for r in clean] == [(20, 2.), (30, 3.)]
    for invalid in (12, 11, 21, 0):
        bad = [rows[0], dict(rows[1], timestamp_sample=invalid)]
        with pytest.raises(ValueError, match='Conflicting'):
            unique_state_publications(bad, ('vx',), 'native velocity')
