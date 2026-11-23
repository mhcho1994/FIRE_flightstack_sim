"""Native ULog integration for explicitly selected airborne mission windows."""
import json

import pytest
import yaml

from tests.tools.scenario.test_px4_mission import fixture, save, FORMATS
from tools.scenario.flight_log import read_flight_log
from tools.scenario.import_real_flight_mission import main
from tools.scenario.mission_segment import mission_segment_flights
from tools.commander.mission_session import normalize_session


@pytest.fixture
def airborne_log(tmp_path, monkeypatch):
    # Scalar field names preserve pyulog's expanded array keys in these fixtures.
    formats = {
        'vehicle_local_position': 'uint64_t timestamp;float vx;float vy;float vz;float heading;bool xy_valid;bool z_valid;bool v_xy_valid;bool v_z_valid;bool heading_good_for_control;',
        'vehicle_attitude': 'uint64_t timestamp;' + ''.join(f'float q[{i}];' for i in range(4)),
        'vehicle_angular_velocity': 'uint64_t timestamp;' + ''.join(f'float xyz[{i}];' for i in range(3)),
    }
    for key, value in formats.items():
        monkeypatch.setitem(FORMATS, key, value)
    plan, data = fixture()
    data['vehicle_status'] = [
        dict(timestamp=1_100_000, armed_time=1_100_000, arming_state=2, nav_state=2),
        dict(timestamp=4_000_000, armed_time=1_100_000, arming_state=2, nav_state=3),
        dict(timestamp=6_000_000, armed_time=1_100_000, arming_state=2, nav_state=4),
        dict(timestamp=7_000_000, armed_time=1_100_000, arming_state=2, nav_state=5),
        dict(timestamp=10_000_000, armed_time=0, arming_state=1, nav_state=5),
    ]
    gp = data['vehicle_global_position'][0]
    data['vehicle_global_position'] = [dict(gp, timestamp=t) for t in range(3_950_000, 6_100_000, 100_000)]
    data['vehicle_local_position'] = [dict(timestamp=3_950_000, vx=.1, vy=.2, vz=-.1, heading=.5,
        xy_valid=True, z_valid=True, v_xy_valid=True, v_z_valid=True, heading_good_for_control=True)]
    data['vehicle_attitude'] = [dict(timestamp=3_950_000, **{f'q[{i}]': v for i, v in enumerate([1., 0., 0., 0.])})]
    data['vehicle_angular_velocity'] = [dict(timestamp=3_950_000, **{f'xyz[{i}]': 0. for i in range(3)})]
    from tests.tools.scenario.test_px4_mission import native_log, FLIGHT_PARAMETERS
    from tests.tools.scenario.test_capture_flight_parameters_file import values
    p = tmp_path/'source.plan'
    p.write_text(json.dumps(plan))
    log = tmp_path/'source.ulg'
    params = {**values('px4'), **FLIGHT_PARAMETERS, 'MIS_TAKEOFF_ALT': 2.5}
    def write():
        native_log(log, data, parameters=params)
        return read_flight_log(log)
    decoded = write()
    segment = next(s for s in decoded.analyze([p])['flights'][0]['segments'] if s['mode'] == 'MISSION')
    return p, log, data, write, 'flight_000:' + segment['id']


def test_airborne_segment_import_reports_setup_and_keeps_whole_plan(airborne_log, tmp_path):
    p, log, data, write, selector = airborne_log
    out = tmp_path/'out'
    assert main(['--flight-log', str(log), '--plan', str(p), '--mission-segment', selector, '--outdir', str(out)]) == 0
    scenario = yaml.safe_load((out/'scenario.yaml').read_text())
    phase = normalize_session(scenario, 'px4')['phases'][0]
    assert phase['start_source_sequence'] == 1
    assert phase['comparison_duration_s'] == 2.
    assert len(phase['mission_items']) == 3
    assert phase['initial_state']['kind'] == 'in_air'
    assert phase['initial_state']['velocity_ned_m_s'] == pytest.approx([.1, .2, -.1])
    assert phase['initial_state']['position_ned_m'][2] == pytest.approx(-2.)
    assert phase['initial_state']['internal_state_restored'] is False
    assert phase['flight_parameters']['values']['MIS_TAKEOFF_ALT'] == 2.5
    report = json.loads((out/'import_report.json').read_text())
    assert report['phases'][0]['start_condition'] == 'in_air'
    assert 'EKF' in ' '.join(report['phases'][0]['initial_state']['limitations'])
    import csv
    with (out/'replay/reference_trajectory.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(4_000_000 <= int(r['boot_time_us']) < 6_000_000 for r in rows)


@pytest.mark.parametrize('topic, change, error', [
    ('vehicle_local_position', {'timestamp': 4_010_000}, 'fresh sample'),
    ('vehicle_attitude', {'timestamp': 3_000_000}, 'fresh sample'),
    ('vehicle_local_position', {'v_xy_valid': False}, 'Invalid position'),
    ('vehicle_local_position', {'vx': float('nan')}, 'nonfinite'),
])
def test_initial_state_never_uses_future_stale_or_invalid_samples(airborne_log, topic, change, error):
    p, log, data, write, selector = airborne_log
    data[topic][0].update(change)
    with pytest.raises(ValueError, match=error):
        mission_segment_flights(write(), [p], selector)


def test_explicit_selection_does_not_relax_whole_flight_contract(airborne_log):
    from tools.scenario.flight_replay import timeline_flights
    p, log, data, write, selector = airborne_log
    with pytest.raises(ValueError, match='ground state'):
        timeline_flights(write(), [p])
    with pytest.raises(ValueError, match='Unknown'):
        mission_segment_flights(write(), [p], 'flight_000:segment_999')


def test_on_ground_segment_is_reported_without_air_setup(tmp_path):
    p, log = save(tmp_path, *fixture())
    decoded = read_flight_log(log)
    segment = next(s for s in decoded.analyze([p])['flights'][0]['segments'] if s['mode'] == 'MISSION')
    phase = mission_segment_flights(decoded, [p], 'flight_000:' + segment['id'])[0][0]
    assert phase['initial_state']['kind'] == 'on_ground'
    assert 'setup' not in phase['initial_state']


def test_session_rejects_unbounded_or_unmapped_air_start(airborne_log, tmp_path):
    p, log, _, _, selector = airborne_log
    out = tmp_path/'out'
    main(['--flight-log', str(log), '--plan', str(p), '--mission-segment', selector, '--outdir', str(out)])
    scenario = yaml.safe_load((out/'scenario.yaml').read_text())
    phase = scenario['mission_session']['phases'][0]
    phase['start_source_sequence'] = 99
    with pytest.raises(ValueError, match='mapping'):
        normalize_session(scenario, 'px4')
    phase['start_source_sequence'] = 1
    phase['initial_state']['setup']['velocity_tolerance_m_s'] = 0
    with pytest.raises(ValueError, match='velocity_tolerance'):
        normalize_session(scenario, 'px4')
