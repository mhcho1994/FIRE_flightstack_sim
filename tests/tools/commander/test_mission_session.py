import copy
import csv
import hashlib
import json
from collections import deque
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pymavlink import mavutil

from tools.commander.mission_session import normalize_session, phase_settings
from tools.commander import mission_session_runner as engine
from tools.commander.pymavlink_ardupilot_commander import ArduPilotMissionRunner
from tools.commander.pymavlink_px4_commander import PX4MissionRunner
from tools.commander.flight_parameters import speed_defaults
from tools.launcher.startup_parameters import collect_boot_parameters


def scenario(ap, count=1):
    items = [dict(command=c, frame=3, autocontinue=1, p1=0., p2=0., p3=0., p4=None,
                  p5=40., p6=-86., p7=h) for c, h in [(22, 2.), (16, 4.), (21, 0.)]]
    return dict(autopilots={ap: {}}, mission_session=dict(version=1, autopilot=ap,
        target_origin_lla=[40., -86., 150.], phases=[dict(id=f'flight_{i:03d}',
        mission_items=copy.deepcopy(items), ground_wait_s=0., timeout_s=10.) for i in range(count)]))


def msg(kind, **fields):
    return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: 1, get_srcComponent=lambda: 1, **fields)


@pytest.mark.parametrize('ap,cls', [('ardupilot', ArduPilotMissionRunner), ('px4', PX4MissionRunner)])
@pytest.mark.parametrize('fail_second', [False, True])
def test_common_runner_retains_one_connection_until_all_phases_and_stops_on_failure(tmp_path, ap, cls, fail_second):
    path = tmp_path/'scenario.yaml'; path.write_text(yaml.safe_dump(scenario(ap, 3)))
    r = cls(path, target_firmware=dict(autopilot=ap))
    for name in ('session_ready', 'session_prepare', 'session_start', '_clear_mission', '_upload_mission_items'):
        setattr(r, name, MagicMock(return_value=True))
    m = MagicMock(); m.wait_heartbeat.return_value = msg('HEARTBEAT')
    m.recv_match.return_value = msg('GLOBAL_POSITION_INT', time_boot_ms=100)
    seen = []
    def monitor(*args):
        assert not r.is_done()
        phase = args[2]; seen.append(phase['id'])
        assert len(phase['upload_items']) == (4 if ap == 'ardupilot' else 3)
        if fail_second and len(seen) == 2:
            raise RuntimeError('unexpected disarm')
    with patch.object(engine.mavutil, 'mavlink_connection', return_value=m) as connect, \
         patch.object(engine, 'verify_firmware_identity'), \
         patch.object(engine, 'ParameterClient'), \
         patch.object(engine, 'monitor_phase', side_effect=monitor):
        r._run()
    assert seen == ['flight_000', 'flight_001'] if fail_second else seen == ['flight_000', 'flight_001', 'flight_002']
    assert r.is_done() and r.get_status().success == (not fail_second)
    connect.assert_called_once(); m.close.assert_called_once()
    report = json.loads((tmp_path/('ardu_logs' if ap == 'ardupilot' else 'px4_logs')/'session_result.json').read_text())
    assert report['status'] == ('failed' if fail_second else 'completed')
    assert report['phases'][-1]['status'] == ('failed' if fail_second else 'completed')


def test_new_and_legacy_session_normalize_identically_without_mutation():
    new = scenario('ardupilot'); original = copy.deepcopy(new)
    legacy = copy.deepcopy(new); block = legacy.pop('mission_session'); legacy['replay_session'] = block
    phase = block['phases'][0]; phase['status'] = 'landed_disarmed'
    phase['mission_items'].insert(0, dict(command=16, frame=0))
    a = normalize_session(new, 'ardupilot'); b = normalize_session(legacy, 'ardupilot')
    assert [it['command'] for it in a['phases'][0]['mission_items']] == [22, 16, 21]
    assert [it['command'] for it in b['phases'][0]['mission_items']] == [22, 16, 21]
    assert new == original
    with pytest.raises(ValueError, match='autopilot'):
        normalize_session(new, 'px4')


def test_phase_zero_overrides_top_level_boot_parameters(tmp_path):
    data = scenario('px4', 2)
    data['autopilots']['px4']['flight_parameters'] = speed_defaults(1, 'px4', source='test')
    data['mission_session']['phases'][0]['flight_parameters'] = speed_defaults(3, 'px4', source='test')
    _, values, _ = collect_boot_parameters(data, 'px4', tmp_path)
    assert values == {'MPC_XY_CRUISE': 3.}
    session = normalize_session(data, 'px4')
    assert session['phases'][1]['flight_parameters']['values'] == {'MPC_XY_CRUISE': 1.}


@pytest.mark.parametrize('change', ['missing_land', 'zero_timeout', 'bad_frame', 'midair_takeoff'])
def test_invalid_completion_or_phase_contract_fails_before_launch(change):
    data = scenario('px4'); phase = data['mission_session']['phases'][0]
    if change == 'missing_land': phase['mission_items'].pop()
    elif change == 'zero_timeout': phase['timeout_s'] = 0
    elif change == 'bad_frame': phase['mission_items'][1]['frame'] = 0
    else: phase['mission_items'].insert(2, copy.deepcopy(phase['mission_items'][0]))
    with pytest.raises(ValueError): normalize_session(data, 'px4')


def monitor(messages, *, timeout=10, wind=None):
    clock = [0.]; messages = deque(messages)
    def receive(**kwargs):
        clock[0] += .2
        return messages.popleft() if messages else None
    m = SimpleNamespace(target_system=1, target_component=1, recv_match=receive)
    runner = SimpleNamespace(_check_stop=lambda: None, session_observe=lambda m: None, session_mode_ok=lambda *a: True)
    phase = dict(id='flight_000', timeout_s=timeout, wall_timeout_s=20, upload_items=[{}]*4)
    entry = {}; stream = StringIO()
    with patch.object(engine.time, 'monotonic', side_effect=lambda: clock[0]):
        engine.monitor_phase(runner, m, phase, 0, [40., -86., 0.], csv.writer(stream), stream, entry, wind)
    return entry


def position(t, height):
    return msg('GLOBAL_POSITION_INT', time_boot_ms=t, relative_alt=height, vz=0, lat=400000000, lon=-860000000)


def test_stale_ground_flag_cannot_complete_midair_disarm():
    with pytest.raises(RuntimeError, match='landing confirmation'):
        monitor([msg('EXTENDED_SYS_STATE', landed_state=1), position(1000, 20000),
                 msg('MISSION_CURRENT', seq=3), msg('HEARTBEAT', base_mode=0, custom_mode=3)])


def test_disarm_requires_airborne_and_final_land_and_fresh_ground():
    with pytest.raises(RuntimeError, match='unexpected disarm'):
        monitor([position(1000, 0), msg('MISSION_CURRENT', seq=3), msg('HEARTBEAT', base_mode=0)])
    result = monitor([position(1000, 2000), msg('MISSION_CURRENT', seq=3), position(2000, 0),
                      msg('EXTENDED_SYS_STATE', landed_state=1), msg('HEARTBEAT', base_mode=0)])
    assert result['completion_reason'] == 'landed_disarmed'


def test_timeout_uses_simulated_flight_time():
    with pytest.raises(TimeoutError, match='simulated'):
        monitor([position(11000, 2000)], timeout=10)


@pytest.mark.parametrize('ap,cls,name,schema', [
    ('ardupilot', ArduPilotMissionRunner, 'WPNAV_SPEED', 'arducopter-4.6'),
    ('px4', PX4MissionRunner, 'MPC_XY_CRUISE', 'px4-mc-v1'),
])
@pytest.mark.parametrize('bad_readback', [False, True])
def test_disarmed_phase_parameters_apply_restore_and_block_start_on_mismatch(tmp_path, ap, cls, name, schema, bad_readback):
    data = scenario(ap, 3)
    phases = data['mission_session']['phases']
    phases[0]['flight_parameters'] = speed_defaults(3, ap, source='test')
    phases[1]['flight_parameters'] = speed_defaults(4, ap, source='test')
    phases[2]['flight_parameters'] = None
    baseline = phases[0]['flight_parameters']['values'][name]
    path = tmp_path/'scenario.yaml'; path.write_text(yaml.safe_dump(data))
    runner = cls(path, target_firmware=dict(autopilot=ap, parameter_schema=schema))
    for method in ('session_ready', 'session_prepare', 'session_start', '_clear_mission', '_upload_mission_items'):
        setattr(runner, method, MagicMock(return_value=True))
    connection = MagicMock(); connection.wait_heartbeat.return_value = msg('HEARTBEAT')
    connection.recv_match.return_value = position(100, 0)
    client = MagicMock(); values = {name: baseline}; changes = []
    client.get_param.side_effect = lambda key: values[key]
    def set_param(key, value):
        # Firmware parameter transactions may only happen before starting a phase.
        assert client.require_disarmed.call_count == runner.session_start.call_count + 2
        changes.append((key, value))
        if not bad_readback: values[key] = value
    client.set_param.side_effect = set_param
    with patch.object(engine.mavutil, 'mavlink_connection', return_value=connection), \
         patch.object(engine, 'verify_firmware_identity'), patch.object(engine, 'verify_flight_parameters'), \
         patch.object(engine, 'ParameterClient', return_value=client), patch.object(engine, 'monitor_phase'):
        runner._run()
    assert runner.get_status().success == (not bad_readback)
    if bad_readback:
        assert runner.session_start.call_count == 1
        assert 'readback mismatch' in runner.get_status().error
    else:
        assert runner.session_start.call_count == 3
        assert changes == [(name, phases[1]['flight_parameters']['values'][name]), (name, baseline)]
        assert values[name] == baseline


def test_telemetry_loss_is_not_a_successful_landing():
    with pytest.raises(TimeoutError, match='telemetry was lost'):
        monitor([position(1000, 2000), msg('MISSION_CURRENT', seq=3)])


def test_wind_monitor_and_completion_share_the_same_received_messages():
    messages = [position(1000, 2000), msg('MISSION_CURRENT', seq=3), position(2000, 0),
                msg('EXTENDED_SYS_STATE', landed_state=1), msg('HEARTBEAT', base_mode=0)]
    wind = MagicMock(done=True)
    assert monitor(messages, wind=wind)['completion_reason'] == 'landed_disarmed'
    assert [call.args[0] for call in wind.update.call_args_list] == messages
    wind = MagicMock(done=False)
    with pytest.raises(RuntimeError, match='wind checkpoints'):
        monitor(messages, wind=wind)


def test_later_phase_parameter_file_validated_before_launch(tmp_path):
    from tools.launcher.attempt_logs import capture_run_inputs
    data = scenario('px4', 2)
    phase = data['mission_session']['phases'][1]
    phase['flight_parameters'] = dict(speed_defaults(4, 'px4', source='test'), parameter_file='missing.bson')
    (tmp_path/'scenario.yaml').write_text(yaml.safe_dump(data))
    with pytest.raises(FileNotFoundError, match='missing.bson'):
        capture_run_inputs(tmp_path, 'px4')


def test_mode_interruption_does_not_advance_the_session():
    clock = [0.]
    messages = deque([position(i*200, 2000) for i in range(1,17)] +
                     [msg('HEARTBEAT', base_mode=128, custom_mode=5)])
    def receive(**kwargs):
        clock[0] += .2
        return messages.popleft() if messages else None
    connection = SimpleNamespace(target_system=1, target_component=1, recv_match=receive)
    runner = SimpleNamespace(_check_stop=lambda: None, session_mode_ok=lambda *a: False)
    phase = dict(id='flight_001', timeout_s=10, wall_timeout_s=20, upload_items=[{}]*4)
    stream = StringIO()
    with patch.object(engine.time, 'monotonic', side_effect=lambda: clock[0]), \
         pytest.raises(RuntimeError, match='unexpected mode change'):
        engine.monitor_phase(runner, connection, phase, 0, [40., -86., 0.], csv.writer(stream), stream, {})


def test_wind_sequence_maps_preserve_each_phase(tmp_path):
    from tools.launcher.mission_wind_controller import MissionWind
    wind = SimpleNamespace(logs_dir=tmp_path)
    MissionWind.save_plan(wind, [dict(command=22, scenario_index=0, source_sequence=1)], phase_id='flight_000')
    MissionWind.save_plan(wind, [dict(command=21, scenario_index=5, source_sequence=6)], phase_id='flight_001')
    maps = json.loads((tmp_path/'mission_phase_sequence_maps.json').read_text())
    assert maps['flight_000'][0]['source_sequence'] == 1
    assert maps['flight_001'][0]['source_sequence'] == 6
