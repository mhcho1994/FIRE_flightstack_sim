import json
import struct
from types import SimpleNamespace

import pytest
import yaml

from tools.commander.fc_parameter_profiles import normalize_profiles, resolve_profiles
from tools.commander.fc_parameter_registry import PX4_MC, DEFAULT_GROUPS
from tools.commander.mavlink_parameters import ParameterClient
from tools.scenario.create_params_sets import create_params_sets, encode_parameter_values
from tools.scenario.capture_flight_parameters import read_flight_parameters
from tools.scenario.import_real_flight_mission import main
from tools.launcher.startup_parameters import collect_boot_parameters
from tests.tools.scenario.test_px4_mission import fixture, native_log, FLIGHT_PARAMETERS
from tests.tools.scenario.test_capture_flight_parameters_file import values


def profile():
    return dict(px4=dict(groups=['trajectory', 'auto_yaw'],
        baseline=dict(MPC_XY_TRAJ_P=.5, MPC_XY_ERR_MAX=2., MPC_YAW_MODE=0,
                      MPC_YAWRAUTO_MAX=45., MPC_YAWRAUTO_ACC=60.),
        group_parameters={'trajectory': dict(operation='scale', uniform=[.8, 1.2]), 'auto_yaw': 'fixed'},
        parameters={'MPC_YAW_MODE': dict(operation='set', choices=[0, 3, 5])}))


def test_group_rules_fixed_groups_and_integer_choices_are_reproducible():
    raw = profile()
    p = normalize_profiles(raw)
    a = resolve_profiles(p, 3, 4)['px4']
    assert a == resolve_profiles(p, 3, 4)['px4']
    assert a['values']['MPC_YAWRAUTO_MAX'] == 45
    assert .4 <= a['values']['MPC_XY_TRAJ_P'] <= .6
    assert isinstance(a['values']['MPC_YAW_MODE'], int)
    assert normalize_profiles(create_params_sets(raw))['px4']['parameters'] == p['px4']['parameters']
    assert DEFAULT_GROUPS['px4'] == ('rate', 'attitude', 'position', 'velocity')


@pytest.mark.parametrize('rule', [dict(operation='set', uniform=[0, 3]),
                                  dict(operation='set', choices=[99]),
                                  dict(operation='set', value=1.5)])
def test_invalid_integer_rules_rejected(rule):
    p = profile(); p['px4']['parameters']['MPC_YAW_MODE'] = rule
    with pytest.raises(ValueError): normalize_profiles(p)


def test_native_bson_int32_and_negative_configuration_float():
    payload = encode_parameter_values({'MPC_YAW_MODE': 3, 'SYS_VEHICLE_RESP': -.4}, 'px4')
    assert b'\x10MPC_YAW_MODE\x00' + struct.pack('<i', 3) in payload
    assert b'\x01SYS_VEHICLE_RESP\x00' in payload
    assert struct.unpack('<i', payload[:4])[0] == len(payload)


def test_px4_integer_wire_encoding_and_readback():
    sent = []
    m = SimpleNamespace(target_system=1, target_component=1,
        mav=SimpleNamespace(param_set_send=lambda *args: sent.append(args)))
    c = ParameterClient(m, autopilot='px4')
    msg = SimpleNamespace(get_type=lambda: 'PARAM_VALUE', param_id='MPC_YAW_MODE',
                          param_type=6, param_value=struct.unpack('<f', struct.pack('<i', 3))[0])
    c._receive = lambda deadline: msg
    assert c.set_param('MPC_YAW_MODE', 3) == 3
    assert sent[0][-1] == 6
    assert struct.unpack('<i', struct.pack('<f', sent[0][-2]))[0] == 3
    msg.param_type = 9
    with pytest.raises(ValueError): c._wait_value('MPC_YAW_MODE')


def test_replay_snapshot_rules_and_boot_payload(tmp_path):
    plan, data = fixture()
    plan_path = tmp_path/'mission.plan'; plan_path.write_text(json.dumps(plan))
    log = tmp_path/'flight.ulg'
    params = {**values('px4'), **FLIGHT_PARAMETERS, 'MPC_XY_TRAJ_P': .5, 'MPC_XY_ERR_MAX': 2.}
    native_log(log, data, changes=[(1_500_000, 'MPC_XY_TRAJ_P', .4)], parameters=params)
    rules = tmp_path/'rules.yaml'
    rules.write_text(yaml.safe_dump(dict(group_parameters={'trajectory': 'fixed', 'horizontal_motion': 'fixed'},
        parameters={'MPC_XY_CRUISE': dict(operation='set', value=4.)})))
    run = tmp_path/'run'
    assert main(['--plan', str(plan_path), '--flight-log', str(log), '--outdir', str(run),
                 '--parameter-groups', *DEFAULT_GROUPS['px4'], 'trajectory', 'horizontal_motion',
                 '--parameter-experiment', str(rules)]) == 0
    scenario = yaml.safe_load((run/'scenario.yaml').read_text())
    phase = scenario['mission_session']['phases'][0]
    c = phase['controller_gains']
    assert c['baseline']['MPC_XY_TRAJ_P'] == pytest.approx(.4)
    assert c['values']['MPC_XY_TRAJ_P'] == pytest.approx(.4)
    assert c['values']['MPC_XY_CRUISE'] == 4
    assert phase['flight_parameters']['values']['MPC_XY_CRUISE'] == 4
    _, boot, _ = collect_boot_parameters(scenario, 'px4', run)
    assert boot['MPC_XY_CRUISE'] == 4
    # Missing selected groups never substitute firmware defaults.
    with pytest.raises(ValueError, match='missing'):
        read_flight_parameters(log, groups=['hover_estimator'])


def test_reboot_required_changes_rejected_between_phases(tmp_path):
    from tools.commander.mission_session import normalize_session
    # Use the generated replay fixture to retain valid timeline evidence.
    plan, data = fixture()
    p = tmp_path/'plan.plan'; p.write_text(json.dumps(plan))
    log = tmp_path/'log.ulg'; native_log(log, data)
    run = tmp_path/'run'; assert main(['--plan', str(p), '--flight-log', str(log), '--outdir', str(run)]) == 0
    scenario = yaml.safe_load((run/'scenario.yaml').read_text())
    import copy
    first = scenario['mission_session']['phases'][0]
    first['controller_gains']['values']['IMU_GYRO_CUTOFF'] = 40
    second = copy.deepcopy(first); second['id'] = 'second'
    second['controller_gains']['values']['IMU_GYRO_CUTOFF'] = 100
    scenario['mission_session']['phases'].append(second)
    with pytest.raises(ValueError, match='Reboot-required'):
        normalize_session(scenario, 'px4')
