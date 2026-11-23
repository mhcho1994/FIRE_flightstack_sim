import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.commander.fc_parameter_profiles import normalize_profiles, resolve_profiles
from tools.commander.fc_parameter_registry import ARDUPILOT_CATALOG, parameter_names
from tools.commander.mavlink_parameters import ParameterClient
from tools.launcher.parameter_bridge import bridge_gains
from tools.launcher.startup_parameters import collect_boot_parameters
from tools.scenario.capture_flight_parameters import read_flight_parameters
from tools.scenario.create_params_sets import write_parameter_files
from tests.tools.scenario.test_capture_flight_parameters_file import write_bin


def source_values():
    report = json.loads((Path(__file__).parents[3]/'docs/ardupilot_parameter_groups_260910.json').read_text())
    return {r['parameter']:r['values']['00000006.BIN'] for rows in report['groups'].values() for r in rows}


def capture(tmp_path, **kwargs):
    log = write_bin(tmp_path/'flight.BIN', source_values(),
        changes=[(2000, 'WP_SPD', 8.), (4000, 'WP_SPD', 6.)],
        firmware_messages=['ArduCopter V4.7.0 (1511f271)'])
    return read_flight_parameters(log, groups=['all'], snapshot='time:0.003', **kwargs)[0]


def test_full_bin_snapshot_group_variation_and_native_boot(tmp_path):
    raw = capture(tmp_path)
    p = raw['ardupilot']
    assert len(p['baseline']) == 152
    assert p['baseline']['WP_SPD'] == 8
    p['group_parameters'] = {'motion_limits': {'operation':'scale', 'uniform':[.9,1.1]}, 'propulsion':'fixed'}
    p['parameters'] = {'WP_YAW_BEHAVIOR': {'operation':'set', 'choices':[0,2,3]}}
    profiles = normalize_profiles(raw)
    resolved = resolve_profiles(profiles, 42, 7)
    assert resolved == resolve_profiles(profiles, 42, 7)
    c = resolved['ardupilot']
    assert 7.2 <= c['values']['WP_SPD'] <= 8.8
    assert c['values']['MOT_THST_HOVER'] == p['baseline']['MOT_THST_HOVER']
    assert isinstance(c['values']['WP_YAW_BEHAVIOR'], int)
    bridged, changes = bridge_gains(c, 'ardupilot', {'parameter_schema':'arducopter-4.7'})
    assert not changes and len(bridged['values']) == 152
    files = write_parameter_files(resolved, tmp_path)
    data = {'autopilots':{'ardupilot':{'controller_gains':files['ardupilot']}}}
    content, values, _ = collect_boot_parameters(data, 'ardupilot', tmp_path)
    assert len(values)==152 and b'WP_SPD ' in content
    assert len(parameter_names('ardupilot', ['rate','controller'], 'arducopter-4.7')) == len(set(parameter_names('ardupilot', ['rate','controller'], 'arducopter-4.7')))
    with pytest.raises(ValueError, match='verified'):
        bridge_gains(c, 'ardupilot', {'parameter_schema':'arducopter-4.6'})


@pytest.mark.parametrize('name,tag,value', [('WP_YAW_BEHAVIOR',2,2),('MOT_YAW_HEADROOM',4,200),('RTL_LOIT_TIME',6,5000),('INS_GYRO_FILTER',4,20),('TKOFF_RPM_MIN',4,0)])
def test_ardupilot_integer_numeric_encoding_and_type_checked_readback(name,tag,value):
    sent=[]
    connection=SimpleNamespace(target_system=1,target_component=1,
        mav=SimpleNamespace(param_set_send=lambda *args:sent.append(args)))
    client=ParameterClient(connection,autopilot='ardupilot')
    msg=SimpleNamespace(get_type=lambda:'PARAM_VALUE',param_id=name,param_type=tag,param_value=float(value))
    client._receive=lambda deadline:msg
    assert client.set_param(name,value)==value
    assert sent[0][-2:]==(float(value),tag)
    msg.param_type=9
    with pytest.raises(ValueError):client._wait_value(name)


@pytest.mark.parametrize('name,rule', [
    ('WP_YAW_BEHAVIOR',dict(operation='set',choices=[99])),
    ('WP_YAW_BEHAVIOR',dict(operation='set',uniform=[0,3])),
    ('MOT_YAW_HEADROOM',dict(operation='set',value=1.5)),
    ('RTL_OPTIONS',dict(operation='set',value=1)),
])
def test_invalid_discrete_rules_rejected(tmp_path,name,rule):
    raw=capture(tmp_path);raw['ardupilot']['parameters']={name:rule}
    with pytest.raises(ValueError):normalize_profiles(raw)


def test_conflicting_alias_groups_fail_and_missing_log_values_never_default(tmp_path):
    raw=capture(tmp_path)
    raw['ardupilot']['group_parameters']={g:dict(operation='scale',value=v) for g,v in [('rate',.9),('controller',1.1)]}
    with pytest.raises(ValueError,match='overlapping'):normalize_profiles(raw)
    vals=source_values();del vals['WP_ACC']
    log=write_bin(tmp_path/'missing.BIN',vals,firmware_messages=['ArduCopter V4.7.0 (1511f271)'])
    with pytest.raises(ValueError,match='WP_ACC'):read_flight_parameters(log,groups=['motion_limits'])
