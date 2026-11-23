import numpy as np
import pytest

from tests.tools.scenario.test_px4_mission import fixture,save,native_log
from tools.plotting.compare_flight_replay import ulog_trajectories


def test_native_ulog_alignment_uses_mode_transition_not_heartbeat_delay(tmp_path):
    logs=tmp_path/'px4_logs';logs.mkdir()
    _,path=save(logs,*fixture())
    session=dict(target_origin_lla=[40.44237474,-86.88899243,160.])
    result=dict(phases=[dict(id='flight_000',sitl_start_time_us=2_900_000,sitl_end_time_us=10_000_000)])
    # Add native samples at <1 s intervals, preserving a constant stationary position.
    plan,data=fixture()
    base=data['vehicle_global_position'][0]
    data['vehicle_global_position']=[dict(base,timestamp=t) for t in range(2_100_000,10_000_000,500_000)]
    native_log(path,data)
    values,info=ulog_trajectories(logs,session,result)
    assert values['flight_000'][0,0]==pytest.approx(.1)
    assert info['markers']['flight_000']['runtime_anchor_offset_s']==.9
    assert np.allclose(values['flight_000'][:,1:],[0,0,2.])


def segment_comparison_fixture(tmp_path, *, leave_early=False, sequence=1):
    plan, data = fixture()
    original = data['vehicle_status'][0]
    data['vehicle_status'] = [dict(original, timestamp=t, nav_state=mode) for t, mode in (
        (1_100_000, 2), (2_000_000, 3), (3_000_000, 4), (4_000_000, 3),
        (5_000_000 if leave_early else 6_000_000, 4), (7_000_000, 18))]
    data['vehicle_status'].append(dict(original, timestamp=10_000_000, arming_state=1, nav_state=18))
    base = data['vehicle_global_position'][0]
    data['vehicle_global_position'] = [dict(base, timestamp=t) for t in range(2_050_000, 10_000_000, 100_000)]
    logs = tmp_path/'px4_logs'
    logs.mkdir()
    _, path = save(logs, plan, data)
    session = dict(target_origin_lla=[40.44237474, -86.88899243, 160.],
                   phases=[dict(id='selected', replay_scope='mission_segment',
                                comparison_duration_s=1.5, start_source_sequence=sequence)])
    result = dict(phases=[dict(id='selected', sitl_start_time_us=4_200_000, sitl_end_time_us=10_000_000)])
    return logs, session, result


def test_segment_native_alignment_selects_requested_entry_and_excludes_cleanup(tmp_path):
    logs, session, result = segment_comparison_fixture(tmp_path)
    trajectories, info = ulog_trajectories(logs, session, result)
    a = trajectories['selected']
    assert a[0, 0] == pytest.approx(.05)
    assert a[-1, 0] == pytest.approx(1.45)
    assert info['markers']['selected']['start_time_us'] == 4_000_000
    assert info['markers']['selected']['end_time_us'] == 5_500_000


def test_segment_comparison_rejects_premature_mission_exit(tmp_path):
    logs, session, result = segment_comparison_fixture(tmp_path, leave_early=True)
    with pytest.raises(ValueError, match='left MISSION'):
        ulog_trajectories(logs, session, result)


def test_segment_comparison_checks_native_start_sequence(tmp_path):
    logs, session, result = segment_comparison_fixture(tmp_path, sequence=0)
    with pytest.raises(ValueError, match='initial mission sequence'):
        ulog_trajectories(logs, session, result)
