import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools.commander.air_start import AirStartPreparation, state_errors, within_tolerance
from tools.commander.mission_session_runner import TimelineExecution
from tests.tools.commander.test_mission_session import msg


def state():
    return dict(position_ned_m=[10., 20., -4.], velocity_ned_m_s=[1., 0., 0.], yaw_rad=math.pi-.01,
        setup=dict(policy='position_velocity_yaw', position_tolerance_m=.5,
                   velocity_tolerance_m_s=.3, yaw_tolerance_rad=.1, settle_time_s=1., timeout_s=120.))


def actual(t, p, v, yaw=math.pi-.01):
    return dict(boot_time_us=t*1000, position_ned_m=p, velocity_ned_m_s=v, yaw_rad=yaw)


def test_setup_requires_settle_then_matching_velocity_and_wrapped_yaw():
    prep = AirStartPreparation(state())
    assert prep.target(1000) == ([7., 20., -4.], [0., 0., 0.])
    assert prep.observe(actual(1000, prep.lead, [0., 0., 0.])) is None
    assert prep.observe(actual(2000, prep.lead, [0., 0., 0.])) is None
    assert prep.target(5000) == ([10., 20., -4.], [1., 0., 0.])
    assert prep.observe(actual(5000, prep.goal, [0., 0., 0.])) is None
    matched = prep.observe(actual(5100, prep.goal, prep.velocity, -math.pi+.01))
    assert matched['matched'] and not matched['internal_state_restored']
    assert matched['errors']['yaw_rad'] == pytest.approx(.02)


def test_setup_cannot_silently_continue_after_missing_target():
    prep = AirStartPreparation(state())
    prep.moving_since_ms = 1000
    with pytest.raises(TimeoutError, match='without matching'):
        prep.target(10_000)


def test_segment_comparison_finishes_before_cleanup_landing():
    runner = SimpleNamespace(session_mode_name=lambda m: m.mode, session_set_mode=MagicMock())
    phase = dict(events=[], upload_items=[dict(source_sequence=i) for i in range(3)],
                 replay_scope='mission_segment', comparison_duration_s=2.)
    entry = {}
    engine = TimelineExecution(runner, MagicMock(), phase, entry, start_ms=1000)
    engine.update(msg('HEARTBEAT', base_mode=128, mode='MISSION'), 1500)
    assert not engine.landing_allowed(True)
    engine.update(msg('GLOBAL_POSITION_INT'), 3500)
    runner.session_set_mode.assert_not_called()
    engine.update(msg('GLOBAL_POSITION_INT'), 3750)
    runner.session_set_mode.assert_called_once_with(engine.connection, 'LAND')
    assert entry['comparison_end_time_us'] == 3_000_000
    assert entry['events'][0]['purpose'] == 'outside_comparison_window'
    assert not engine.landing_allowed(False)
    engine.update(msg('HEARTBEAT', base_mode=128, mode='LAND'), 3800)
    assert engine.landing_allowed(False)
