"""Regression checks for diagnostic units, discrete states, and phase clocks."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.plotting.flight_replay_diagnostics import (
    bin_diagnostics, read_diagnostics, step_series, ulog_diagnostics,
)


def log_with(*definitions):
    datasets = [SimpleNamespace(name=name, multi_id=instance,
                                data={key: np.asarray(value) for key, value in data.items()})
                for name, instance, data in definitions]

    def get_dataset(name, instance=0):
        for ds in datasets:
            if ds.name == name and ds.multi_id == instance:
                return ds
        raise KeyError(name)

    return SimpleNamespace(data_list=datasets, get_dataset=get_dataset)


BOUNDS = dict(start_time_us=2_000_000, end_time_us=4_000_000, end_exclusive=True)


def test_discrete_state_carries_initial_value_and_excludes_end_transition():
    result = step_series([1e6, 3e6, 4e6, 5e6], [3, 4, 5, 18], BOUNDS)
    np.testing.assert_array_equal(result, [[0, 3, 1e6], [1, 4, 3e6], [2, 4, 3e6]])


def test_discrete_state_does_not_backfill_future_or_turn_invalid_into_waypoint_zero():
    result = step_series([3e6], [4], BOUNDS)
    assert result[0, 0] == 1
    result = step_series([1e6, 3e6], [np.nan, 4], BOUNDS)
    assert np.isnan(result[0, 1])
    assert result[1, 1] == 4


def test_motor_commands_are_normalized_and_keep_native_channel_indices():
    log = log_with(
        ('actuator_motors', 0, dict(timestamp=[1e6, 2e6, 3e6, 4e6],
            **{'control[0]': [1, .2, .4, .9], 'control[2]': [0, .3, .5, .8],
               'control[1]': [np.nan] * 4})),
        ('actuator_outputs', 1, dict(timestamp=[2e6, 3e6], noutputs=[1, 1],
                                   **{'output[0]': [1200, 1400]})))
    item = ulog_diagnostics(log, BOUNDS)
    assert len(item['commands']) == 1
    stream = item['commands'][0]
    assert stream['topic'] == 'actuator_motors'
    assert stream['columns'] == ['control[0]', 'control[2]']
    assert stream['units'] == 'Normalized thrust [-1, 1]'
    np.testing.assert_allclose(stream['values'], [[0, .2, .3], [1, .4, .5]])


def test_raw_output_fallback_finds_active_instance_without_claiming_pwm_units():
    log = log_with(
        ('actuator_outputs', 0, dict(timestamp=[1e6], noutputs=[1], **{'output[0]': [1000]})),
        ('actuator_outputs', 1, dict(timestamp=[2e6, 3e6, 4e6], noutputs=[2, 2, 2],
            **{'output[0]': [1500, 1600, 1700], 'output[1]': [0, 0, 0], 'output[2]': [1900] * 3})))
    stream, = ulog_diagnostics(log, BOUNDS)['commands']
    assert stream['instance'] == 1
    assert stream['columns'] == ['output[0]']
    assert stream['units'] == 'Native output units'
    np.testing.assert_array_equal(stream['values'], [[0, 1500], [1, 1600]])


def test_waypoint_validity_and_hold_mode_are_observed_not_inferred():
    log = log_with(
        ('vehicle_status', 0, dict(timestamp=[1e6, 3e6], nav_state=[3, 4])),
        ('mission_result', 0, dict(timestamp=[1e6, 2.5e6, 3.5e6, 4e6],
                                  seq_current=[0, 4, 4, 5], valid=[False, True, True, True])))
    item = ulog_diagnostics(log, BOUNDS)
    assert np.isnan(item['waypoint'][0, 1])
    np.testing.assert_array_equal(item['waypoint'][1:], [[.5, 4, 2.5e6], [2, 4, 2.5e6]])
    assert item['mode'][-1, 1] == 4
    assert item['waypoint'][-1, 1] == 4  # Still item 4 while HOLD, not zero or a fabricated advance.


def test_bin_pwm_modes_and_native_item_numbers_are_preserved():
    records = [
        dict(mavpackettype='MODE', TimeUS=1e6, ModeNum=3),
        dict(mavpackettype='MODE', TimeUS=3e6, ModeNum=5),
        dict(mavpackettype='MISE', TimeUS=1e6, CNum=2),
        dict(mavpackettype='MISE', TimeUS=4e6, CNum=0),
        dict(mavpackettype='RCOU', TimeUS=2e6, C1=1200, C2=1400),
        dict(mavpackettype='RCOU', TimeUS=3e6, C1=1250, C2=1450),
    ]
    item = bin_diagnostics(records, BOUNDS)
    stream, = item['commands']
    assert stream['units'] == 'Output PWM [us]'
    np.testing.assert_array_equal(stream['values'], [[0, 1200, 1400], [1, 1250, 1450]])
    np.testing.assert_array_equal(item['waypoint'][:, 1], [2, 2])
    np.testing.assert_array_equal(item['mode'][:, 1], [3, 5, 5])


def test_diagnostic_reader_binds_each_phase_to_its_log_hash(monkeypatch):
    logs = {name: log_with(('vehicle_status', 0, dict(timestamp=[1e6], nav_state=[mode])))
            for name, mode in [('a', 3), ('b', 4)]}
    monkeypatch.setattr('tools.scenario.px4_mission.read_log', lambda path: (logs[path], path))
    sources = [dict(path=name, sha256=name) for name in logs]
    markers = {name: dict(BOUNDS, source_sha256=name) for name in logs}
    result = read_diagnostics(sources, markers, 'ULog vehicle_global_position')
    assert result['a']['mode'][0, 1] == 3
    assert result['b']['mode'][0, 1] == 4
    with pytest.raises(ValueError, match='one identified'):
        read_diagnostics(sources, {'ambiguous': BOUNDS}, 'ULog vehicle_global_position')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        read_diagnostics([dict(path='a', sha256='wrong')], markers, 'ULog vehicle_global_position')
