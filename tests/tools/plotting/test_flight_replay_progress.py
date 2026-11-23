"""Progress diagnostics preserve source identity, phase bounds, and setpoint semantics."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.plotting.flight_replay_progress import (
    active_targets, changes, latest, local_reference, selected_logs, window,
)


class Log:
    def __init__(self, **topics):
        self.topics = topics

    def get_dataset(self, name, instance):
        return SimpleNamespace(data=self.topics[name])


def arrays(**fields):
    return {k: np.asarray(v) for k, v in fields.items()}


def test_window_carries_known_state_without_future_backfill_or_cleanup():
    data = arrays(timestamp=[50, 150, 200, 300], nav_state=[3, 4, 5, 18])
    bounds = dict(start_time_us=100, end_time_us=200, end_exclusive=True)
    assert latest(data, 49) is None
    assert window(data, bounds) == [1]
    assert window(data, bounds, carry=True) == [0, 1]
    events = changes(data, ['nav_state'], bounds)
    assert [(e['source_time_us'], e['nav_state']) for e in events] == [(50, 3), (150, 4)]
    assert events[0]['time_s'] == 0
    assert changes(arrays(timestamp=[250], nav_state=[3]), ['nav_state'], bounds) == []


def test_target_join_preserves_unknown_and_invalid_states_and_raw_land_altitude():
    sp = arrays(**{'timestamp': [50, 150, 175, 200], 'current.valid': [1, 1, 0, 1],
                   'current.type': [0, 4, 0, 0], 'current.lat': [40.] * 4,
                   'current.lon': [-86.] * 4, 'current.alt': [120, 999, 130, 140]})
    log = Log(position_setpoint_triplet=sp,
              vehicle_status=arrays(timestamp=[120], nav_state=[3]),
              mission_result=arrays(timestamp=[130], seq_current=[4], seq_reached=[3], valid=[1]))
    values = active_targets(log, dict(start_time_us=100, end_time_us=200, end_exclusive=True),
                            [40, -86, 100], 100)
    assert values.shape == (3, 14)
    assert values[0, 0] == 50 and values[0, 1] == 0
    assert np.isnan(values[0, 2:6]).all()
    np.testing.assert_allclose(values[1, 2:8], [3, 4, 3, 1, 4, 1])
    assert values[1, 10] == 999  # LAND raw altitude retained for audit.
    assert np.isnan(values[2, 11:]).all()  # Invalid target is not a position.


def test_local_reference_uses_observed_origin_and_ned_sign_without_future_fill():
    data = arrays(**{'timestamp': [50, 150, 250], 'position[0]': [1, 1, 1],
                     'position[1]': [2, 2, 2], 'position[2]': [-3, -3, -3]})
    local = arrays(timestamp=[100, 200], ref_lat=[40, 40], ref_lon=[-86, -86],
                   ref_alt=[105, 110], xy_global=[1, 0], z_global=[1, 1])
    values = local_reference(data, local, dict(start_time_us=0, end_time_us=300),
                             [40, -86, 100], 100)
    assert np.isnan(values[0, 1:]).all()
    np.testing.assert_allclose(values[1, 1:], [1, 2, 8])
    assert np.isnan(values[2, 1:3]).all()
    assert values[2, 3] == 13


def test_progress_sources_require_hash_binding_even_with_overlapping_boot_clocks(monkeypatch):
    from tools.plotting import flight_replay_progress as module

    logs = {'a': object(), 'b': object()}
    monkeypatch.setattr(module, 'read_log', lambda path: (logs[path], path))
    info = dict(sources=[dict(path=k, sha256=k) for k in logs],
                markers={'segment_007': dict(start_time_us=10, end_time_us=20, source_sha256='b'),
                         'another_flight': dict(start_time_us=10, end_time_us=20, source_sha256='a')})
    selected = selected_logs(info)
    assert selected['segment_007'][0] is logs['b']
    assert selected['another_flight'][0] is logs['a']
    info['markers']['segment_007'].pop('source_sha256')
    with pytest.raises(ValueError, match='one identified'):
        selected_logs(info)
    info['sources'][0]['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='SHA256'):
        selected_logs(info)
