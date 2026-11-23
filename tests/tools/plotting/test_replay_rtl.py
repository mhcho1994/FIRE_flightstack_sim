"""RTL work-item timing must remain distinct from autopilot navigation mode."""
import numpy as np
import pytest

from tools.plotting.analyze_replay_rtl import home_height, rtl_events, stage_series


def test_home_height_translates_different_estimator_origins():
    # Same 15 m height with different AMSL references / local z origins.
    real = home_height(-15.12, 158.86, 158.98)
    sitl = home_height(-15.07, -.14, -.07)
    np.testing.assert_allclose([real, sitl], [15., 15.], atol=1e-12)


def flight_topics():
    # Mission's earlier nav_cmd 16 and preflight ground flags are irrelevant.
    # LAND is an internal RTL stage; nav_state stays 5 through landing.
    status = {'timestamp': np.array([10,20,28,31])*1_000_000,
              'nav_state': np.array([3,5,5,4]), 'arming_state': np.array([2,2,2,1])}
    navigator = {'timestamp': np.array([15,21,24,26])*1_000_000,
                 'nav_cmd': np.array([16,31,16,21])}
    land = {'timestamp': np.array([5,10,28,29])*1_000_000,
            'ground_contact': np.array([1,0,1,1]), 'landed': np.array([1,0,0,1])}
    return status, navigator, land


def test_land_detected_from_work_item_without_auto_land_transition():
    events = rtl_events(*flight_topics(), 10_000_000, 31_000_000)
    assert events == {'rtl':10., 'climb':11., 'return':14., 'land':16.,
                      'ground_contact':18., 'landed':19., 'disarmed':21.}
    np.testing.assert_array_equal(stage_series(np.array([10,11,14,16,18,19,21]),events),
                                  [0,1,2,3,3,4,5])


def test_missing_climb_evidence_is_not_silently_inferred_from_mode():
    status,navigator,land = flight_topics()
    navigator['nav_cmd'][1] = 16
    with pytest.raises(ValueError,match='nav_cmd=31'):
        rtl_events(status,navigator,land,10_000_000,31_000_000)
