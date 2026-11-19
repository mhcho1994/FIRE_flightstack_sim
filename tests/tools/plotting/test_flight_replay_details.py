import numpy as np
import pytest

from tools.plotting.flight_replay_details import align_velocity_axes, mission_waypoints


def test_waypoint_plot_excludes_home_and_in_place_takeoff_and_keeps_relative_height():
    origin = [40., -86., 157.]
    phase = {'mission_items': [
        dict(command=16, frame=0, p5=40., p6=-86., p7=157.),
        dict(command=22, frame=3, p5=41., p6=-87., p7=2.),
        dict(command=16, frame=3, p5=40., p6=-86., p7=10.),
        dict(command=21, frame=3, p5=40., p6=-86., p7=0.),
    ]}
    result = mission_waypoints(phase, origin)
    np.testing.assert_allclose(result, [[2, 16, 0, 0, 10], [3, 21, 0, 0, 0]])


def test_velocity_alignment_never_crosses_flights_or_extrapolates_and_converts_down_to_up():
    # E samples are offset from N/D. Only N times 2 and 3 have all three axes.
    north = [[t * 1e6, 3, 10] for t in [1, 2, 3, 4]]
    east = [[t * 1e6, 2 * t, 20] for t in [1.5, 2.5, 3.5]]
    down = [[t * 1e6, -12, -30] for t in [1, 2, 3, 4]]
    for axis in (north, east, down):
        axis.extend([[0, 999, 999], [5e6, 999, 999]])
    result = align_velocity_axes([north, east, down], 1_000_000, 4_000_000)
    np.testing.assert_allclose(result, [[1, 3, 4, 12, 10, 20, 30], [2, 3, 6, 12, 10, 20, 30]])
    assert np.linalg.norm(result[0, 1:3]) == 5
    assert np.linalg.norm(result[0, 1:4]) == 13


def test_velocity_alignment_rejects_ambiguous_restarted_log_timestamps():
    rows = [[1, 0, 0], [2, 0, 0], [2, 10, 10], [3, 0, 0]]
    with pytest.raises(ValueError, match='duplicate'):
        align_velocity_axes([rows, rows, rows], 1, 3)


def test_velocity_alignment_requires_samples_inside_the_selected_flight():
    rows = [[0, 0, 0], [10, 0, 0]]
    with pytest.raises(ValueError, match='Insufficient'):
        align_velocity_axes([rows, rows, rows], 2, 8)
