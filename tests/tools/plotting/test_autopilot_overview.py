"""Protect physical units, common clocks, and native mission settings."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from tools.commander.mission_session import normalize_session
from tools.plotting.compare_autopilot_response import acceptance_radius, align_groundtruth, waypoint_passages
from tools.plotting.trajectory_loader import AutopilotResponse, TimeSeries


def test_ascent_alignment_rejects_startup_spike_and_preserves_attitude_clock():
    t = np.arange(100, 104, .1)
    velocity = np.zeros((len(t), 3))
    velocity[0, 2] = -5  # Isolated startup spike.
    velocity[20:, 2] = -1
    attitude_t = np.array([101.5, 102.5, 103.5])
    response = AutopilotResponse("px4", Path("flight.ulg"),
        velocity=TimeSeries.from_boot_time(t, velocity, ("N", "E", "D")),
        attitude=TimeSeries.from_boot_time(attitude_t, np.zeros((3, 3)), ("r", "p", "y")))
    aligned, origin = align_groundtruth(response)
    assert origin == pytest.approx(102)
    np.testing.assert_allclose(aligned.attitude.t, [-.5, .5, 1.5])
    np.testing.assert_allclose(aligned.attitude.boot_t, attitude_t)


def test_radius_uses_logged_native_units_and_px4_mission_override():
    native = dict(waypoint_acceptance_radius_m=0)
    assert acceptance_radius("px4", {"NAV_ACC_RAD": 2}, native) == ("NAV_ACC_RAD", 2)
    assert acceptance_radius("ardu", {"WPNAV_RADIUS": 175}, native) == ("WPNAV_RADIUS", 1.75)
    assert acceptance_radius("ardu", {"WP_RADIUS_M": 2, "WPNAV_RADIUS": 175}, native) == ("WP_RADIUS_M", 2)
    assert acceptance_radius("px4", {"NAV_ACC_RAD": 2}, {}) == ("mission param2", 1)
    with pytest.raises(ValueError, match="No logged"):
        acceptance_radius("ardu", {}, native)


def test_waypoint_passages_use_altitude_and_order_not_just_horizontal_distance():
    position = TimeSeries(np.arange(5.), np.array([[0, 0, 0], [0, 0, -10],
        [0, 2, -10], [0, 0, -20], [0, 5, -20]]), ("N", "E", "D"))
    passages = waypoint_passages(position, np.array([[0, 0, -10], [0, 0, -20]]))
    assert [p["index"] for p in passages] == [1, 3]
    assert all(p["distance_3d_m"] == 0 for p in passages)


def test_generator_native_defaults_upload_exactly_four_waypoints_and_land_at_endpoint(tmp_path):
    root = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, "-m", "tools.scenario.scenario_generator", "three_d_n_pts",
        "--outdir", str(tmp_path), "--seed", "2027", "--n", "5",
        "--edge-m", "0", "70", "60", "70", "--vertex-deg", "0", "90", "180", "90",
        "--alt-m", "12", "12", "32", "18", "--takeoff-alt-m", "12",
        "--px4-initial-heading-deg", "90", "--native-navigation-defaults",
        "--land-at-last-waypoint"], cwd=root, check=True, capture_output=True)
    data = yaml.safe_load((tmp_path / "run_000/scenario.yaml").read_text())
    assert data["autopilots"]["px4"]["sim"]["initial_heading_deg"] == 90
    for autopilot in ("px4", "ardupilot"):
        items = normalize_session(data, autopilot)["phases"][0]["mission_items"]
        assert [i["command"] for i in items] == [22, 16, 16, 16, 16, 21]
        assert all(i["p2"] == 0 for i in items if i["command"] == 16)
        assert items[-1]["p5"] == items[-2]["p5"]
        assert items[-1]["p6"] == items[-2]["p6"]
        assert items[-1]["p7"] == 0
    assert all("controller_gains" not in s and "flight_parameters" not in s
               for s in data["autopilots"].values())


@pytest.mark.parametrize("radius", [-1, float("nan"), float("inf")])
def test_invalid_acceptance_radius_is_rejected_before_upload(radius):
    data = dict(common=dict(scenario=dict(home_lla=[40, -86, 0], takeoff_alt_m=10,
        command=[22, 16, 21], waypoints_lla=[None, [40, -86, 10], None],
        waypoint_acceptance_radius_m=radius)))
    with pytest.raises(ValueError, match="waypoint_acceptance_radius_m"):
        normalize_session(data, "px4")
