"""Check response coordinates, time alignment, and incomplete-flight handling."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.plotting import plot_sitl_run_comparison as comparison
from tools.plotting.trajectory_loader import AutopilotResponse, TimeSeries, find_source_file


class RunComparisonTests(unittest.TestCase):
    def make_runs(self, root, last_status="success"):
        runs = [root / "run_000", root / "run_001"]
        for i, path in enumerate(runs):
            (path / "px4_logs").mkdir(parents=True)
            (path / "px4_logs/result.json").write_text(json.dumps({
                "status": "success" if i == 0 else last_status,
            }))
            (path / "scenario.yaml").write_text(yaml.safe_dump({
                "common": {"scenario": {"name": "mission", "wind_m_s": [3*i, 5*i, 2*i]}},
            }))
        return runs

    def test_loader_converts_ned_to_enu_without_recentering_and_aligns_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = self.make_runs(Path(temporary))
            response = AutopilotResponse(
                "px4", Path("test.ulg"),
                TimeSeries(np.array([0., 1., 2.]), np.array([[10., 20., -3.], [11., 22., -4.], [12., 24., -5.]]), ("x", "y", "z")),
                TimeSeries(np.array([0., 1., 2.]), np.array([[0., 0., 0.], [0., 0., -0.5], [0., 0., -0.5]]), ("vx", "vy", "vz")),
                TimeSeries(np.array([0.1, 1.1, 2.1]), np.full((3, 3), np.pi / 4), ("roll", "pitch", "yaw")),
                TimeSeries(np.array([0.2, 1.2, 2.2]), np.full((3, 3), np.pi / 2), ("p", "q", "r")),
            )
            with patch.object(comparison, "load_run", return_value={"px4": response}), redirect_stdout(io.StringIO()):
                traces = comparison.load_comparison_runs(runs, "px4")
            np.testing.assert_array_equal(traces[0].position.t, [-1., 0., 1.])
            np.testing.assert_array_equal(traces[0].position.values[0], [20., 10., 3.])
            self.assertIn("wind ENU [3, 5, 2]", traces[1].label)
            np.testing.assert_array_equal(traces[0].velocity.values[1], [0., 0., 0.5])
            np.testing.assert_array_equal(traces[0].velocity.t, [-1., 0., 1.])
            np.testing.assert_allclose(traces[0].attitude.t, [-0.9, 0.1, 1.1])
            np.testing.assert_allclose(traces[0].rate.t, [-0.8, 0.2, 1.2])
            np.testing.assert_array_equal(traces[0].attitude.values, response.attitude.values)
            np.testing.assert_array_equal(traces[0].rate.values, response.rate.values)

    def make_archived_runs(self, root, autopilot="ardu"):
        run = root / "run_000"
        live = root / "run_001"
        session = live / f"{autopilot}_attempts" / "session_test"
        attempt = session / "attempt_001"
        scenario_key = "ardupilot" if autopilot == "ardu" else autopilot
        for directory, wind, mission in (
            (run, [0, 0, 0], "mission"),
            (live, [9, 9, 9], "changed_live_mission"),
            (session, [2, 1, 0.5], "mission"),
        ):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "scenario.yaml").write_text(yaml.safe_dump({
                "common": {"scenario": {"name": mission, "wind_m_s": wind}},
                "autopilots": {scenario_key: {"sim": {"frame": "test_vehicle"}}},
            }))
        for directory, status in ((run / f"{autopilot}_logs", "success"), (attempt, "timeout")):
            (directory / "logs").mkdir(parents=True)
            (directory / "result.json").write_text(json.dumps({
                "autopilot": scenario_key, "status": status,
            }))
            (directory / "logs/response.csv").write_text(
                "time_s,x,y,z,vx,vy,vz,roll,pitch,yaw,p,q,r\n"
                "0,1,2,-3,0,0,0,0,0,0,0,0,0\n"
                "1,2,4,-5,1,2,-0.5,0.1,0.2,0.3,0.1,0.2,0.3\n"
            )
            (directory / "eeprom.bin").write_bytes(b"not a flight log")
        return run, attempt

    def test_mixed_run_and_attempt_use_archived_scenario_and_read_real_logs(self):
        for autopilot in ("ardu", "px4"):
            with self.subTest(autopilot=autopilot), tempfile.TemporaryDirectory() as temporary:
                run, attempt = self.make_archived_runs(Path(temporary), autopilot)
                with warnings.catch_warnings(record=True) as caught, redirect_stdout(io.StringIO()):
                    warnings.simplefilter("always")
                    traces = comparison.load_comparison_runs(
                        [run, attempt], autopilot, include_incomplete=True,
                    )
                self.assertEqual(len(caught), 1)
                self.assertIn("timeout", str(caught[0].message))
                self.assertIn("wind ENU [2, 1, 0.5]", traces[1].label)
                self.assertIn("timeout", traces[1].label)
                np.testing.assert_array_equal(traces[1].position.values, [[2, 1, 3], [4, 2, 5]])
                np.testing.assert_array_equal(traces[1].position.t, [-1, 0])
                self.assertIsNotNone(traces[1].velocity)
                self.assertIsNotNone(traces[1].attitude)
                self.assertIsNotNone(traces[1].rate)

    def test_archived_timeout_requires_include_incomplete_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = self.make_archived_runs(Path(temporary))
            with patch.object(comparison, "load_run") as load:
                with self.assertRaisesRegex(ValueError, "timeout.*--include-incomplete"):
                    comparison.load_comparison_runs(runs, "ardupilot")
                load.assert_not_called()

    def test_empty_attempt_does_not_select_eeprom_or_another_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, attempt = self.make_archived_runs(Path(temporary))
            (attempt / "logs/response.csv").unlink()
            sibling = attempt.parent / "attempt_002/logs"
            sibling.mkdir(parents=True)
            (sibling / "00000001.BIN").write_bytes(b"another attempt")
            self.assertIsNone(find_source_file(attempt, "ardupilot"))
            self.assertIsNone(find_source_file(attempt, "px4"))
            # A run without active logs must not select an archived attempt.
            self.assertIsNone(find_source_file(attempt.parents[2], "ardu"))

    def test_missing_session_snapshot_does_not_use_current_run_scenario(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = self.make_archived_runs(Path(temporary))
            (runs[1].parent / "scenario.yaml").unlink()
            with warnings.catch_warnings(record=True) as caught, redirect_stdout(io.StringIO()):
                warnings.simplefilter("always")
                traces = comparison.load_comparison_runs(runs, "ardu", include_incomplete=True)
            self.assertIn("wind ENU [unknown]", traces[1].label)
            self.assertTrue(any("no scenario.yaml" in str(w.message) for w in caught))

    def test_ardupilot_settings_are_checked_under_the_full_scenario_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = self.make_archived_runs(Path(temporary))
            scenario_path = runs[1].parent / "scenario.yaml"
            scenario = yaml.safe_load(scenario_path.read_text())
            scenario["autopilots"]["ardupilot"]["sim"]["frame"] = "different_vehicle"
            scenario_path.write_text(yaml.safe_dump(scenario))
            with warnings.catch_warnings(record=True) as caught, redirect_stdout(io.StringIO()):
                warnings.simplefilter("always")
                comparison.load_comparison_runs(runs, "ardu", include_incomplete=True)
            self.assertTrue(any("settings other than wind differ" in str(w.message) for w in caught))

    def test_running_flight_is_rejected_before_logs_are_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = self.make_runs(Path(temporary), "running")
            with patch.object(comparison, "load_run") as load:
                with self.assertRaisesRegex(ValueError, "running"):
                    comparison.load_comparison_runs(runs, "px4")
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
