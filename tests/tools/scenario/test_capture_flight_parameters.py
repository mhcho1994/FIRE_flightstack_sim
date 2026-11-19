"""Unified capture source selection, argument validation and SITL CLI lifecycle."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.scenario import capture_flight_parameters as capture
from tools.scenario.create_params_sets import create_params_sets


class CaptureFlightParametersTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="unified capture ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "profile.yaml"
        self.scenario = self.root / "scenario.yaml"
        self.scenario.write_text(yaml.safe_dump({"autopilots": {"px4": {"sim": {
            "px4_dir": str(self.root / "px4"), "vehicle": 4006, "frame": "gz_fire_px4vision",
            "world": "default_fire", "location": "Purdue",
        }}}}))

    def test_cli_requires_one_source_and_rejects_wrong_mode_options_before_work(self):
        cases = [
            [],
            ["--from-file", "flight.BIN", "--from-sitl", str(self.scenario)],
            ["--from-file", "flight.BIN", "--output", str(self.output), "--timeout", "60"],
            ["--from-file", "flight.BIN", "--output", str(self.output), "--autopilots", "px4"],
            ["--from-file", "flight.BIN", "--output", str(self.output), "--work-dir", str(self.root)],
            ["--from-sitl", str(self.scenario), "--snapshot", "initial"],
            ["--from-sitl", str(self.scenario), "--autopilot", "auto"],
            ["--from-sitl", str(self.scenario), "--groups", "attitude", "position"],
            ["--from-sitl", str(self.scenario), "--merge"],
            ["--from-file", "flight.BIN", "--output", str(self.output), "--merge", "--overwrite"],
            ["--from-sitl", str(self.scenario), "--output", str(self.root / "profile.json")],
        ]
        with patch.object(capture, "_capture_from_file") as file_capture, \
             patch.object(capture, "_capture_from_sitl") as sitl_capture:
            for args in cases:
                with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    capture.main(args)
                self.assertEqual(error.exception.code, 2)
            file_capture.assert_not_called()
            sitl_capture.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertFalse((self.root / "parameter_captures").exists())

    def test_python_api_routes_file_options_and_never_starts_sitl(self):
        result = {"file": "result"}
        with patch.object(capture, "_capture_from_file", return_value=result) as file_capture, \
             patch.object(capture, "_capture_from_sitl") as sitl_capture:
            self.assertIs(capture.capture_flight_parameters(from_file="flight.ulg"), result)
            file_capture.assert_called_once_with("flight.ulg", ROOT / "data/parameters/flight_test_controller_gains.yaml", autopilot="auto",
                snapshot="initial", groups=None, merge=False, overwrite=False, parameter_schema=None)
            capture.capture_flight_parameters(from_file="flight.ulg", output=self.output,
                autopilot="px4", snapshot="final", groups=["attitude", "position"], merge=True)
            file_capture.assert_called_with("flight.ulg", self.output, autopilot="px4",
                snapshot="final", groups=["attitude", "position"], merge=True, overwrite=False, parameter_schema=None)
            sitl_capture.assert_not_called()

    def test_python_api_routes_sitl_defaults_and_explicit_options(self):
        result = {"sitl": "result"}
        with patch.object(capture, "_capture_from_file") as file_capture, \
             patch.object(capture, "_capture_from_sitl", return_value=result) as sitl_capture:
            self.assertIs(capture.capture_flight_parameters(from_sitl=self.scenario), result)
            sitl_capture.assert_called_once_with(self.scenario, ROOT / "data/parameters/sitl_controller_gains.yaml",
                autopilots=None, work_dir=None, timeout=60.0, overwrite=False)
            capture.capture_flight_parameters(from_sitl=self.scenario, output=self.output,
                autopilots=["px4"], work_dir=self.root, timeout=15.0, overwrite=True)
            sitl_capture.assert_called_with(self.scenario, self.output,
                autopilots=["px4"], work_dir=self.root, timeout=15.0, overwrite=True)
            file_capture.assert_not_called()

    def test_python_api_also_rejects_ambiguous_or_incompatible_sources(self):
        cases = ({}, {"from_file": "flight.BIN", "from_sitl": self.scenario},
                 {"from_sitl": self.scenario, "merge": True},
                 {"from_sitl": self.scenario, "snapshot": "final"},
                 {"from_file": "flight.BIN", "output": self.output, "timeout": 60})
        with patch.object(capture, "_capture_from_file") as file_capture, \
             patch.object(capture, "_capture_from_sitl") as sitl_capture:
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    capture.capture_flight_parameters(**options)
            file_capture.assert_not_called()
            sitl_capture.assert_not_called()

    def test_sitl_cli_publishes_selected_stack_and_session_metadata(self):
        profile = create_params_sets(autopilots=["px4"])["px4"]
        work_dir = self.root / "sessions"
        previous_signal = signal.getsignal(signal.SIGTERM)
        with patch.object(capture, "_capture_stack", return_value=(profile, {})) as collect, \
             patch.object(capture, "DEFAULT_SITL_OUTPUT", self.output), \
             patch.object(capture, "_capture_from_file") as file_capture, redirect_stdout(io.StringIO()):
            status = capture.main(["--from-sitl", str(self.scenario),
                "--autopilots", "px4", "--work-dir", str(work_dir), "--timeout", "12"])
        self.assertEqual(status, 0)
        self.assertEqual(create_params_sets(self.output), {"px4": profile})
        metadata = json.loads(self.output.with_suffix(".meta.json").read_text())
        session = (self.output.parent / metadata["session_dir"]).resolve()
        self.assertEqual(session.parent, work_dir)
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(set(metadata["autopilots"]), {"px4"})
        self.assertEqual(json.loads((session / "capture.json").read_text())["session_dir"], ".")
        collect.assert_called_once_with("px4", self.scenario, session, 12.0)
        file_capture.assert_not_called()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_signal)

    def test_sitl_cli_interrupt_keeps_output_and_restores_signal_handler(self):
        self.output.write_text("previous profile")
        metadata = self.output.with_suffix(".meta.json")
        metadata.write_text("previous metadata")
        previous_signal = signal.getsignal(signal.SIGTERM)
        with patch.object(capture, "_capture_stack", side_effect=KeyboardInterrupt), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = capture.main(["--from-sitl", str(self.scenario), "--output", str(self.output),
                                   "--autopilots", "px4", "--overwrite"])
        self.assertEqual(status, 130)
        self.assertEqual(self.output.read_text(), "previous profile")
        self.assertEqual(metadata.read_text(), "previous metadata")
        report = next((self.root / "parameter_captures").glob("*/capture.json"))
        self.assertEqual(json.loads(report.read_text())["status"], "failed")
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_signal)

    def test_import_and_help_do_not_require_sitl_or_log_decoder_dependencies(self):
        # A ULog-only user must not need pymavlink or launcher dependencies just
        # to import the unified entrypoint; decoders are loaded on demand.
        code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('pymavlink', 'pyulog', 'tools.launcher')):
        raise AssertionError('unexpected eager dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools.scenario.capture_flight_parameters import main
main(['--help'])
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--from-file", result.stdout)
        self.assertIn("--from-sitl", result.stdout)


if __name__ == "__main__":
    unittest.main()
