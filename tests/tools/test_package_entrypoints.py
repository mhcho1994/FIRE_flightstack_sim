"""Exercise CLI imports in fresh processes without an inherited PYTHONPATH."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = (
    "tools.commander.fake_gcs_heartbeat",
    "tools.commander.pymavlink_px4_commander",
    "tools.commander.pymavlink_ardupilot_commander",
    "tools.launcher.run_px4_gz_sitl",
    "tools.launcher.run_ardupilot_gz_sitl",
    "tools.launcher.mission_wind_controller",
    "tools.scenario.scenario_generator",
    "tools.scenario.create_params_sets",
    "tools.scenario.capture_flight_parameters",
    "tools.plotting.plot_sitl_run_comparison",
    "tools.plotting.plot_sitl_autopilot_response",
    "tools.postprocessor.sitl_log_converter",
)


class PackageEntrypointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.env = os.environ.copy()
        self.env.pop("PYTHONPATH", None)
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.env["MPLCONFIGDIR"] = str(self.directory / "matplotlib")

    def assert_help(self, arguments, cwd):
        result = subprocess.run(
            [sys.executable, "-B", *arguments, "--help"],
            cwd=cwd, env=self.env, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("usage:", result.stdout.lower())

    def test_module_execution_from_repository_root(self):
        for module in ENTRYPOINTS:
            with self.subTest(module=module):
                self.assert_help(["-m", module], ROOT)

    def test_direct_file_execution_outside_repository(self):
        for module in ENTRYPOINTS:
            with self.subTest(module=module):
                script = ROOT / (module.replace(".", "/") + ".py")
                self.assert_help([str(script)], self.directory)
