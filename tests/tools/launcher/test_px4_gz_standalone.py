"""Run: python3 -B -m unittest discover -s tests/tools/launcher -v."""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/launcher"))
import px4_gz_standalone as standalone


class StandaloneTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.px4 = self.root / "ap/px4"
        self.build = self.px4 / "build/px4_sitl_default"
        self.fire = self.root / "gz/FIRE_moonshot_gazebo"
        self.write(self.px4 / "Makefile", "")
        self.create_build()
        self.write(self.px4 / "src/modules/simulation/gz_bridge/server.config", "<server_config/>")
        self.write(self.fire / "worlds/default_fire.sdf", '<sdf><world name="actual_world"/></sdf>')
        self.write(self.fire / "models/fire_px4vision/model.sdf", '<sdf><model name="fire_px4vision"/></sdf>')
        self.addCleanup(patch.stopall)
        patch.object(standalone, "PROJECT_ROOT", self.root).start()
        patch.dict(os.environ, {}, clear=True).start()

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def prepare(self):
        return standalone.prepare_simulation(self.px4, "default_fire", "fire_px4vision", 2, "test-partition")

    def create_build(self, *args, **kwargs):
        for path in ("bin/px4", "etc/init.d-posix/rcS", "etc/init.d-posix/airframes/4006_gz_px4vision"):
            self.write(self.build / path, "")
        (self.build / "src/modules/simulation/gz_plugins").mkdir(parents=True, exist_ok=True)

    def test_px4_cli_builds_before_attaching(self):
        shutil.rmtree(self.build)
        with patch.object(sys, "argv", ["standalone", "px4"]), \
             patch.object(standalone.subprocess, "run", side_effect=self.create_build) as make, \
             patch.object(standalone, "wait_for_scene"), \
             patch.object(standalone.os, "execvpe", side_effect=SystemExit(0)) as launch:
            with self.assertRaises(SystemExit) as result:
                standalone.main()
        self.assertEqual(result.exception.code, 0)
        make.assert_called_once()
        self.assertEqual(launch.call_args.args[0], str(self.build / "bin/px4"))
        self.assertIn(str(self.build / "src/modules/simulation/gz_plugins"),
                      launch.call_args.args[2]["GZ_SIM_SYSTEM_PLUGIN_PATH"].split(":"))

    def test_gui_mode_runs_server_then_spawn_then_gui(self):
        sim = self.prepare()
        server, gui = Mock(), Mock()
        server.poll.return_value = None
        server.returncode = None
        gui.poll.return_value = 0
        gui.returncode = 0
        events = []

        def launch(cmd, **kwargs):
            self.assertEqual(kwargs["env"], sim.env)
            self.assertTrue(kwargs["start_new_session"])
            if "-g" in cmd:
                events.append("gui")
                return gui
            self.assertIn("-s", cmd)
            self.assertNotIn("--headless-rendering", cmd)
            events.append("server")
            return server

        with patch.object(sys, "argv", ["standalone", "gazebo"]), \
             patch.object(standalone, "prepare_simulation", return_value=sim), \
             patch.object(standalone.subprocess, "Popen", side_effect=launch), \
             patch.object(standalone, "spawn_model", side_effect=lambda *a, **k: events.append("spawn_ready")), \
             patch.object(standalone.signal, "signal"), \
             patch.object(standalone, "stop_process") as stop:
            self.assertEqual(standalone.main(), 0)
        self.assertEqual(events, ["server", "spawn_ready", "gui"])
        self.assertEqual([call.args[0] for call in stop.call_args_list], [gui, server])

    def test_headless_cli_never_launches_gui(self):
        shutil.rmtree(self.build)
        server = Mock()
        server.wait.return_value = 0
        with patch.object(sys, "argv", ["standalone", "gazebo", "--headless"]), \
             patch.object(standalone.subprocess, "run", side_effect=self.create_build) as make, \
             patch.object(standalone.subprocess, "Popen", return_value=server) as launch, \
             patch.object(standalone, "spawn_model"), \
             patch.object(standalone.signal, "signal"), \
             patch.object(standalone, "stop_process"):
            self.assertEqual(standalone.main(), 0)
        make.assert_called_once()
        launch.assert_called_once()
        self.assertIn("-s", launch.call_args.args[0])
        self.assertIn("--headless-rendering", launch.call_args.args[0])
        self.assertIn(str(self.build / "src/modules/simulation/gz_plugins"),
                      launch.call_args.kwargs["env"]["GZ_SIM_SYSTEM_PLUGIN_PATH"].split(":"))

    def test_cli_stops_server_if_gui_launch_fails(self):
        server = Mock()
        with patch.object(sys, "argv", ["standalone", "gazebo"]), \
             patch.object(standalone, "prepare_simulation", return_value=self.prepare()), \
             patch.object(standalone.subprocess, "Popen", side_effect=[server, OSError("GUI launch failed")]), \
             patch.object(standalone, "spawn_model"), \
             patch.object(standalone.signal, "signal"), \
             patch.object(standalone, "stop_process") as stop:
            with self.assertRaisesRegex(OSError, "GUI launch failed"):
                standalone.main()
        stop.assert_called_once_with(server)


if __name__ == "__main__":
    unittest.main()
