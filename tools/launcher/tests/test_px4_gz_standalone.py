import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import px4_gz_standalone as standalone
import run_px4_gz_sitl as batch


class StandaloneTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.px4 = self.root / "ap/px4"
        self.fire = self.root / "gz/FIRE_moonshot_gazebo"
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

    def test_fire_resolution_and_internal_world_name(self):
        self.write(self.px4 / "Tools/simulation/gz/models/fire_px4vision/model.sdf", '<sdf><model name="wrong_model"/></sdf>')
        sim = self.prepare()
        self.assertEqual(sim.world_name, "actual_world")
        self.assertEqual(sim.model_path, self.fire / "models/fire_px4vision/model.sdf")
        self.assertEqual(sim.entity_name, "fire_px4vision_2")
        self.assertEqual(sim.env["GZ_PARTITION"], "test-partition")

    def test_attach_clears_inherited_spawn_settings(self):
        with patch.dict(os.environ, {"PX4_SIM_MODEL": "gz_x500", "PX4_GZ_MODEL": "x500", "PX4_GZ_MODEL_POSE": "1,2"}):
            env = standalone.px4_environment(self.prepare(), "4006")
        self.assertEqual(env["PX4_GZ_MODEL_NAME"], "fire_px4vision_2")
        self.assertEqual(env["PX4_GZ_STANDALONE"], "1")
        self.assertNotIn("PX4_SIM_MODEL", env)
        self.assertNotIn("PX4_GZ_MODEL", env)

    def test_ardupilot_world_is_rejected(self):
        self.write(self.fire / "worlds/default_fire.sdf", '<sdf><world name="test"><include><plugin filename="ArduPilotPlugin"/></include></world></sdf>')
        with self.assertRaisesRegex(ValueError, "ArduPilotPlugin"):
            self.prepare()

    def test_spawn_rejects_negative_gazebo_reply(self):
        with patch.object(standalone, "wait_for_scene", return_value='name: "actual_world"'), patch.object(standalone, "_service", return_value=subprocess.CompletedProcess([], 0, "data: false", "")):
            with self.assertRaisesRegex(RuntimeError, "creation failed"):
                standalone.spawn_model(self.prepare(), (0, 0, 0.25, 0, 0, 0))

    def test_spawn_does_not_duplicate_existing_entity(self):
        with patch.object(standalone, "wait_for_scene", return_value='name: "fire_px4vision_2"'), patch.object(standalone, "_service") as service:
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                standalone.spawn_model(self.prepare(), (0, 0, 0.25, 0, 0, 0))
            service.assert_not_called()

    def test_world_ready_without_world_name_in_scene_message(self):
        reply = subprocess.CompletedProcess([], 0, 'ambient { r: 0.4 }\nmodel { name: "ground_plane" }', "")
        with patch.object(standalone, "_service", return_value=reply):
            self.assertEqual(standalone.wait_for_scene(self.prepare()), reply.stdout)

    def test_failed_run_with_only_raw_ulog_is_not_skipped(self):
        run = self.root / "run_000"
        self.write(run / "px4_logs/rootfs/log/failed.ulg", "incomplete")
        self.assertFalse(batch._prepare_run_dir(run, force=False))

    def test_wait_stops_when_server_exits(self):
        proc = Mock()
        proc.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "exited"):
            standalone.wait_for_scene(self.prepare(), process=proc)

    def test_binary_uses_requested_instance_and_separate_work_directory(self):
        build = self.px4 / "build/px4_sitl_default"
        for path in ("bin/px4", "etc/init.d-posix/rcS", "etc/init.d-posix/airframes/4006_gz_px4vision"):
            self.write(build / path, "")
        work = self.root / "run/rootfs"
        cmd = standalone.px4_command(self.px4, 2, work, "4006")
        self.assertEqual(cmd[cmd.index("-i") + 1], "2")
        self.assertEqual(cmd[cmd.index("-w") + 1], str(work))
        self.assertEqual(cmd[-1], str(build / "etc"))

    def test_batch_cleans_up_gazebo_when_spawn_fails(self):
        handle = Mock()
        current = dict(gz=None, sitl=None, gcs=None)
        with patch.object(batch, "prepare_simulation", return_value=self.prepare()), patch.object(batch, "px4_command", return_value=["px4"]), patch.object(batch, "_popen", return_value=handle), patch.object(batch, "spawn_model", side_effect=RuntimeError("spawn failed")), patch.object(batch, "_finalize_proc") as finalize:
            rc = batch.run_once(self.px4, 0, self.root / "scenario.yaml", self.root, 14550, "udp:127.0.0.1:14540", 0, 1, {"flag": False}, current, "4006", "gz_fire_px4vision", "default_fire", "Purdue", True, False)
        self.assertEqual(rc, 1)
        finalize.assert_any_call(handle)
        self.assertTrue(all(value is None for value in current.values()))

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
        server = Mock()
        server.wait.return_value = 0
        with patch.object(sys, "argv", ["standalone", "gazebo", "--headless"]), \
             patch.object(standalone, "prepare_simulation", return_value=self.prepare()), \
             patch.object(standalone.subprocess, "Popen", return_value=server) as launch, \
             patch.object(standalone, "spawn_model"), \
             patch.object(standalone.signal, "signal"), \
             patch.object(standalone, "stop_process"):
            self.assertEqual(standalone.main(), 0)
        launch.assert_called_once()
        self.assertIn("-s", launch.call_args.args[0])
        self.assertIn("--headless-rendering", launch.call_args.args[0])

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

    def test_batch_gui_starts_after_spawn_and_is_cleaned_up(self):
        for headless in (False, True):
            with self.subTest(headless=headless):
                sim = self.prepare()
                events, handles, current = [], {}, {}

                def launch(name, cmd, **kwargs):
                    events.append(name)
                    handle = Mock()
                    handle.proc.poll.return_value = None
                    handles[name] = handle
                    if name in ("gazebo", "gazebo_gui"):
                        self.assertEqual(kwargs["env"], sim.env)
                        self.assertIn("-s" if name == "gazebo" else "-g", cmd)
                    return handle

                runner = Mock()
                runner.is_done.return_value = True
                runner.get_status.return_value.success = True
                with patch.object(batch, "prepare_simulation", return_value=sim), \
                     patch.object(batch, "px4_command", return_value=["px4"]), \
                     patch.object(batch, "_popen", side_effect=launch), \
                     patch.object(batch, "spawn_model", side_effect=lambda *a, **k: events.append("spawn_ready")), \
                     patch.object(batch, "_run_qgc_cmd", return_value=["qgc"]), \
                     patch.object(batch, "PX4MissionRunner", return_value=runner), \
                     patch.object(batch, "_finalize_proc") as finalize:
                    rc = batch.run_once(self.px4, 0, self.root / "scenario.yaml", self.root,
                                        14550, "udp:127.0.0.1:14540", 0, 1, {"flag": False},
                                        current, "4006", "gz_fire_px4vision", "default_fire",
                                        "Purdue", headless, False)
                self.assertEqual(rc, 0)
                if headless:
                    self.assertNotIn("gazebo_gui", events)
                else:
                    self.assertEqual(events[:3], ["gazebo", "spawn_ready", "gazebo_gui"])
                    finalize.assert_any_call(handles["gazebo_gui"])
                self.assertTrue(all(value is None for value in current.values()))

    def test_collects_ulog_from_run_directory(self):
        run = self.root / "px4_logs"
        self.write(run / "sitl.log", "Opened full log file: ./log/date/flight.ulg\n")
        self.write(run / "rootfs/log/date/flight.ulg", "ulog data")
        self.assertTrue(batch._collect_px4_logs(run))
        self.assertEqual((run / "flight.ulg").read_text(), "ulog data")


if __name__ == "__main__":
    unittest.main()
