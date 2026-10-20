"""Wind transport failures must prevent missions; missing settings stay compatible."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/launcher"))
import gazebo_wind as wind
import run_px4_gz_sitl as px4
import run_ardupilot_gz_sitl as ardu


class WindTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / "wind.json"
        self.env = {"GZ_PARTITION": "private-attempt", "GZ_IP": "127.0.0.1"}
        self.addCleanup(patch.stopall)
        self.clock = 0.0
        patch.object(wind.time, "monotonic", side_effect=lambda: self.clock).start()
        patch.object(wind.time, "sleep", side_effect=self.advance).start()
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def advance(self, seconds):
        self.clock += seconds

    def test_scenario_loaders_preserve_enu_and_validate_vectors(self):
        loaders = (px4._load_scenario_yaml_px4, ardu._load_scenario_yaml_ardupilot)
        for scenario, expected in (({}, None), ({"wind_m_s": [5, -2, 1]}, (5.0, -2.0, 1.0)),
                                   ({"wind_m_s": [0, 0, 0]}, (0.0, 0.0, 0.0))):
            (self.root / "scenario.yaml").write_text(px4.yaml.safe_dump({"common": {"scenario": scenario}}))
            for loader in loaders:
                self.assertEqual(loader(self.root).wind_m_s, expected)
        for bad in (None, "5 0 0", [], [1, 2], [1, 2, 3, 4], [True, 0, 0],
                    [float("nan"), 0, 0], [0, float("inf"), 0], ["five", 0, 0]):
            (self.root / "scenario.yaml").write_text(px4.yaml.safe_dump({"common": {"scenario": {"wind_m_s": bad}}}))
            for loader in loaders:
                with self.subTest(loader=loader.__name__, value=bad):
                    with self.assertRaisesRegex(ValueError, "wind_m_s"):
                        loader(self.root)

    def test_discovery_and_lost_publication_are_retried_in_same_partition(self):
        publications = []
        queries = []

        def transport(cmd, **kwargs):
            self.assertIs(kwargs["env"], self.env)
            if cmd[1] == "topic":
                self.assertEqual(cmd[cmd.index("-t") + 1], "/world/internal_name/wind")
                publications.append(cmd[cmd.index("-p") + 1])
                return subprocess.CompletedProcess(cmd, 0, "", "")
            queries.append(cmd)
            if len(queries) == 1:
                return subprocess.CompletedProcess(cmd, 1, "", "not discovered")
            # The first publication is lost; disabled/stale replies must not pass.
            text = ('linear_velocity { x: 5 y: -2 z: 1 } enable_wind: true'
                    if len(publications) >= 2 else 'linear_velocity { x: 9 }')
            return subprocess.CompletedProcess(cmd, 0, text, "")

        with patch.object(wind.subprocess, "run", side_effect=transport):
            wind.set_world_wind("internal_name", (5, -2, 1), env=self.env, log_path=self.log, timeout=5)
        self.assertEqual(len(publications), 2)
        self.assertIn("enable_wind: true", publications[-1])
        saved = json.loads(self.log.read_text())
        self.assertEqual(saved["status"], "acknowledged")
        self.assertEqual(saved["requested_wind_m_s"], [5, -2, 1])
        self.assertEqual(saved["acknowledged_wind_m_s"], [5, -2, 1])
        self.assertEqual(saved["partition"], "private-attempt")

    def test_explicit_zero_is_published_even_if_initial_target_is_zero(self):
        reply = subprocess.CompletedProcess([], 0, 'linear_velocity {}\nenable_wind: true\n', "")
        with patch.object(wind.subprocess, "run", return_value=reply) as run:
            wind.set_world_wind("world", (0, 0, 0), env=self.env, log_path=self.log)
        commands = [call.args[0] for call in run.call_args_list if call.args[0][1] == "topic"]
        self.assertEqual(len(commands), 1)
        self.assertIn("x: 0.0, y: 0.0, z: 0.0", commands[0][-1])
        self.assertEqual(json.loads(self.log.read_text())["acknowledged_wind_m_s"], [0, 0, 0])

    def test_unavailable_or_mismatched_service_fails_with_evidence(self):
        for response in ("Service call timed out", 'linear_velocity { x: 99 } enable_wind: true'):
            with self.subTest(response=response):
                self.clock = 0
                reply = subprocess.CompletedProcess([], 0, response, "")
                with patch.object(wind.subprocess, "run", return_value=reply):
                    with self.assertRaisesRegex(TimeoutError, "WindEffects"):
                        wind.set_world_wind("world", (5, 0, 0), env=self.env, log_path=self.log, timeout=0.4)
                self.assertEqual(json.loads(self.log.read_text())["status"], "failed")

    def test_shutdown_and_cancellation_stop_waiting(self):
        process = Mock(returncode=7)
        process.poll.return_value = 7
        for options, error, status in (({"process": process}, RuntimeError, "failed"),
                                       ({"cancelled": lambda: True}, InterruptedError, "interrupted")):
            with patch.object(wind.subprocess, "run") as run:
                with self.assertRaises(error):
                    wind.set_world_wind("world", (5, 0, 0), env=self.env, log_path=self.log, **options)
            run.assert_not_called()
            self.assertEqual(json.loads(self.log.read_text())["status"], status)

    def test_both_launchers_confirm_wind_before_mission_and_cleanup_on_failure(self):
        world = self.root / "world_ardupilot.sdf"
        world.write_text('<sdf><world name="actual_world"/></sdf>')
        for module in (px4, ardu):
            for target, fail in (((5, -2, 1), False), ((0, 0, 0), False), ((5, -2, 1), True)):
                with self.subTest(launcher=module.__name__, target=target, fail=fail), ExitStack() as stack:
                    events, handles, current = [], {}, {}
                    stack.enter_context(patch.dict(os.environ, self.env, clear=True))

                    def launch(name, command, **kwargs):
                        handle = Mock()
                        handle.proc.poll.return_value = None
                        handles[name] = handle
                        if name == "gazebo":
                            self.assertEqual(kwargs["env"]["GZ_PARTITION"], "private-attempt")
                        events.append(name)
                        return handle

                    def apply(world_name, velocity, **kwargs):
                        self.assertEqual(world_name, "actual_world")
                        self.assertEqual(velocity, target)
                        self.assertEqual(kwargs["env"]["GZ_PARTITION"], "private-attempt")
                        self.assertIs(kwargs["process"], handles["gazebo"].proc)
                        self.assertEqual(kwargs["log_path"], self.log)
                        events.append("wind")
                        if fail:
                            raise TimeoutError("WindEffects missing")

                    runner = Mock()
                    runner.is_done.return_value = True
                    runner.get_status.return_value.success = True
                    runner.start.side_effect = lambda: events.append("mission")
                    stack.enter_context(patch.object(module, "_popen", side_effect=launch))
                    finalize = stack.enter_context(patch.object(module, "_finalize_proc"))
                    stack.enter_context(patch.object(module, "set_world_wind", side_effect=apply))
                    factory = stack.enter_context(patch.object(module, "PX4MissionRunner" if module is px4 else "ArduPilotMissionRunner", return_value=runner))
                    kwargs = dict(instance=0, scenario_path=self.root / "scenario.yaml", logs_dir=self.root,
                                  gcs_outport=14550, mavlink_url="udp:127.0.0.1:14540", startup_delay_s=0,
                                  max_run_s=60, stop={"flag": False}, current=current, world=str(world),
                                  location="Purdue", headless=True, verbose=False, wind_m_s=target)
                    if module is px4:
                        sim = px4.Simulation(world, "actual_world", self.root / "model.sdf",
                                             "fire_px4vision", "fire_px4vision_0", self.env)
                        stack.enter_context(patch.object(px4, "prepare_simulation", return_value=sim))
                        stack.enter_context(patch.object(px4, "px4_command", return_value=["px4"]))
                        stack.enter_context(patch.object(px4, "spawn_model"))
                        kwargs.update(px4_dir=self.root, vehicle="4006", frame="gz_fire_px4vision")
                    else:
                        kwargs.update(ardupilot_dir=self.root, vehicle="ArduCopter", frame="gazebo-px4vision", model="JSON")
                    self.assertEqual(module.run_once(**kwargs), 1 if fail else 0)
                    if fail:
                        factory.assert_not_called()
                    else:
                        self.assertLess(events.index("wind"), events.index("mission"))
                    finalize.assert_any_call(handles["gazebo"])
                    self.assertTrue(all(handle is None for handle in current.values()))


if __name__ == "__main__":
    unittest.main()
