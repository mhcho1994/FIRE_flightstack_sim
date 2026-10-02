"""Run: python3 -B -m unittest discover -s tests/launcher -v."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/launcher"))

import attempt_logs
import batch_timing
import run_px4_gz_sitl as px4
import run_ardupilot_gz_sitl as ardu


class RetryLogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "run_048"
        self.write(self.run / "scenario.yaml", "common: {}\n")
        self.write(self.run / "generated/vehicle/model.sdf", "model snapshot")

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def read_result(self, path):
        return json.loads((path / "result.json").read_text())

    def run_batch(self, module, outcomes, *, retries=None, force=False, fail_archive=False,
                  fail_collection=False, inter_run_delay=None):
        kind = "px4" if module is px4 else "ardu"
        config = (px4.PX4ScenarioConfig(px4_dir=str(self.root / "ap/px4")) if module is px4
                  else ardu.ArdupilotScenarioConfig(ardupilot_dir=str(self.root / "ap/ardupilot")))
        calls = []

        def run_once(**kwargs):
            index = len(calls)
            self.assertLess(index, len(outcomes), "Unexpected extra attempt")
            calls.append(kwargs)
            outcome = outcomes[index]
            logs = kwargs["logs_dir"]
            self.assertFalse(attempt_logs.flight_logs(logs), "Previous attempt contaminated the next flight")
            for name in ("sitl.log", "gazebo.log", "gcs.log", "gazebo_gui.log"):
                text = f"attempt {index + 1}: {name}\n"
                if module is px4 and name == "sitl.log":
                    text += "Opened full log file: ./log/date/flight.ulg\n"
                self.write(logs / name, text)
            self.write(logs / ("rootfs/parameters.bson" if module is px4 else "eeprom.bin"), "runtime state")
            if outcome != "missing_log":
                relative = "rootfs/log/date/flight.ulg" if module is px4 else "logs/00000001.BIN"
                self.write(logs / relative, f"flight {index + 1}")
            if outcome == "signal":
                kwargs["stop"]["flag"] = True
                return 0
            return 0 if outcome == "missing_log" else outcome

        argv = ["batch", "--run-root", str(self.root), "--headless", "--max-retries",
                str(max(len(outcomes) - 1, 0) if retries is None else retries)]
        if force:
            argv.append("--force")
        if inter_run_delay is not None:
            argv.extend(["--inter-run-delay", str(inter_run_delay)])
        loader = "_load_scenario_yaml_px4" if module is px4 else "_load_scenario_yaml_ardupilot"
        builder = "ensure_px4_built" if module is px4 else "_ensure_ardupilot_built"
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.object(module.signal, "signal"))
            stack.enter_context(patch.object(module, loader, return_value=config))
            stack.enter_context(patch.object(module, builder))
            stack.enter_context(patch.object(module, "run_once", side_effect=run_once))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if fail_collection:
                collector = "_collect_px4_logs" if module is px4 else "_check_ardupilot_logs"
                stack.enter_context(patch.object(module, collector, side_effect=OSError("collection failed")))
            if fail_archive:
                copytree = attempt_logs.shutil.copytree

                def fail_copy(src, dst, *args, **kwargs):
                    if Path(src) == self.run / f"{kind}_logs":
                        self.write(Path(dst) / "sitl.log", "partially copied output")
                        raise OSError("disk full")
                    return copytree(src, dst, *args, **kwargs)

                stack.enter_context(patch.object(attempt_logs.shutil, "copytree", side_effect=fail_copy))
            rc = module.main()
        return rc, calls, self.run / f"{kind}_logs", self.run / f"{kind}_attempts"

    def test_delay_before_every_actual_attempt_and_not_after_last(self):
        for module, kind in ((px4, "px4"), (ardu, "ardu")):
            with self.subTest(autopilot=kind):
                # Completed runs before, between, and after actual runs must not add waits.
                for name in ("run_047", "run_049", "run_051"):
                    relative = "flight.ulg" if module is px4 else "logs/00000001.BIN"
                    self.write(self.root / name / f"{kind}_logs" / relative, "legacy success")
                self.write(self.root / "run_050/scenario.yaml", "common: {}\n")
                clock = [0.0]
                waits = {0: 0.0, 1: 0.0, 2: 0.0}

                def sleep(seconds):
                    count = module.run_once.call_count
                    self.assertIn(count, waits, "Wait occurred after the last attempt")
                    if count > 0:
                        logs = module.run_once.call_args.kwargs["logs_dir"]
                        result = self.read_result(logs)
                        self.assertEqual(result["status"], "timeout" if count == 1 else "success")
                        if count == 1:
                            self.assertTrue((logs.parent / result["archived_to"] / "result.json").is_file())
                    waits[count] += seconds
                    clock[0] += seconds

                with patch.object(batch_timing.time, "monotonic", side_effect=lambda: clock[0]), \
                     patch.object(batch_timing.time, "sleep", side_effect=sleep):
                    rc, calls, _, _ = self.run_batch(module, [124, 0, 0], retries=1, inter_run_delay=3)
                self.assertEqual(rc, 0)
                self.assertEqual([call["logs_dir"].parent.name for call in calls],
                                 ["run_048", "run_048", "run_050"])
                for seconds in waits.values():
                    self.assertAlmostEqual(seconds, 3.0)
                self.assertAlmostEqual(clock[0], 9.0)

    def test_default_and_zero_delay_do_not_sleep(self):
        for module in (px4, ardu):
            for delay in (None, 0):
                with self.subTest(autopilot=module.__name__, delay=delay):
                    with patch.object(batch_timing.time, "sleep") as sleep:
                        rc, calls, _, _ = self.run_batch(module, [124, 0], force=True, inter_run_delay=delay)
                    self.assertEqual(rc, 0)
                    self.assertEqual(len(calls), 2)
                    sleep.assert_not_called()

    def test_interrupt_during_initial_or_retry_delay_prevents_launch(self):
        for module in (px4, ardu):
            for completed_attempts in (0, 1):
                with self.subTest(autopilot=module.__name__, completed_attempts=completed_attempts):
                    clock = [0.0]

                    def sleep(seconds):
                        if module.run_once.call_count == completed_attempts:
                            handler = module.signal.signal.call_args_list[0].args[1]
                            handler(module.signal.SIGINT, None)
                        else:
                            clock[0] += seconds

                    with patch.object(batch_timing.time, "monotonic", side_effect=lambda: clock[0]), \
                         patch.object(batch_timing.time, "sleep", side_effect=sleep):
                        rc, calls, logs, archives = self.run_batch(
                            module, [124], retries=3, force=True, inter_run_delay=3,
                        )
                    self.assertEqual(rc, 130)
                    self.assertEqual(len(calls), completed_attempts)
                    if completed_attempts:
                        self.assertEqual(self.read_result(logs)["status"], "timeout")
                        self.assertEqual(len(list(archives.glob("*/attempt_*"))), 1)
                    else:
                        self.assertFalse((logs / "result.json").exists())
                        self.assertFalse(archives.exists())

    def test_invalid_inter_run_delays_are_rejected_before_launch(self):
        for module in (px4, ardu):
            for delay in ("-1", "nan", "inf"):
                with self.subTest(autopilot=module.__name__, delay=delay):
                    argv = ["batch", f"--inter-run-delay={delay}"]
                    with patch.object(sys, "argv", argv), patch.object(module, "run_once") as run, \
                         redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        module.main()
                    self.assertEqual(error.exception.code, 2)
                    run.assert_not_called()

    def test_first_success_keeps_existing_layout_without_attempts(self):
        for module in (px4, ardu):
            with self.subTest(autopilot=module.__name__):
                rc, calls, logs, archives = self.run_batch(module, [0], retries=0)
                self.assertEqual(rc, 0)
                self.assertEqual(len(calls), 1)
                self.assertEqual(self.read_result(logs)["status"], "success")
                self.assertFalse(archives.exists())
                self.assertEqual(attempt_logs.flight_logs(logs)[0].read_text(), "flight 1")
                # A subsequent invocation recognizes the completed result.
                with patch.object(batch_timing.time, "sleep") as sleep:
                    rc, calls, _, _ = self.run_batch(module, [], inter_run_delay=3)
                self.assertEqual(rc, 0)
                self.assertEqual(calls, [])
                sleep.assert_not_called()

    def test_failures_are_archived_before_success_without_mixing_flight_logs(self):
        for module in (px4, ardu):
            with self.subTest(autopilot=module.__name__):
                rc, calls, logs, archives = self.run_batch(module, [124, 1, 0])
                self.assertEqual(rc, 0)
                self.assertEqual(len(calls), 3)
                sessions = list(archives.iterdir())
                self.assertEqual(len(sessions), 1)
                session = sessions[0]
                self.assertEqual((session / "scenario.yaml").read_text(), "common: {}\n")
                self.assertEqual((session / "generated/vehicle/model.sdf").read_text(), "model snapshot")
                for n, status in ((1, "timeout"), (2, "failed")):
                    attempt = session / f"attempt_{n:03d}"
                    result = self.read_result(attempt)
                    self.assertEqual(result["status"], status)
                    self.assertEqual(result["attempt"], n)
                    self.assertIn("started_at", result)
                    self.assertIn("ended_at", result)
                    self.assertEqual((attempt / "sitl.log").read_text().splitlines()[0], f"attempt {n}: sitl.log")
                    for relative in result["flight_logs"]:
                        self.assertEqual((attempt / relative).read_text(), f"flight {n}")
                self.assertFalse((session / "attempt_003").exists())
                self.assertEqual(self.read_result(logs)["status"], "success")
                self.assertEqual([p.read_text() for p in attempt_logs.flight_logs(logs)], ["flight 3"])
                runtime = logs / ("rootfs/parameters.bson" if module is px4 else "eeprom.bin")
                self.assertEqual(runtime.read_text(), "runtime state")

    def test_interruption_is_archived_and_does_not_count_as_completed(self):
        for module in (px4, ardu):
            for outcome in (130, "signal"):
                with self.subTest(autopilot=module.__name__, outcome=outcome):
                    rc, calls, logs, archives = self.run_batch(module, [outcome], retries=3)
                    self.assertEqual(rc, 130)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(self.read_result(logs)["status"], "interrupted")
                    self.assertEqual(attempt_logs.flight_logs(logs), [])
                    self.assertFalse(attempt_logs.prepare_run_logs(logs, force=False))
                    self.assertTrue(list(archives.glob("*/attempt_001/result.json")))

    def test_all_failed_and_missing_logs_are_recorded_even_without_retries(self):
        for module in (px4, ardu):
            for outcome, status in ((124, "timeout"), (1, "failed"), ("missing_log", "missing_log")):
                with self.subTest(autopilot=module.__name__, outcome=outcome):
                    rc, calls, logs, archives = self.run_batch(module, [outcome], retries=0)
                    self.assertNotEqual(rc, 0)
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(self.read_result(logs)["status"], status)
                    self.assertEqual(attempt_logs.flight_logs(logs), [])
                    self.assertFalse(attempt_logs.prepare_run_logs(logs, force=False))
                    self.assertTrue(list(archives.glob("*/attempt_001/sitl.log")))

    def test_archive_failure_stops_retries_and_preserves_original_evidence(self):
        for module in (px4, ardu):
            with self.subTest(autopilot=module.__name__):
                rc, calls, logs, archives = self.run_batch(module, [124], retries=3, fail_archive=True)
                self.assertEqual(rc, 2)
                self.assertEqual(len(calls), 1)
                self.assertEqual(attempt_logs.flight_logs(logs)[0].read_text(), "flight 1")
                self.assertTrue((logs / "sitl.log").exists())
                self.assertNotIn("archived_to", self.read_result(logs))
                self.assertTrue(list(archives.glob("*/attempt_001.partial/sitl.log")))
                self.assertFalse(list(archives.glob("*/attempt_001")))
                self.assertFalse(attempt_logs.prepare_run_logs(logs, force=False))
                self.assertTrue(list(archives.glob("*/attempt_001/result.json")))
                self.assertEqual(attempt_logs.flight_logs(logs), [])

    def test_collection_failure_preserves_raw_flight_and_error(self):
        for module in (px4, ardu):
            with self.subTest(autopilot=module.__name__):
                rc, calls, logs, archives = self.run_batch(module, [0], retries=0, fail_collection=True)
                self.assertEqual(rc, 1)
                self.assertEqual(len(calls), 1)
                saved = next(archives.glob("*/attempt_001"))
                result = self.read_result(saved)
                self.assertEqual(result["status"], "missing_log")
                self.assertEqual(result["collection_error"], "collection failed")
                relative = "rootfs/log/date/flight.ulg" if module is px4 else "logs/00000001.BIN"
                self.assertIn(relative, result["flight_logs"])
                self.assertEqual((saved / relative).read_text(), "flight 1")
                self.assertEqual(attempt_logs.flight_logs(logs), [])

    def test_force_keeps_previous_archives_and_new_invocations_use_new_sessions(self):
        for module in (px4, ardu):
            with self.subTest(autopilot=module.__name__):
                _, _, logs, archives = self.run_batch(module, [124, 0])
                previous = set(archives.glob("*/attempt_001/result.json"))
                contents = {p: p.read_bytes() for p in previous}
                rc, _, _, _ = self.run_batch(module, [0], force=True)
                self.assertEqual(rc, 0)
                self.assertEqual(set(archives.glob("*/attempt_001/result.json")), previous)
                rc, _, _, _ = self.run_batch(module, [124, 0], force=True)
                self.assertEqual(rc, 0)
                self.assertEqual(len(list(archives.glob("*/attempt_001/result.json"))), 2)
                for path, data in contents.items():
                    self.assertEqual(path.read_bytes(), data)

    def test_abruptly_interrupted_output_is_recovered_before_rerun(self):
        for kind, relative in (("px4", "rootfs/log/date/raw.ulg"), ("ardu", "logs/00000001.BIN")):
            with self.subTest(autopilot=kind):
                logs = self.run / f"{kind}_logs"
                history = attempt_logs.AttemptLogs(logs)
                history.begin(1)
                self.write(logs / relative, "unfinished flight")
                self.write(logs / "sitl.log", "partial output")
                self.assertFalse(attempt_logs.prepare_run_logs(logs, force=True))
                self.assertFalse(logs.exists())
                saved = next((self.run / f"{kind}_attempts").glob("*/attempt_001"))
                self.assertEqual((saved / relative).read_text(), "unfinished flight")
                self.assertEqual(self.read_result(saved)["status"], "incomplete")

    def test_snapshot_keeps_firmware_symlinks_without_copying_targets(self):
        logs = self.run / "px4_logs"
        history = attempt_logs.AttemptLogs(logs)
        history.begin(1)
        target = self.write(self.root / "firmware/etc/config", "outside runtime tree").parent
        link = logs / "rootfs/etc"
        link.parent.mkdir(parents=True)
        link.symlink_to(target, target_is_directory=True)
        self.write(logs / "rootfs/log/date/raw.ulg", "uncollected flight")
        history.finish(1, False)
        saved = self.run / self.read_result(logs)["archived_to"]
        self.assertTrue((saved / "rootfs/etc").is_symlink())
        self.assertEqual((saved / "rootfs/log/date/raw.ulg").read_text(), "uncollected flight")
        self.assertEqual((target / "config").read_text(), "outside runtime tree")

    def test_legacy_success_logs_still_skip_without_creating_archives(self):
        for kind, relative in (("px4", "flight.ulg"), ("ardu", "logs/00000001.BIN")):
            with self.subTest(autopilot=kind):
                logs = self.run / f"{kind}_logs"
                self.write(logs / relative, "legacy flight")
                self.assertTrue(attempt_logs.prepare_run_logs(logs, force=False))
                self.assertFalse((self.run / f"{kind}_attempts").exists())

    def test_ardupilot_startup_failure_stops_processes_before_archiving(self):
        current = {"gz": None, "gcs": None, "sitl": None}
        server = Mock()
        with patch.object(ardu, "_popen", side_effect=[server, OSError("GCS failed")]), \
             patch.object(ardu, "_finalize_proc") as finalize, \
             patch.object(ardu.time, "sleep"), redirect_stdout(io.StringIO()):
            rc = ardu.run_once(self.root, 0, self.run / "scenario.yaml", self.run / "ardu_logs",
                               14551, "udp:127.0.0.1:14550", 0, 60, {"flag": False},
                               current, "ArduCopter", "gazebo-iris", "JSON", "default", "Purdue", True, False)
        self.assertEqual(rc, 1)
        finalize.assert_any_call(server)
        self.assertTrue(all(handle is None for handle in current.values()))


if __name__ == "__main__":
    unittest.main()
