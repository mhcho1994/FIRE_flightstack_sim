"""SITL capture is read-only, complete, isolated, and fails without publishing."""

from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml
from pymavlink import mavutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.scenario import capture_flight_parameters as capture
from tools.scenario.create_params_sets import create_params_sets, resolve_params_sets


class Connection:
    target_system = 1
    target_component = 1

    def __init__(self, autopilot, values):
        self.values = dict(values)
        self.values.update({name: 1.0 for name in capture.CONTEXT_PARAMETERS[autopilot]})
        self.queue = deque()
        self.reads = Counter()
        self.change = None
        self.param_type = mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        ap = (mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA if autopilot == "ardupilot"
              else mavutil.mavlink.MAV_AUTOPILOT_PX4)
        self.heartbeat = self.message("HEARTBEAT", base_mode=0, autopilot=ap)
        self.mav = SimpleNamespace(param_request_read_send=self.read, param_set_send=Mock(),
                                   command_long_send=Mock())

    def message(self, kind, **fields):
        return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: 1,
                               get_srcComponent=lambda: 1, **fields)

    def read(self, system, component, encoded, index):
        name = encoded.decode()
        self.reads[name] += 1
        if name in self.values:
            value = self.values[name]
            if self.change == name and self.reads[name] > 1:
                value += 1e-8
            self.queue.append(self.message("PARAM_VALUE", param_id=name, param_value=value,
                                           param_type=self.param_type))

    def recv_match(self, **kwargs):
        return self.queue.popleft() if self.queue else self.heartbeat


class CaptureFlightParametersSITLTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profiles = create_params_sets()
        identity = patch.object(capture, 'read_firmware_identity', side_effect=lambda connection, ap, **kw:
            dict(autopilot=ap, version='4.6.2' if ap == 'ardupilot' else '1.16.2', git_hash='abcdef01'))
        identity.start()
        self.addCleanup(identity.stop)
        self.scenario = self.root / "scenario.yaml"
        self.data = {"autopilots": {ap: {"sim": {
            f"{ap}_dir": str(ROOT / "ap" / ap), "instance": 0,
            "vehicle": "ArduCopter" if ap == "ardupilot" else 4006,
            "frame": "gazebo-px4vision" if ap == "ardupilot" else "gz_fire_px4vision",
            "model": "JSON", "world": "default_fire_px4vision" if ap == "ardupilot" else "default_fire",
            "location": "Purdue",
        }} for ap in capture.GAIN_GROUPS}}
        self.scenario.write_text(yaml.safe_dump(self.data))

    def read(self, connection, autopilot):
        return capture.read_controller_parameters(connection, autopilot, timeout=.002, attempts=1)

    def test_all_groups_are_read_twice_and_export_without_overrides(self):
        for ap, count in (("ardupilot", 23), ("px4", 20)):
            with self.subTest(autopilot=ap):
                connection = Connection(ap, self.profiles[ap]["baseline"])
                profile, context = self.read(connection, ap)
                self.assertEqual(len(profile["baseline"]), count)
                self.assertEqual(profile["groups"], list(capture.GAIN_GROUPS[ap]))
                self.assertEqual(profile["parameters"], {})
                self.assertEqual(set(context), set(capture.CONTEXT_PARAMETERS[ap]))
                self.assertTrue(all(count == 2 for count in connection.reads.values()))
                connection.mav.param_set_send.assert_not_called()
                connection.mav.command_long_send.assert_not_called()
                resolved = resolve_params_sets({ap: profile}, seed=42, run_id=0, run_dir=self.root)
                self.assertEqual(resolved[ap]["values"], profile["baseline"])
                self.assertTrue((self.root / resolved[ap]["parameter_file"]).is_file())

    def test_47_sitl_capture_requests_native_names_without_legacy_aliases(self):
        from tools.commander.fc_parameter_registry import ARDU47
        native = {name: .05 for names in ARDU47.values() for name in names}
        connection = Connection('ardupilot', native)
        with patch.object(capture, 'read_firmware_identity', return_value=dict(
                autopilot='ardupilot', version='4.7.0', git_hash='c068bfcd')):
            profile, _ = self.read(connection, 'ardupilot')
        self.assertEqual(profile['baseline'], native)
        self.assertEqual(profile['parameter_schema'], 'arducopter-4.7')
        self.assertEqual(set(connection.reads), set(native))
        self.assertNotIn('PSC_ACCZ_P', connection.reads)
        self.assertTrue(all(count == 2 for count in connection.reads.values()))
        connection.mav.param_set_send.assert_not_called()

    def test_missing_changing_wrong_type_and_nonfinite_values_fail(self):
        for failure in ("missing", "changing", "type", "nonfinite"):
            with self.subTest(failure=failure):
                connection = Connection("px4", self.profiles["px4"]["baseline"])
                name = "MC_ROLLRATE_P"
                if failure == "missing":
                    del connection.values[name]
                elif failure == "changing":
                    connection.change = name
                elif failure == "type":
                    connection.param_type = mavutil.mavlink.MAV_PARAM_TYPE_INT32
                else:
                    connection.values[name] = float("nan")
                with self.assertRaises((TimeoutError, ValueError, RuntimeError)):
                    self.read(connection, "px4")
                connection.mav.param_set_send.assert_not_called()

    def test_armed_wrong_autopilot_and_cancellation_fail(self):
        for failure in ("armed", "autopilot", "cancelled"):
            with self.subTest(failure=failure):
                connection = Connection("px4", self.profiles["px4"]["baseline"])
                if failure == "armed":
                    connection.heartbeat.base_mode = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                elif failure == "autopilot":
                    connection.heartbeat.autopilot = mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA
                check = Mock(side_effect=InterruptedError("cancelled")) if failure == "cancelled" else lambda: None
                with self.assertRaises((RuntimeError, InterruptedError)):
                    capture.read_controller_parameters(connection, "px4", timeout=.002, attempts=1, check_stop=check)
                self.assertEqual(connection.reads, {})

    def test_success_publishes_profile_metadata_and_uses_fresh_sessions(self):
        output = self.root / "profiles with spaces/sitl_controller_gains.yaml"
        sessions = []
        originals = {}
        firmware = self.root / "firmware with spaces"
        binary = firmware / "bin/autopilot"
        startup = [firmware / "defaults.parm", firmware / "frame.parm"]
        world = self.root / "worlds/default.sdf"

        def collect(ap, scenario, session, timeout):
            self.assertEqual(scenario, self.scenario)
            sessions.append(session)
            runtime = session / ap / "rootfs"
            if ap == "ardupilot":
                command = [str(binary), "-w", "--defaults", ",".join(map(str, startup)),
                           "--home", "40,-86,0,0", "--sim-address=127.0.0.1"]
            else:
                command = [str(binary), "-w", str(runtime), "-s", "etc/init.d-posix/rcS", str(firmware / "etc")]
            report = dict(clean_storage=True, command=command, command_cwd=str(session / ap),
                          connection="tcp:127.0.0.1:5760", world=str(world),
                          firmware=dict(directory=str(firmware), binary=str(binary),
                                        startup_files={str(path): "hash" for path in startup}))
            originals[ap] = report
            return self.profiles[ap], report

        with patch.object(capture, "_capture_stack", side_effect=collect):
            capture.capture_flight_parameters(from_sitl=self.scenario, output=output)
            first = output.read_bytes()
            capture.capture_flight_parameters(from_sitl=self.scenario, output=output, overwrite=True)
        self.assertEqual(first, output.read_bytes())
        self.assertEqual(sessions[0], sessions[1])
        self.assertNotEqual(sessions[0], sessions[2])
        report = json.loads(output.with_suffix(".meta.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["profile_sha256"], hashlib.sha256(first).hexdigest())
        self.assertEqual(create_params_sets(output), self.profiles)
        for destination in (output.with_suffix(".meta.json"), sessions[2] / "capture.json"):
            metadata = json.loads(destination.read_text())
            self.assertEqual(metadata["paths_relative_to"], "metadata_directory")

            def points_to(value, expected):
                self.assertFalse(Path(value).is_absolute())
                self.assertEqual((destination.parent / value).resolve(), expected)

            points_to(metadata["scenario"], self.scenario)
            points_to(metadata["session_dir"], sessions[2])
            for ap, entry in metadata["autopilots"].items():
                points_to(entry["firmware"]["directory"], firmware)
                points_to(entry["firmware"]["binary"], binary)
                self.assertEqual(list(entry["firmware"]["startup_files"].values()), ["hash", "hash"])
                for actual, expected in zip(entry["firmware"]["startup_files"], startup):
                    points_to(actual, expected)
                points_to(entry["world"], world)
                points_to(entry["command_cwd"], sessions[2] / ap)
                points_to(entry["command"][0], binary)
                self.assertEqual(entry["connection"], "tcp:127.0.0.1:5760")
                if ap == "ardupilot":
                    for actual, expected in zip(entry["command"][3].split(","), startup):
                        points_to(actual, expected)
                    self.assertEqual(entry["command"][4:], ["--home", "40,-86,0,0", "--sim-address=127.0.0.1"])
                else:
                    points_to(entry["command"][2], sessions[2] / ap / "rootfs")
                    points_to(entry["command"][4], sessions[2] / ap / "rootfs/etc/init.d-posix/rcS")
                    points_to(entry["command"][5], firmware / "etc")
        # Serialization must not rebase the runtime argv or in-memory source paths.
        for entry in originals.values():
            self.assertEqual(entry["command"][0], str(binary))
            self.assertEqual(entry["firmware"]["directory"], str(firmware))
        with patch.object(capture, "_capture_stack") as collect:
            with self.assertRaises(FileExistsError):
                capture.capture_flight_parameters(from_sitl=self.scenario, output=output)
            collect.assert_not_called()

    def test_partial_failure_preserves_previous_output_and_metadata(self):
        output = self.root / "sitl_controller_gains.yaml"
        metadata = output.with_suffix(".meta.json")
        for path in (output, metadata):
            path.write_text("previous content")
        def collect(ap, *args):
            if ap == "px4":
                raise TimeoutError("missing PX4")
            return self.profiles[ap], {}
        with patch.object(capture, "_capture_stack", side_effect=collect):
            with self.assertRaisesRegex(TimeoutError, "missing PX4"):
                capture.capture_flight_parameters(from_sitl=self.scenario, output=output, overwrite=True)
        for path in (output, metadata):
            self.assertEqual(path.read_text(), "previous content")
        session_report = next((self.root / "parameter_captures").glob("*/capture.json"))
        report = json.loads(session_report.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["session_dir"], ".")
        self.assertEqual((session_report.parent / report["scenario"]).resolve(), self.scenario)

    def test_configured_gains_and_invalid_options_fail_before_launch(self):
        self.data["autopilots"]["px4"]["controller_gains"] = {}
        self.scenario.write_text(yaml.safe_dump(self.data))
        with patch.object(capture, "_capture_stack") as collect:
            with self.assertRaisesRegex(ValueError, "without --controller-gains"):
                capture._capture_from_sitl(self.scenario, self.root / "output.yaml")
            collect.assert_not_called()
        self.assertFalse((self.root / "parameter_captures").exists())
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                capture._capture_from_sitl(self.scenario, self.root / "output.yaml", timeout=timeout)

    def test_occupied_port_is_rejected(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as occupied:
            occupied.bind(("127.0.0.1", 0))
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                capture._check_ports([(socket.SOCK_DGRAM, occupied.getsockname()[1])])

    def test_capture_processes_stop_on_read_failure_without_starting_mission(self):
        from tools.launcher import run_px4_gz_sitl as px4
        from tools.launcher import run_ardupilot_gz_sitl as ardu
        cfg = SimpleNamespace(px4_dir=self.root, vehicle="4006", instance=0,
                              frame="gz_fire_px4vision", world="default_fire", location="Purdue", vehicle_config=None)
        sim = SimpleNamespace(env={}, world_path=self.root / "world.sdf")
        handles = [Mock(), Mock()]
        with patch.object(px4, "_load_scenario_yaml_px4", return_value=cfg), \
             patch.object(px4, "ensure_px4_built"), \
             patch.object(capture, "_check_ports"), \
             patch.object(px4, "prepare_simulation", return_value=sim), \
             patch.object(px4, "gazebo_command", return_value=["gz"]), \
             patch.object(px4, "spawn_model"), \
             patch.object(px4, "px4_command", return_value=["px4"]), \
             patch.object(px4, "px4_environment", return_value={"PX4_PARAM_MC_ROLL_P": "99", "PX4_SIM_SPEED_FACTOR": "3"}), \
             patch.object(px4, "_popen", side_effect=handles) as launch, \
             patch.object(px4, "_finalize_proc") as finalize, \
             patch.object(px4, "PX4MissionRunner") as px4_mission, \
             patch.object(ardu, "ArduPilotMissionRunner") as ardu_mission, \
             patch.object(capture, "_connect_and_read", side_effect=TimeoutError("unavailable")):
            with self.assertRaisesRegex(TimeoutError, "unavailable"):
                capture._capture_stack("px4", self.scenario, self.root, 1)
        self.assertEqual(finalize.call_count, 2)
        self.assertEqual(finalize.call_args_list[0].args, (handles[1],))
        self.assertNotIn("PX4_PARAM_MC_ROLL_P", launch.call_args.kwargs["env"])
        self.assertNotIn("PX4_SIM_SPEED_FACTOR", launch.call_args.kwargs["env"])
        px4_mission.assert_not_called()
        ardu_mission.assert_not_called()


if __name__ == "__main__":
    unittest.main()
