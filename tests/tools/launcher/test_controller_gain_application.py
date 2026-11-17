"""Boot parameter readback failures and commander ARM gating without MAVLink writes."""

from collections import deque
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml
from pymavlink import mavutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.commander import mavlink_parameters as params
from tools.commander.fc_parameter_profiles import GAIN_GROUPS, parse_controller_parameters
from tools.commander import pymavlink_ardupilot_commander as ardu
from tools.commander import pymavlink_px4_commander as px4
from tools.launcher.attempt_logs import AttemptLogs
from tools.scenario.create_params_sets import create_params_sets, resolve_params_sets


def message(kind, *, system=1, component=1, **fields):
    return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: system,
                           get_srcComponent=lambda: component, **fields)


class Connection:
    target_system = 1
    target_component = 1

    def __init__(self, autopilot, baseline):
        self.values = dict(baseline)
        self.values.update({f"MC_{axis}RATE_K": 1.0 for axis in ("ROLL", "PITCH", "YAW")})
        self.values["MC_YAW_WEIGHT"] = .4
        self.reads = []
        self.queue = deque()
        ap = mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA if autopilot == "ardupilot" else mavutil.mavlink.MAV_AUTOPILOT_PX4
        self.heartbeat = message("HEARTBEAT", base_mode=0, autopilot=ap, custom_mode=4)
        self.queue.append(self.heartbeat)
        self.mav = SimpleNamespace(param_request_read_send=self.read, param_set_send=self.write)
        self.writes = []
        self.drop_ack = 0
        self.refuse = None
        self.read_override = None
        self.param_type = mavutil.mavlink.MAV_PARAM_TYPE_REAL32

    def response(self, name, value):
        return message("PARAM_VALUE", param_id=name.encode() + b"\x00", param_value=value,
                       param_type=self.param_type)

    def read(self, system, component, name, index):
        name = name.decode()
        self.reads.append(name)
        if name in self.values:
            value = self.values[name]
            if self.read_override == name:
                value += .0001
            self.queue.append(self.response(name, value))

    def write(self, system, component, name, value, kind):
        name = name.decode()
        self.writes.append((name, value))
        if name != self.refuse:
            self.values[name] = struct.unpack("<f", struct.pack("<f", value))[0]
        if self.drop_ack:
            self.drop_ack -= 1
        else:
            self.queue.append(self.response(name, self.values[name]))

    def recv_match(self, **kwargs):
        return self.queue.popleft() if self.queue else None


class ControllerGainApplication(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.configs = resolve_params_sets(create_params_sets(), seed=17, run_id=0)
        client_class = params.ParameterClient
        patcher = patch.object(params, "ParameterClient", side_effect=lambda *a, **kw: client_class(*a, timeout=.005, attempts=2, **kw))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_both_stacks_verify_boot_values_without_writes(self):
        for ap, config in self.configs.items():
            with self.subTest(ap=ap):
                m = Connection(ap, config["values"])
                path = self.root / ap / "controller_gains.json"
                for _ in range(2):
                    m.queue.append(m.heartbeat)
                    params.verify_controller_parameters(m, config, ap, path)
                report = json.loads(path.read_text())
                self.assertEqual(report["status"], "verified")
                self.assertEqual(report["method"], "startup_file")
                self.assertEqual(set(report["verified"]), set(config["values"]))
                self.assertEqual(m.writes, [])
                self.assertEqual(len(report["context"]), 4 if ap == "px4" else 0)

    def test_position_only_never_reads_or_writes_other_controllers(self):
        for ap, config in self.configs.items():
            names = GAIN_GROUPS[ap]["position"]
            position = parse_controller_parameters(dict(groups=["position"],
                baseline=dict.fromkeys(names, 1.0), values={names[0]: 1.1}), ap)
            m = Connection(ap, {**config["baseline"], **position["values"]})
            other_values = {name: value for name, value in config["baseline"].items()
                            if name not in names}
            # In particular, position-only PX4 does not require the rate K context.
            for name in list(m.values):
                if name.endswith("RATE_K"):
                    del m.values[name]
            path = self.root / f"{ap}_position.json"
            params.verify_controller_parameters(m, position, ap, path)
            self.assertEqual(set(m.reads), set(names))
            self.assertEqual(m.writes, [])
            self.assertEqual(json.loads(path.read_text())["context"], {})
            self.assertEqual({name: m.values[name] for name in other_values}, other_values)

    def test_source_name_type_and_small_gain_validation(self):
        m = Connection("ardupilot", self.configs["ardupilot"]["baseline"])
        name = "ATC_RAT_RLL_D"
        foreign = m.response(name, 999)
        foreign.get_srcSystem = lambda: 2
        other_component = m.response(name, 999)
        other_component.get_srcComponent = lambda: 99
        m.queue.extend([foreign, other_component, m.response("WRONG_NAME", 999),
                        message("STATUSTEXT", text="EKF ready")])
        observed = []
        client = params.ParameterClient(m, on_message=observed.append)
        self.assertAlmostEqual(client.get_param(name), .0036)
        self.assertTrue(any(msg.get_type() == "STATUSTEXT" for msg in observed))
        self.assertFalse(params.values_match(.0016, .0015))
        self.assertFalse(params.values_match(float("nan"), .0015))
        m.param_type = mavutil.mavlink.MAV_PARAM_TYPE_INT32
        with self.assertRaisesRegex(ValueError, "REAL32"):
            client.get_param(name)

    def test_missing_or_armed_vehicle_fails_before_any_write(self):
        config = self.configs["ardupilot"]
        for kind in ("missing", "armed", "wrong_autopilot"):
            with self.subTest(kind=kind):
                m = Connection("ardupilot", config["values"])
                if kind == "missing":
                    del m.values["ATC_RAT_YAW_D"]
                elif kind == "armed":
                    m.heartbeat.base_mode = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                else:
                    m.heartbeat.autopilot = mavutil.mavlink.MAV_AUTOPILOT_PX4
                path = self.root / f"{kind}.json"
                with self.assertRaises((TimeoutError, RuntimeError)):
                    params.verify_controller_parameters(m, config, "ardupilot", path)
                self.assertEqual(m.writes, [])
                report = json.loads(path.read_text())
                self.assertEqual(report["status"], "failed")
                self.assertIn("error", report)

    def test_stale_boot_value_fails_without_attempting_a_write(self):
        config = copy.deepcopy(self.configs["px4"])
        config["values"]["MC_ROLLRATE_D"] = .0016
        m = Connection("px4", config["values"])
        m.values["MC_ROLLRATE_D"] = .0015
        path = self.root / "stale.json"
        with self.assertRaisesRegex(RuntimeError, "MC_ROLLRATE_D"):
            params.verify_controller_parameters(m, config, "px4", path)
        report = json.loads(path.read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["verified"]["MC_ROLLRATE_D"], .0015)
        self.assertEqual(m.writes, [])

    def test_stop_request_interrupts_parameter_configuration(self):
        config = self.configs["px4"]
        m = Connection("px4", config["baseline"])
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            params.verify_controller_parameters(m, config, "px4", self.root / "cancelled.json",
                                   check_stop=Mock(side_effect=RuntimeError("cancelled")))
        self.assertFalse(m.writes)

    def test_commanders_gate_readiness_and_arm_on_gain_verification(self):
        for module, runner_class, ap, logdir in (
            (ardu, ardu.ArduPilotMissionRunner, "ardupilot", "ardu_logs"),
            (px4, px4.PX4MissionRunner, "px4", "px4_logs"),
        ):
            for enabled, failure in ((True, False), (True, True), (False, False)):
                with self.subTest(ap=ap, enabled=enabled, failure=failure):
                    data = {"autopilots": {ap: {}}}
                    if enabled:
                        data["autopilots"][ap]["controller_gains"] = self.configs[ap]
                    scenario = self.root / ap / f"{enabled}_{failure}" / "scenario.yaml"
                    scenario.parent.mkdir(parents=True)
                    scenario.write_text(yaml.safe_dump(data))
                    runner = runner_class(scenario)
                    runner._wait_heartbeat = Mock()
                    runner._wait_ready_to_arm = Mock(side_effect=RuntimeError("test stopped after parameter gate"))
                    runner._arm = Mock()
                    m = Connection(ap, self.configs[ap]["values"])
                    m.queue.append(m.heartbeat)
                    if failure:
                        del m.values[next(iter(self.configs[ap]["baseline"]))]
                    with patch.object(module, "_build_items_from_scenario", return_value=([], 3, True, True)), \
                         patch.object(module.mavutil, "mavlink_connection", return_value=m):
                        runner._run()
                    self.assertEqual(m.writes, [])
                    runner._arm.assert_not_called()
                    self.assertEqual(runner._wait_ready_to_arm.call_count, 0 if failure else 1)
                    report_path = scenario.parent / logdir / "controller_gains.json"
                    if enabled:
                        report = json.loads(report_path.read_text())
                        self.assertEqual(report["status"], "failed" if failure else "verified")
                    else:
                        self.assertFalse(m.writes)
                        self.assertFalse(report_path.exists())

    def test_gain_report_is_archived_and_cleared_between_attempts(self):
        logs = self.root / "run_000/px4_logs"
        logs.mkdir(parents=True)
        (logs.parent / "scenario.yaml").write_text("{}")
        report = logs / "controller_gains.json"
        report.write_text('{"status":"failed"}')
        attempts = AttemptLogs(logs)
        result = {"attempt": 0, "status": "failed"}
        attempts.archive(result)
        self.assertFalse(report.exists())
        archived = logs.parent / result["archived_to"] / report.name
        self.assertEqual(json.loads(archived.read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
