"""Flight default boot files, readback gates, fingerprints and archival."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from tools.commander.flight_parameters import (
    parse_flight_parameters, speed_defaults, write_flight_parameter_file,
)
from tools.commander import pymavlink_ardupilot_commander as ardu
from tools.commander import pymavlink_px4_commander as px4
from tools.launcher.attempt_logs import AttemptLogs, capture_run_inputs
from tools.launcher.startup_parameters import collect_boot_parameters, prepare_boot_parameters
from tests.tools.launcher.test_controller_gain_application import Connection, with_mission, target_for


class FlightParametersTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        from tools.launcher.firmware_probe import binary_sha256
        binary = self.root / 'firmware'
        binary.write_bytes(b'test firmware')
        mock = patch('tools.launcher.startup_parameters.identify_scenario_firmware',
            side_effect=lambda data, ap, config=None: dict(autopilot=ap,
                parameter_schema='arducopter-4.6' if ap == 'ardupilot' else 'px4-mc-v1',
                binary=str(binary), binary_sha256=binary_sha256(binary)))
        mock.start()
        self.addCleanup(mock.stop)

    def scenario(self, ap):
        run = self.root / ap
        run.mkdir()
        config = write_flight_parameter_file(speed_defaults(1, ap, source="plan hoverSpeed"), ap, run)
        data = with_mission({"autopilots": {ap: {"flight_parameters": config}}})
        path = run / "scenario.yaml"
        path.write_text(yaml.safe_dump(data))
        return path, config, data

    def test_flight_defaults_boot_payload_and_tampering(self):
        for ap, logdir in (("ardupilot", "ardu_logs"), ("px4", "px4_logs")):
            with self.subTest(ap=ap):
                scenario, config, data = self.scenario(ap)
                inputs = capture_run_inputs(scenario.parent, ap)
                boot = prepare_boot_parameters(scenario, ap, scenario.parent / logdir)
                self.assertEqual(boot.read_bytes(), collect_boot_parameters(data, ap, scenario.parent)[0])
                report = json.loads(boot.with_suffix('.json').read_text())
                self.assertEqual(report['values'], config['values'])
                changed = copy.deepcopy(data)
                changed['autopilots'][ap]['flight_parameters'] = speed_defaults(2, ap, source='override')
                scenario.write_text(yaml.safe_dump(changed))
                self.assertNotEqual(inputs['sha256'], capture_run_inputs(scenario.parent, ap)['sha256'])
                scenario.write_text(yaml.safe_dump(data))
                (scenario.parent / config['parameter_file']).write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    capture_run_inputs(scenario.parent, ap)
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    prepare_boot_parameters(scenario, ap, scenario.parent / logdir)

    def test_invalid_values_types_and_names(self):
        for ap, name in (("ardupilot", "WPNAV_SPEED"), ("px4", "MPC_XY_CRUISE")):
            for value in (0, -1, True, float('nan'), float('inf'), '1'):
                with self.subTest(ap=ap, value=value), self.assertRaises(ValueError):
                    parse_flight_parameters(dict(values={name: value}), ap)
        with self.assertRaises(ValueError):
            parse_flight_parameters(dict(values={'ARMING_CHECK': 0}), 'ardupilot')

    def test_speed_readback_failure_blocks_both_commanders_before_arm(self):
        for module, cls, ap, logdir in ((ardu, ardu.ArduPilotMissionRunner, 'ardupilot', 'ardu_logs'),
                                       (px4, px4.PX4MissionRunner, 'px4', 'px4_logs')):
            scenario, config, _ = self.scenario(ap)
            for failure in (False, True):
                with self.subTest(ap=ap, failure=failure):
                    runner = cls(scenario, target_firmware=target_for(ap))
                    runner._wait_heartbeat = Mock()
                    runner._wait_ready_to_arm = Mock(side_effect=RuntimeError('test ended at readiness'))
                    runner._arm = Mock()
                    m = Connection(ap, config['values'])
                    m.queue.append(m.heartbeat)
                    if failure:
                        m.values[next(iter(config['values']))] *= 2
                    with patch("tools.commander.mission_session_runner.verify_firmware_identity"), \
                         patch.object(module.mavutil, 'mavlink_connection', return_value=m):
                        runner._run()
                    runner._arm.assert_not_called()
                    self.assertEqual(runner._wait_ready_to_arm.call_count, 0 if failure else 1)
                    self.assertEqual(m.writes, [])
                    report = json.loads((scenario.parent / logdir / 'flight_parameters.json').read_text())
                    self.assertEqual(report['status'], 'failed' if failure else 'verified')
                    self.assertEqual(set(report['verified']), set(config['values']))

    def test_flight_defaults_and_readback_are_archived(self):
        scenario, config, _ = self.scenario('px4')
        (scenario.parent / 'source.plan').write_text('{"fileType":"Plan"}')
        (scenario.parent / 'import_report.json').write_text('{"version":1}')
        logs = scenario.parent / 'px4_logs'
        boot = prepare_boot_parameters(scenario, 'px4', logs)
        report = logs / 'flight_parameters.json'
        report.write_text('{"status":"verified"}')
        firmware_report = logs / 'firmware_identity.json'
        firmware_report.write_text('{"status":"verified"}')
        result = dict(attempt=0, status='failed')
        AttemptLogs(logs).archive(result)
        archived = scenario.parent / result['archived_to']
        self.assertTrue((archived / boot.name).is_file())
        self.assertTrue((archived / report.name).is_file())
        self.assertTrue((archived.parent / config['parameter_file']).is_file())
        self.assertTrue((archived.parent / 'source.plan').is_file())
        self.assertTrue((archived.parent / 'import_report.json').is_file())
        self.assertFalse(report.exists())
        self.assertTrue((archived / firmware_report.name).is_file())
        self.assertFalse(firmware_report.exists())


if __name__ == '__main__':
    unittest.main()
