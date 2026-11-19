"""Native preservation, version conversion, boot/readback agreement and executable binding."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from tools.commander.fc_parameter_registry import parameter_names, schema_for_firmware
from tools.launcher.parameter_bridge import bridge_gains, bridge_flight
from tools.launcher.startup_parameters import prepare_boot_parameters, read_boot_report
from tools.launcher.attempt_logs import capture_run_inputs
from tools.launcher.firmware_probe import binary_sha256
from tools.scenario.create_params_sets import create_params_sets, resolve_params_sets, save_params_sets
from tools.commander.fc_parameter_profiles import load_profiles
from tools.commander import pymavlink_ardupilot_commander as ardu
from tools.commander import pymavlink_px4_commander as px4
from tests.tools.launcher.test_controller_gain_application import Connection, with_mission


def profile47():
    names = parameter_names('ardupilot', ['position', 'velocity', 'acceleration'], 'arducopter-4.7')
    return {'ardupilot': dict(parameter_schema='arducopter-4.7',
        firmware=dict(autopilot='ardupilot', version='4.7.0', git_hash='1511f271'),
        groups=['position', 'velocity', 'acceleration'],
        baseline={name: .05 if name == 'PSC_D_ACC_P' else .1 for name in names},
        parameters={'PSC_D_ACC_P': dict(operation='scale', value=2)})}


class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.binary = self.root / 'firmware'
        self.binary.write_bytes(b'version one')

    def target(self, ap='ardupilot', schema='arducopter-4.7'):
        return dict(autopilot=ap, parameter_schema=schema,
                    binary=str(self.binary), binary_sha256=binary_sha256(self.binary))

    def test_source_schema_survives_selection_save_sampling_and_parameter_export(self):
        profiles = create_params_sets(profile47(), groups=['acceleration'])
        save_params_sets(profiles, self.root / 'gains.yaml')
        loaded = load_profiles(self.root / 'gains.yaml')['ardupilot']
        self.assertEqual(loaded['parameter_schema'], 'arducopter-4.7')
        self.assertEqual(loaded['firmware']['git_hash'], '1511f271')
        config = resolve_params_sets(profiles, seed=3, run_id=7, run_dir=self.root)['ardupilot']
        self.assertEqual(config['baseline']['PSC_D_ACC_P'], .05)
        self.assertEqual(config['values']['PSC_D_ACC_P'], .1)
        self.assertNotIn('PSC_ACCZ_P', config['values'])
        self.assertIn('PSC_D_ACC_P', (self.root / config['parameter_file']).read_text())

    def test_same_version_and_both_conversion_directions_do_not_mutate_or_resample(self):
        source = resolve_params_sets(profile47(), seed=3, run_id=7)['ardupilot']
        original = copy.deepcopy(source)
        same, changes = bridge_gains(source, 'ardupilot', self.target())
        self.assertEqual(same['values'], source['values'])
        self.assertEqual(changes, [])
        older, changes = bridge_gains(source, 'ardupilot', self.target(schema='arducopter-4.6'))
        self.assertEqual(older['values']['PSC_ACCZ_P'], 1.0)
        self.assertEqual(older['baseline']['PSC_ACCZ_P'], .5)
        self.assertEqual(older['values']['PSC_POSXY_P'], source['values']['PSC_NE_POS_P'])
        self.assertEqual(len(changes), 11)
        newer, _ = bridge_gains(older, 'ardupilot', self.target())
        for name, value in source['values'].items():
            self.assertAlmostEqual(newer['values'][name], value)
        self.assertEqual(source, original)
        self.assertEqual(bridge_gains(source, 'ardupilot', self.target(schema='arducopter-4.6'))[0], older)

    def test_legacy_converted_profile_uses_stored_units_not_logged_firmware(self):
        config = resolve_params_sets(create_params_sets(autopilots=['ardupilot']), seed=0, run_id=0)['ardupilot']
        config['firmware'] = dict(autopilot='ardupilot', version='4.7.0')
        effective, _ = bridge_gains(config, 'ardupilot', self.target())
        self.assertAlmostEqual(effective['values']['PSC_D_ACC_P'], config['values']['PSC_ACCZ_P'] / 10)

    def test_speed_conversion_and_unknown_schema_failures(self):
        effective, changes = bridge_flight(dict(values={'WPNAV_SPEED': 150}), 'ardupilot', self.target())
        self.assertEqual(effective['values'], {'WP_SPD': 1.5})
        self.assertEqual(changes[0]['multiplier'], .01)
        reverse, _ = bridge_flight(effective, 'ardupilot', self.target(schema='arducopter-4.6'))
        self.assertEqual(reverse['values'], {'WPNAV_SPEED': 150})
        for version in ('0.0.0', '4.8.0', None):
            with self.assertRaisesRegex(ValueError, 'Cannot select'):
                schema_for_firmware(dict(autopilot='ardupilot', version=version))
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            schema_for_firmware(dict(autopilot='ardupilot', version='4.7.0'), 'arducopter-4.6')
        with self.assertRaises(ValueError):
            bridge_gains(resolve_params_sets(profile47(), seed=0, run_id=0)['ardupilot'],
                         'ardupilot', self.target(schema='px4-mc-v1'))

    def test_boot_manifest_and_fingerprint_bind_effective_values_and_binary(self):
        source = resolve_params_sets(profile47(), seed=0, run_id=0, run_dir=self.root)['ardupilot']
        scenario = self.root / 'scenario.yaml'
        scenario.write_text(yaml.safe_dump(dict(autopilots=dict(ardupilot=dict(controller_gains=source)))))
        original = scenario.read_bytes()
        target = self.target(schema='arducopter-4.6')
        config = dict(target_firmware=target)
        inputs = capture_run_inputs(self.root, 'ardupilot', config)
        boot = prepare_boot_parameters(scenario, 'ardupilot', self.root / 'ardu_logs', target_firmware=target)
        report = read_boot_report(boot, scenario)
        self.assertEqual(report['controller_gains']['values']['PSC_ACCZ_P'], 1)
        self.assertEqual(report['sha256'], inputs['parameters_sha256'])
        self.assertEqual(report['values'], inputs['parameter_bridge']['values'])
        self.assertEqual(scenario.read_bytes(), original)
        self.binary.write_bytes(b'version two')
        with self.assertRaisesRegex(ValueError, 'Firmware binary changed'):
            read_boot_report(boot, scenario)
        with self.assertRaisesRegex(ValueError, 'Firmware binary changed'):
            capture_run_inputs(self.root, 'ardupilot', config)
        config['target_firmware'] = self.target(schema='arducopter-4.6')
        self.assertNotEqual(inputs['sha256'], capture_run_inputs(self.root, 'ardupilot', config)['sha256'])

    def test_both_commanders_use_effective_values_and_block_on_firmware_mismatch(self):
        for module, cls, ap, schema in ((ardu, ardu.ArduPilotMissionRunner, 'ardupilot', 'arducopter-4.6'),
                                       (px4, px4.PX4MissionRunner, 'px4', 'px4-mc-v1')):
            profiles = profile47() if ap == 'ardupilot' else create_params_sets(autopilots=['px4'])
            source = resolve_params_sets(profiles, seed=0, run_id=0)[ap]
            scenario = self.root / 'scenario.yaml'
            scenario.write_text(yaml.safe_dump(with_mission(dict(autopilots={ap: dict(controller_gains=source)}))))
            logs = self.root / ('ardu_logs' if ap == 'ardupilot' else 'px4_logs')
            boot = prepare_boot_parameters(scenario, ap, logs, target_firmware=self.target(ap, schema))
            report = read_boot_report(boot, scenario)
            for mismatch in (False, True):
                with self.subTest(ap=ap, mismatch=mismatch):
                    runner = cls(scenario, boot_parameters=report)
                    runner._wait_heartbeat = Mock()
                    runner._wait_ready_to_arm = Mock(side_effect=RuntimeError('stop at readiness'))
                    runner._arm = Mock()
                    connection = Connection(ap, report['values'])
                    connection.queue.append(connection.heartbeat)
                    with patch.object(module.mavutil, 'mavlink_connection', return_value=connection), \
                         patch('tools.commander.mission_session_runner.verify_firmware_identity', side_effect=RuntimeError('wrong firmware') if mismatch else None) as verify:
                        runner._run()
                    verify.assert_called_once()
                    runner._arm.assert_not_called()
                    self.assertEqual(runner._wait_ready_to_arm.call_count, 0 if mismatch else 1)
                    self.assertEqual(connection.writes, [])
                    if not mismatch:
                        actual = json.loads((logs / 'controller_gains.json').read_text())
                        self.assertEqual(set(actual['verified']), set(report['values']))


if __name__ == '__main__':
    unittest.main()
