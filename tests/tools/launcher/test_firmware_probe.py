"""Version identity must follow the executable, not the checkout or a stale cache."""
from collections import deque
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.commander.firmware_identity import read_firmware_identity, verify_firmware_identity
from tools.launcher import firmware_probe as probe
from tests.tools.launcher.test_controller_gain_application import Connection, message


class FirmwareProbeTests(unittest.TestCase):
    def test_mavlink_version_filters_source_and_requests_without_parameter_writes(self):
        for ap in ('ardupilot', 'px4'):
            connection = Connection(ap, {})
            raw = b'c068bfcd' if ap == 'ardupilot' else bytes.fromhex('00000069629d4ac0')
            release = 0x040700ff if ap == 'ardupilot' else 0x011002ff
            connection.mav.command_long_send = Mock(side_effect=lambda *args: connection.queue.extend([
                message('AUTOPILOT_VERSION', system=42, flight_sw_version=0, flight_custom_version=bytes(8)),
                message('AUTOPILOT_VERSION', flight_sw_version=release, flight_custom_version=raw)]))
            identity = read_firmware_identity(connection, ap, timeout=.1)
            self.assertEqual(identity['version'], '4.7.0' if ap == 'ardupilot' else '1.16.2')
            self.assertEqual(identity['git_hash'], 'c068bfcd' if ap == 'ardupilot' else 'c04a9d6269')
            self.assertEqual(connection.writes, [])
            self.assertEqual(connection.mav.command_long_send.call_args.args[4], 148)

    def test_runtime_mismatch_and_armed_vehicle_fail(self):
        expected = dict(autopilot='ardupilot', version='4.7.0', version_type=255, git_hash='c068bfcd')
        for changed in (dict(version='4.6.2'), dict(git_hash='deadbeef')):
            with patch('tools.commander.firmware_identity.read_firmware_identity', return_value=dict(expected, **changed)):
                with self.assertRaises(RuntimeError):
                    verify_firmware_identity(None, expected)
        connection = Connection('ardupilot', {})
        connection.heartbeat.base_mode = 128
        connection.mav.command_long_send = Mock()
        with self.assertRaisesRegex(RuntimeError, 'armed'):
            read_firmware_identity(connection, 'ardupilot', timeout=.1)
        connection.mav.command_long_send.assert_not_called()

    def test_cache_is_keyed_by_binary_hash_and_unknown_version_needs_explicit_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'build/sitl/bin/arducopter'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'old binary')
            identities = [dict(autopilot='ardupilot', version='4.6.2', git_hash='aaaaaaa1'),
                          dict(autopilot='ardupilot', version='4.7.0', git_hash='bbbbbbb2')]
            with patch.object(probe, '_ardu_probe', side_effect=identities) as identify:
                old = probe.identify_firmware(root, 'ardupilot', cache_dir=root/'cache')
                self.assertEqual(old, probe.identify_firmware(root, 'ardupilot', cache_dir=root/'cache'))
                self.assertEqual(identify.call_count, 1)
                binary.write_bytes(b'new binary')
                new = probe.identify_firmware(root, 'ardupilot', cache_dir=root/'cache')
                self.assertEqual(new['parameter_schema'], 'arducopter-4.7')
                self.assertEqual(identify.call_count, 2)
            px4 = root / 'build/px4_sitl_default/bin/px4'
            px4.parent.mkdir(parents=True)
            px4.write_bytes(b'px4')
            with patch.object(probe, '_px4_probe', return_value=dict(autopilot='px4', version='0.0.0', git_hash='abcdef01')):
                with self.assertRaisesRegex(ValueError, 'explicit parameter_schema'):
                    probe.identify_firmware(root, 'px4', cache_dir=root/'cache')
                result = probe.identify_firmware(root, 'px4', parameter_schema='px4-mc-v1', cache_dir=root/'cache')
                self.assertEqual(result['version'], '0.0.0')
                self.assertEqual(result['schema_selection'], 'explicit')

    def test_binary_changed_during_probe_is_not_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'build/sitl/bin/arducopter'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'before')
            def changed(*args):
                binary.write_bytes(b'after')
                return dict(autopilot='ardupilot', version='4.7.0', git_hash='abcdef01')
            with patch.object(probe, '_ardu_probe', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'changed during probe'):
                    probe.identify_firmware(root, 'ardupilot', cache_dir=root/'cache')
            self.assertEqual(list((root/'cache').glob('*.json')), [])


if __name__ == '__main__':
    unittest.main()
