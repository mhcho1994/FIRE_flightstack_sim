"""Profile generation and native parameter files used by the boot launchers."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.commander.fc_parameter_profiles import normalize_profiles, resolve_profiles
from tools.scenario.create_params_sets import (
    create_params_sets, parameter_file_bytes, resolve_params_sets, save_params_sets, write_parameter_files,
)
from tools.launcher.startup_parameters import (
    prepare_boot_parameters, prepare_px4_startup_script, validate_parameter_file,
)
from tools.launcher.run_ardupilot_gz_sitl import _run_sitl_cmd
from tools.launcher.run_px4_gz_sitl import px4_command
from tools.launcher.attempt_logs import AttemptLogs


class ParameterSetsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="params with spaces ")
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

    def scenario(self, configs):
        path = self.root / "scenario.yaml"
        path.write_text(yaml.safe_dump({"autopilots": {ap: {"controller_gains": gains}
                                                      for ap, gains in configs.items()}}))
        return path

    def test_profile_callable_and_yaml_have_identical_samples(self):
        rules = {"px4": {"MC_ROLLRATE_P": dict(operation="scale", uniform=[.9, 1.1]),
                         "MC_ROLLRATE_I": dict(operation="offset", value=.01),
                         "MC_ROLLRATE_D": dict(operation="set", value=.002)}}
        profile = create_params_sets(autopilots=["px4"], parameters=rules)
        saved = save_params_sets(profile, self.root / "profile.yaml")
        replay = create_params_sets(saved)
        self.assertEqual(profile, replay)
        first = resolve_params_sets(profile, seed=42, run_id=3)
        second = resolve_params_sets(replay, seed=42, run_id=3)
        self.assertEqual(first, second)
        self.assertEqual(first["px4"]["groups"], ["rate", "attitude", "position", "velocity"])
        self.assertEqual(len(first["px4"]["values"]), 20)
        self.assertEqual(first["px4"]["values"]["MC_ROLLRATE_D"], .002)
        self.assertAlmostEqual(first["px4"]["values"]["MC_ROLLRATE_I"], .21)
        self.assertAlmostEqual(first["px4"]["values"]["MC_PITCHRATE_P"], .15)
        self.assertEqual(rules["px4"]["MC_ROLLRATE_P"]["uniform"], [.9, 1.1])

    def test_snapshot_paths_are_frozen_before_saving_elsewhere(self):
        raw = create_params_sets(autopilots=["ardupilot"], groups=["position"], parameters={"ardupilot": {}})
        baseline = raw["ardupilot"]["baseline"]
        (self.root / "baseline.parm").write_text("\n".join(f"{k} {v}" for k, v in baseline.items()))
        raw["ardupilot"]["baseline"] = "baseline.parm"
        source = self.root / "source.yaml"
        source.write_text(yaml.safe_dump(raw))
        frozen = create_params_sets(source)
        saved = save_params_sets(frozen, self.root / "elsewhere/profile.yaml")
        (self.root / "baseline.parm").unlink()
        self.assertEqual(create_params_sets(saved)["ardupilot"]["baseline"], baseline)

    def test_native_files_preserve_float32_values_and_relative_hashes(self):
        resolved = resolve_params_sets(create_params_sets(), seed=42, run_id=0, run_dir=self.root)
        for ap, config in resolved.items():
            path = self.root / config["parameter_file"]
            content = path.read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), config["parameter_file_sha256"])
            self.assertFalse(Path(config["parameter_file"]).is_absolute())
            if ap == "ardupilot":
                actual = dict(line.split() for line in content.decode().splitlines() if not line.startswith("#"))
                actual = {name: float(value) for name, value in actual.items()}
            else:
                self.assertEqual(struct.unpack("<i", content[:4])[0], len(content))
                self.assertEqual(content[-1], 0)
                actual, cursor = {}, 4
                while cursor < len(content) - 1:
                    self.assertEqual(content[cursor], 1)  # BSON double, as PX4 exports FLOAT parameters.
                    end = content.index(b"\x00", cursor + 1)
                    name = content[cursor + 1:end].decode()
                    actual[name] = struct.unpack("<d", content[end + 1:end + 9])[0]
                    cursor = end + 9
            self.assertEqual(set(actual), set(config["values"]))
            for name, value in config["values"].items():
                self.assertEqual(struct.pack("<f", actual[name]), struct.pack("<f", value))

    def test_missing_tampered_and_stale_files_are_rejected(self):
        configs = resolve_params_sets(create_params_sets(), seed=1, run_id=0, run_dir=self.root)
        for ap, config in configs.items():
            path = self.root / config["parameter_file"]
            original = path.read_bytes()
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                validate_parameter_file(config, ap, self.root)
            path.write_bytes(original + b"bad")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_parameter_file(config, ap, self.root)
            path.write_bytes(original)
            stale = copy.deepcopy(config)
            stale["values"][next(iter(stale["values"]))] += .01
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_parameter_file(stale, ap, self.root)

    def test_legacy_inline_values_materialize_without_resampling(self):
        resolved = resolve_params_sets(create_params_sets(), seed=10, run_id=3)
        scenario = self.scenario(resolved)
        for ap, config in resolved.items():
            logs = self.root / ("ardu_logs" if ap == "ardupilot" else "px4_logs")
            first = prepare_boot_parameters(scenario, ap, logs)
            original = first.read_bytes()
            first.write_bytes(b"old attempt")
            second = prepare_boot_parameters(scenario, ap, logs)
            self.assertEqual(second.read_bytes(), original)
            self.assertEqual(original, parameter_file_bytes(config, ap))
            manifest = json.loads((logs / "boot_parameters.json").read_text())
            self.assertEqual(manifest["method"], "startup_file")
        self.assertEqual(prepare_boot_parameters(self.scenario({}), "px4", self.root), None)

    def test_ardupilot_adds_defaults_and_wipes_only_configured_runs(self):
        args = ("ArduCopter", "gazebo-px4vision", "JSON", 0, 14551, "udp:127.0.0.1:14550", "Purdue")
        plain = _run_sitl_cmd(*args)
        self.assertNotIn("--wipe-eeprom", plain)
        self.assertNotIn("--add-param-file", plain)
        param = self.root / "boot.parm"
        command = _run_sitl_cmd(*args, parameter_file=param)
        self.assertIn("--wipe-eeprom", command)
        self.assertEqual(command[command.index("--add-param-file") + 1], str(param))

    def test_px4_import_order_and_failure_stop_startup(self):
        logs = self.root / "px4_logs"
        work = logs / "rootfs"
        work.mkdir(parents=True)
        airframe = self.root / "airframe"
        airframe.write_text("echo airframe\n")
        original = self.root / "rcS"
        # Mirror PX4's source/marker boundaries, using local shell stubs.
        original.write_text(f'''#!/bin/sh
param() {{ echo "import $*"; return "${{IMPORT_FAIL:-0}}"; }}
dataman() {{ echo dataman; }}
autostart_file='{airframe}'
. "$autostart_file"

dataman start
echo controllers
''')
        native = logs / "boot_parameters.bson"
        native.write_bytes(b"file")
        startup = prepare_px4_startup_script(original, work, native)
        result = subprocess.run(["sh", startup], cwd=work, text=True, capture_output=True, check=True)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], "airframe")
        self.assertIn(str(native), lines[1])
        self.assertEqual(lines[2:], ["dataman", "controllers"])
        import os
        result = subprocess.run(["sh", startup], cwd=work, env=dict(os.environ, IMPORT_FAIL="1"), text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("controllers", result.stdout)
        self.assertNotIn("FIRE", original.read_text())
        original.write_text("dataman start\n")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            prepare_px4_startup_script(original, work, native)

    def test_px4_command_passes_the_per_attempt_startup_script(self):
        px4 = self.root / "px4"
        build = px4 / "build/px4_sitl_default"
        for file in ("bin/px4", "etc/init.d-posix/airframes/4006_model", "etc/init.d-posix/rcS"):
            path = build / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('. "$autostart_file"\n\ndataman start\n')
        work = self.root / "px4_logs/rootfs"
        native = self.root / "boot.bson"
        command = px4_command(px4, 0, work, "4006", parameter_file=native)
        self.assertEqual(command[command.index("-s") + 1], "../boot_parameters.rcS")
        self.assertTrue((work.parent / "boot_parameters.rcS").is_file())
        plain = px4_command(px4, 0, work, "4006")
        self.assertEqual(plain[plain.index("-s") + 1], "etc/init.d-posix/rcS")

    def test_archive_contains_replayable_parameter_files_and_attempt_manifest(self):
        configs = resolve_params_sets(create_params_sets(), seed=2, run_id=4, run_dir=self.root)
        scenario = self.scenario(configs)
        logs = self.root / "px4_logs"
        native = prepare_boot_parameters(scenario, "px4", logs)
        result = dict(attempt=0, status="failed")
        AttemptLogs(logs).archive(result)
        archived = self.root / result["archived_to"]
        self.assertTrue((archived / native.name).is_file())
        self.assertTrue((archived / "boot_parameters.json").is_file())
        self.assertTrue((archived.parent / configs["px4"]["parameter_file"]).is_file())
        self.assertFalse(native.exists())

    def test_cli_generates_reusable_yaml_and_native_files(self):
        output = self.root / "profile.yaml"
        result = subprocess.run([
            sys.executable, "-B", "-m", "tools.scenario.create_params_sets",
            "--output", str(output), "--params-dir", str(self.root), "--seed", "42",
            "--autopilots", "px4", "--baseline-only",
            "--random-scale", "MC_ROLLRATE_P", ".9", "1.1",
            "--offset", "MC_ROLLRATE_I", ".01", "--set", "MC_ROLLRATE_D", ".002",
        ], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        profile = yaml.safe_load(output.read_text())
        self.assertEqual(len(profile["px4"]["parameters"]), 3)
        self.assertEqual(profile["px4"]["groups"], ["rate", "attitude", "position", "velocity"])
        self.assertEqual(len(profile["px4"]["baseline"]), 20)
        self.assertTrue((self.root / "parameters/px4.bson").is_file())
        self.assertTrue((self.root / "controller_gains.resolved.yaml").is_file())

    def test_invalid_profile_cannot_create_native_files(self):
        raw = create_params_sets()
        raw["px4"]["parameters"]["MC_ROLLRATE_P"] = dict(operation="set", value=-1)
        with self.assertRaises(ValueError):
            resolve_params_sets(raw, seed=0, run_id=0, run_dir=self.root)
        self.assertFalse((self.root / "parameters").exists())
        with self.assertRaises(ValueError):
            create_params_sets(groups=["acceleration"])

    def test_generator_writes_native_files_and_default_profile_flag(self):
        output = self.root / "dataset"
        result = subprocess.run([
            sys.executable, "-B", "-m", "tools.scenario.scenario_generator", "planar_n_pts",
            "--controller-gains", "--outdir", str(output), "--seed", "42", "--runs", "1",
        ], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run = output / "run_000"
        scenario = yaml.safe_load((run / "scenario.yaml").read_text())
        expected = resolve_params_sets(create_params_sets(), seed=42, run_id=0)
        for ap in expected:
            config = scenario["autopilots"][ap]["controller_gains"]
            self.assertEqual(config["values"], expected[ap]["values"])
            self.assertEqual(validate_parameter_file(config, ap, run), (run / config["parameter_file"]).read_bytes())
