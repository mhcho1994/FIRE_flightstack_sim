"""Optional gain provenance/environment policies without building or starting SITL."""

import copy
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.commander.fc_parameter_profiles import normalize_profiles
from tools.launcher import gain_environment as env
from tools.launcher.attempt_logs import AttemptLogs, capture_run_inputs, prepare_run_logs
from tools.scenario import scenario_generator as generator
from tools.scenario.create_params_sets import create_params_sets, save_params_sets
from tools.scenario.gain_provenance import load_capture_provenance


class GainEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profiles = create_params_sets()
        self.profile = save_params_sets(self.profiles, self.root / "capture.yaml")
        self.meta = self.profile.with_suffix(".meta.json")
        self.targets = {}
        for ap in self.profiles:
            root = self.root / ap
            vehicle, frame = ("4006", "gz_test") if ap == "px4" else ("ArduCopter", "gazebo-test")
            self.targets[ap] = {f"{ap}_dir": str(root), "vehicle": vehicle, "frame": frame}
            if ap == "px4":
                files = ("build/px4_sitl_default/bin/px4", "build/px4_sitl_default/etc/init.d-posix/rcS",
                         "build/px4_sitl_default/etc/init.d/rc.mc_defaults",
                         "build/px4_sitl_default/etc/init.d-posix/airframes/4006_test")
            else:
                files = ("build/sitl/bin/arducopter", "Tools/autotest/default_params/copter.parm",
                         "Tools/autotest/default_params/test.parm")
                registry = {vehicle: {"frames": {frame: {"default_params_filename": [
                    "default_params/copter.parm", "default_params/test.parm"]}}}}
                self.write(root / "Tools/autotest/pysim/vehicleinfo.py", f"class VehicleInfo:\n    options = {registry!r}\n")
            for name in files:
                self.write(root / name, name)
        self.environment = {ap: env.collect_environment(ap, cfg) for ap, cfg in self.targets.items()}
        entries = {}
        normalized = normalize_profiles(self.profiles)
        for ap, current in self.environment.items():
            root = self.targets[ap][f"{ap}_dir"]
            entries[ap] = dict(baseline_sha256=normalized[ap]["baseline_sha256"],
                               vehicle=current["vehicle"], frame=current["frame"],
                               firmware=dict(directory=root, commit=current["commit"], binary_sha256=current["binary_sha256"],
                                   startup_files={str(Path(root) / name): value for name, value in current["startup_files"].items()}))
        self.metadata = dict(status="complete", profile_sha256=hashlib.sha256(self.profile.read_bytes()).hexdigest(),
                             autopilots=entries)
        self.write_meta()

    def write(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_meta(self):
        self.meta.write_text(json.dumps(self.metadata))

    def generate(self, name, policy=None, profile=None):
        output = self.root / name
        args = ["generator", "planar_n_pts", "--outdir", str(output), "--runs", "1", "--seed", "42",
                "--controller-gains", str(profile or self.profile)]
        for ap, target in self.targets.items():
            for name in ("dir", "vehicle", "frame"):
                args.extend([f"--{ap}-{name}", str(target[f"{ap}_dir" if name == "dir" else name])])
        if policy is not None:
            args.extend(["--gain-environment-policy", policy])
        with patch.object(sys, "argv", args), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            generator.main()
        return output

    def test_default_and_explicit_off_ignore_invalid_metadata_and_collect_nothing(self):
        self.meta.write_text("{not-json")
        with patch.object(env, "collect_environment", side_effect=AssertionError("environment read while off")), \
             patch.object(env, "load_capture_provenance", side_effect=AssertionError("metadata read while off")):
            for name, policy in (("default", None), ("explicit", "off")):
                output = self.generate(name, policy)
                run = output / "run_000"
                data = yaml.safe_load((run / "scenario.yaml").read_text())
                for ap in self.profiles:
                    self.assertNotIn("controller_gain_environment", data["autopilots"][ap])
                    self.assertNotIn("gain_environment", capture_run_inputs(run, ap))
            self.assertEqual(env.inspect_gain_environments(Path("missing"), {}, {}, "off"), {})
        self.assertEqual((self.root / "default/run_000/scenario.yaml").read_bytes(),
                         (self.root / "explicit/run_000/scenario.yaml").read_bytes())

    def test_default_builder_does_not_read_or_write_capture_metadata(self):
        self.meta.write_text("invalid")
        out = self.root / "builder.yaml"
        command = [sys.executable, "-B", "-m", "tools.scenario.create_params_sets", "--profile", str(self.profile),
                   "--output", str(out)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(out.with_suffix(".meta.json").exists())

    def test_record_preserves_gains_and_reports_changed_binary(self):
        plain = self.generate("plain")
        binary = self.root / "px4/build/px4_sitl_default/bin/px4"
        binary.write_text("new binary")
        recorded = self.generate("record", "record")
        a = yaml.safe_load((plain / "run_000/scenario.yaml").read_text())
        b = yaml.safe_load((recorded / "run_000/scenario.yaml").read_text())
        for ap in self.profiles:
            self.assertEqual(a["autopilots"][ap]["controller_gains"], b["autopilots"][ap]["controller_gains"])
        check = b["autopilots"]["px4"]["controller_gain_environment"]["comparison"]
        self.assertEqual(check["status"], "mismatch")
        self.assertTrue(any(item["field"] == "binary_sha256" for item in check["differences"]))

    def test_require_match_accepts_capture_and_rejects_changed_airframe_before_output(self):
        self.generate("matching", "require-match")
        startup = self.root / "px4/build/px4_sitl_default/etc/init.d-posix/airframes/4006_test"
        startup.write_text("changed startup parameters")
        with self.assertRaises(SystemExit):
            self.generate("mismatch", "require-match")
        self.assertFalse((self.root / "mismatch").exists())

    def test_missing_target_build_is_pending_then_must_be_checked_before_launch(self):
        binary = self.root / "px4/build/px4_sitl_default/bin/px4"
        original = binary.read_bytes()
        airframe = self.root / "px4/build/px4_sitl_default/etc/init.d-posix/airframes/4006_test"
        original_airframe = airframe.read_bytes()
        binary.unlink()
        airframe.unlink()
        output = self.generate("pending", "require-match")
        run = output / "run_000"
        data = yaml.safe_load((run / "scenario.yaml").read_text())
        self.assertEqual(data["autopilots"]["px4"]["controller_gain_environment"]["comparison"]["status"], "pending")
        with self.assertRaisesRegex(ValueError, "require-match"):
            capture_run_inputs(run, "px4")
        binary.write_bytes(original)
        airframe.write_bytes(original_airframe)
        self.assertEqual(capture_run_inputs(run, "px4")["gain_environment"]["comparison"]["status"], "match")

    def test_missing_metadata_is_recorded_but_require_match_rejects_it(self):
        self.meta.unlink()
        output = self.generate("unknown", "record")
        data = yaml.safe_load((output / "run_000/scenario.yaml").read_text())
        self.assertEqual(data["autopilots"]["px4"]["controller_gain_environment"]["comparison"]["status"], "unknown")
        with self.assertRaises(SystemExit):
            self.generate("strict", "require-match")

    def test_flight_log_and_legacy_metadata_do_not_claim_a_full_environment_match(self):
        normalized = normalize_profiles(self.profiles)
        self.metadata = dict(schema_version=2, profile_sha256=hashlib.sha256(self.profile.read_bytes()).hexdigest(),
            autopilots={ap: dict(autopilot=ap, baseline_sha256=config["baseline_sha256"], groups=config["groups"],
                                firmware_identity={"autopilot": ap, "version": "1.0.0"}, source_log="flight.log")
                        for ap, config in normalized.items()})
        self.write_meta()
        output = self.generate("flight", "record")
        data = yaml.safe_load((output / "run_000/scenario.yaml").read_text())
        self.assertEqual(data["autopilots"]["px4"]["controller_gain_environment"]["comparison"]["status"], "unknown")
        with self.assertRaises(SystemExit):
            self.generate("flight-strict", "require-match")
        one = {"px4": self.profiles["px4"]}
        self.profile = save_params_sets(one, self.root / "legacy.yaml")
        self.meta = self.profile.with_suffix(".meta.json")
        self.meta.write_text(json.dumps(dict(self.metadata["autopilots"]["px4"],
            profile_sha256=hashlib.sha256(self.profile.read_bytes()).hexdigest())))
        self.assertEqual(load_capture_provenance(self.profile, one)["px4"]["source_kind"], "flight_log")

    def test_enabled_modes_reject_profile_and_baseline_hash_mismatches(self):
        for field in ("profile_sha256", "baseline_sha256"):
            original = copy.deepcopy(self.metadata)
            if field == "profile_sha256":
                self.metadata[field] = "0" * 64
            else:
                self.metadata["autopilots"]["px4"][field] = "0" * 64
            self.write_meta()
            for policy in ("record", "require-match"):
                with self.subTest(field=field, policy=policy), self.assertRaises(SystemExit):
                    self.generate(f"bad-{field}-{policy}", policy)
            self.metadata = original

    def test_derived_profile_keeps_original_provenance_and_selected_baseline(self):
        selected = create_params_sets(self.profile, autopilots=["px4"], groups=["rate"],
                                     parameters={"px4": {"MC_ROLLRATE_P": dict(operation="scale", uniform=[.9, 1.1])}})
        origins = load_capture_provenance(self.profile, selected)
        path = save_params_sets(selected, self.root / "elsewhere/derived.yaml", provenance=origins)
        self.profile.unlink()
        self.meta.unlink()
        self.assertEqual(load_capture_provenance(path, selected), origins)
        output = self.generate("derived", "require-match", path)
        data = yaml.safe_load((output / "run_000/scenario.yaml").read_text())
        gains = data["autopilots"]["px4"]["controller_gains"]
        self.assertEqual(gains["baseline"], selected["px4"]["baseline"])
        self.assertNotEqual(gains["values"]["MC_ROLLRATE_P"], gains["baseline"]["MC_ROLLRATE_P"])
        altered = copy.deepcopy(selected)
        altered["px4"]["baseline"]["MC_ROLLRATE_P"] += .1
        with self.assertRaisesRegex(ValueError, "provenance"):
            save_params_sets(altered, self.root / "bad-derived.yaml", provenance=origins)

    def test_builder_cli_preserves_metadata_for_enabled_generation(self):
        path = self.root / "derived.yaml"
        command = [sys.executable, "-B", "-m", "tools.scenario.create_params_sets", "--profile", str(self.profile),
                   "--output", str(path), "--preserve-gain-provenance", "--baseline-only",
                   "--random-scale", "MC_ROLLRATE_P", ".9", "1.1"]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(path.with_suffix(".meta.json").is_file())
        self.generate("cli-derived", "require-match", path)

    def test_enabled_modes_require_a_gain_profile_and_invalid_choices_fail(self):
        for policy in ("record", "require-match", "typo"):
            args = ["generator", "planar_n_pts", "--outdir", str(self.root / "invalid"),
                    "--gain-environment-policy", policy]
            with patch.object(sys, "argv", args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                generator.main()
        self.assertFalse((self.root / "invalid").exists())

    def test_environment_is_rechecked_and_previous_success_is_not_reused_after_binary_change(self):
        output = self.generate("reuse", "record")
        run = output / "run_000"
        for ap, kind in (("px4", "px4"), ("ardupilot", "ardu")):
            with self.subTest(autopilot=ap):
                logs = run / f"{kind}_logs"
                history = AttemptLogs(logs, inputs=capture_run_inputs(run, ap))
                history.begin(1)
                self.write(logs / ("flight.ulg" if ap == "px4" else "logs/00000001.BIN"), "successful flight")
                self.assertTrue(history.finish(0, True))
                self.assertTrue(prepare_run_logs(logs, force=False, inputs=capture_run_inputs(run, ap)))
                binary = (self.root / "px4/build/px4_sitl_default/bin/px4" if ap == "px4"
                          else self.root / "ardupilot/build/sitl/bin/arducopter")
                binary.write_text("updated firmware")
                self.assertFalse(prepare_run_logs(logs, force=False, inputs=capture_run_inputs(run, ap)))
                saved = next((run / f"{kind}_attempts").glob("*/attempt_001/gain_environment.json"))
                self.assertEqual(json.loads(saved.read_text())["comparison"]["status"], "match")

    def test_saved_snapshot_works_without_original_capture_and_strict_rechecks_current_files(self):
        output = self.generate("frozen", "require-match")
        run = output / "run_000"
        self.profile.unlink()
        self.meta.unlink()
        self.assertEqual(capture_run_inputs(run, "px4")["gain_environment"]["comparison"]["status"], "match")
        (self.root / "px4/build/px4_sitl_default/bin/px4").write_text("new")
        with self.assertRaisesRegex(ValueError, "require-match"):
            capture_run_inputs(run, "px4")

    def test_px4_inherited_overrides_are_reported(self):
        with patch.dict(env.os.environ, {"PX4_PARAM_MC_ROLLRATE_P": "0.99"}):
            current = env.collect_environment("px4", self.targets["px4"])
        origins = load_capture_provenance(self.profile, self.profiles)
        report = env.compare_environments(origins["px4"], current, "px4")
        self.assertEqual(report["status"], "mismatch")
        self.assertTrue(any(d["field"] == "startup_overrides" for d in report["differences"]))


if __name__ == "__main__":
    unittest.main()
