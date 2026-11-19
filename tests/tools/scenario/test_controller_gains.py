"""Controller gain sampling and generator round trips, without starting SITL."""

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.commander.fc_parameter_profiles import GAIN_GROUPS, load_profiles, parse_controller_parameters, resolve_profiles
from tools.scenario.create_params_sets import create_params_sets


EXAMPLE = Path(__file__).with_name("fixtures") / "controller_gain_experiments.yaml"
GENERATOR = ROOT / "tools/scenario/scenario_generator.py"


class ControllerGainScenarios(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = create_params_sets(EXAMPLE)

    def load(self, raw):
        path = self.root / "profile.yaml"
        path.write_text(yaml.safe_dump(raw))
        return load_profiles(path)

    def test_operations_and_missing_overrides_restore_baseline(self):
        profiles = self.load(self.raw)
        result = resolve_profiles(profiles, 42, 0)
        ardu = result["ardupilot"]
        for name, rule in profiles["ardupilot"]["parameters"].items():
            draw = ardu["sampling"]["draws"][name]
            base = ardu["baseline"][name]
            expected = {"set": draw, "scale": base * draw, "offset": base + draw}[rule["operation"]]
            self.assertEqual(ardu["values"][name], expected)
        self.assertEqual(ardu["values"]["ATC_RAT_PIT_D"], 0.0036)
        self.assertGreater(ardu["values"]["PSC_VELZ_D"], 0)
        for autopilot, count in (("ardupilot", 23), ("px4", 20)):
            self.assertEqual(len(parse_controller_parameters(result[autopilot], autopilot)["values"]), count)
        manual = parse_controller_parameters(dict(groups=ardu["groups"], baseline=ardu["baseline"],
                                                  values={"ATC_RAT_RLL_P": .19}), "ardupilot")
        self.assertEqual(manual["values"]["ATC_RAT_RLL_P"], .19)
        self.assertEqual(manual["values"]["ATC_RAT_PIT_P"], .135)

    def test_sampling_is_stable_by_parameter_axis_autopilot_and_run(self):
        profiles = self.load(self.raw)
        first = resolve_profiles(profiles, 7, 11)
        self.assertEqual(first, resolve_profiles(profiles, 7, 11))
        self.assertNotEqual(first["ardupilot"]["values"], resolve_profiles(profiles, 7, 12)["ardupilot"]["values"])
        changed = copy.deepcopy(profiles)
        changed["ardupilot"]["parameters"]["ATC_RAT_PIT_D"] = dict(operation="set", value=.004)
        changed["ardupilot"]["parameters"] = dict(reversed(list(changed["ardupilot"]["parameters"].items())))
        other = resolve_profiles(changed, 7, 11)
        for name, value in first["ardupilot"]["values"].items():
            if name != "ATC_RAT_PIT_D":
                self.assertEqual(other["ardupilot"]["values"][name], value)
        self.assertEqual(other["px4"], first["px4"])
        for profile in profiles.values():
            profile["parameters"] = {name: dict(operation="set", uniform=[.1, .2])
                                     for name in profile["baseline"]}
        draws = [value for config in resolve_profiles(profiles, 7, 11).values() for value in config["values"].values()]
        self.assertEqual(len(set(draws)), 43)

    def test_snapshot_is_frozen_and_hash_checked(self):
        snapshot = self.root / "mav.parm"
        baseline = self.raw["ardupilot"]["baseline"]
        snapshot.write_text("# Snapshot\nOTHER_PARAMETER 42\n" + "\n".join(f"{k} {v}" for k, v in baseline.items()))
        self.raw["ardupilot"]["baseline"] = "mav.parm"
        profiles = self.load(self.raw)
        snapshot.unlink()
        resolved = resolve_profiles(profiles, 1, 1)["ardupilot"]
        self.assertEqual(resolved["baseline"], baseline)
        resolved["baseline"]["ATC_RAT_RLL_P"] = .2
        with self.assertRaisesRegex(ValueError, "sha256"):
            parse_controller_parameters(resolved, "ardupilot")

    def test_invalid_profiles_fail_before_sampling(self):
        bad_rules = [
            dict(operation="scale", value=1, uniform=[.9, 1.1]),
            dict(operation="offset", uniform=[-.2, .01]),
            dict(operation="set", value=float("nan")),
            dict(operation="set", value=True),
            dict(operation="scale", uniform=[2, 1]),
            dict(operation="scale", uniform=[.9, 1.1], bounds=[.13, .14]),
            dict(operation="typo", value=1),
            dict(operation="set", value=1e40),
            dict(operation="set", value=1e-50),
        ]
        for rule in bad_rules:
            with self.subTest(rule=rule):
                raw = copy.deepcopy(self.raw)
                raw["ardupilot"]["parameters"]["ATC_RAT_RLL_P"] = rule
                with self.assertRaises(ValueError):
                    self.load(raw)
        raw = copy.deepcopy(self.raw)
        raw["ardupilot"]["parameters"]["ARMING_CHECK"] = dict(operation="set", value=0)
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.load(raw)
        raw = copy.deepcopy(self.raw)
        del raw["ardupilot"]["baseline"]["ATC_RAT_YAW_D"]
        with self.assertRaisesRegex(ValueError, "all 23"):
            self.load(raw)
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_controller_parameters(self.raw["ardupilot"], "ardupilot")  # Unresolved rules are not executable.

    def test_all_controller_groups_resolve(self):
        profiles = load_profiles(EXAMPLE)
        resolved = resolve_profiles(profiles, 42, 0)
        for ap, count in (("ardupilot", 23), ("px4", 20)):
            config = parse_controller_parameters(resolved[ap], ap)
            self.assertEqual(len(config["values"]), count)
            self.assertEqual(config["groups"], list(GAIN_GROUPS[ap]))
        self.assertEqual(resolved["ardupilot"]["values"]["PSC_POSZ_P"], 1.05)
        self.assertEqual(resolved["px4"]["values"]["MPC_Z_VEL_P_ACC"], 5.1)
        self.assertGreater(resolved["ardupilot"]["values"]["PSC_VELZ_D"], 0)

    def test_each_group_can_be_selected_without_other_groups(self):
        raw = yaml.safe_load(EXAMPLE.read_text())
        for ap, groups in GAIN_GROUPS.items():
            for group, names in groups.items():
                with self.subTest(ap=ap, group=group):
                    baseline = {name: raw[ap]["baseline"][name] for name in names}
                    profile = {ap: dict(groups=[group], baseline=baseline,
                                       parameters={names[0]: dict(operation="scale", value=1.05)})}
                    config = resolve_profiles(self.load(profile), 1, 0)[ap]
                    self.assertEqual(set(config["values"]), set(names))
                    parsed = parse_controller_parameters(config, ap)
                    self.assertEqual(parsed["groups"], [group])
                    del profile[ap]["baseline"][names[-1]]
                    with self.assertRaisesRegex(ValueError, "required"):
                        self.load(profile)

    def test_snapshot_selects_requested_groups(self):
        raw = yaml.safe_load(EXAMPLE.read_text())
        for ap in GAIN_GROUPS:
            snapshot = self.root / f"{ap}.parm"
            snapshot.write_text("\n".join(f"{name} {value}" for name, value in raw[ap]["baseline"].items()))
            groups = ["position", "velocity"]
            selected = self.load({ap: dict(groups=groups, baseline=snapshot.name)})[ap]
            expected = {name for group in groups for name in GAIN_GROUPS[ap][group]}
            self.assertEqual(set(selected["baseline"]), expected)

    def test_invalid_groups_and_cross_group_overrides_are_rejected(self):
        for groups in ([], "position", ["rate", "rate"], ["unknown"], [None], ["acceleration"]):
            with self.subTest(groups=groups):
                raw = copy.deepcopy(self.raw)
                raw["px4"]["groups"] = groups
                with self.assertRaises(ValueError):
                    self.load(raw)
        raw = create_params_sets(EXAMPLE, groups=["position", "velocity"])
        raw["px4"]["parameters"]["MC_ROLL_P"] = dict(operation="set", value=6)
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.load(raw)
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_controller_parameters(dict(groups=raw["px4"]["groups"],
                baseline=raw["px4"]["baseline"], values={"MC_ROLL_P": 6}), "px4")

    def test_controller_cli_and_group_reordering_do_not_change_samples(self):
        new = self.generate("new_cli", ["--controller-gains", str(EXAMPLE)])
        alias = self.generate("old_cli", ["--rate-gains", str(EXAMPLE)])
        self.assertEqual((new / "run_000/scenario.yaml").read_text(), (alias / "run_000/scenario.yaml").read_text())
        profiles = load_profiles(EXAMPLE)
        expected = resolve_profiles(profiles, 42, 0)
        raw = yaml.safe_load(EXAMPLE.read_text())
        for ap in raw:
            raw[ap]["groups"].reverse()
            raw[ap]["parameters"] = dict(reversed(list(raw[ap]["parameters"].items())))
        reordered = resolve_profiles(self.load(raw), 42, 0)
        for ap in expected:
            self.assertEqual(expected[ap]["values"], reordered[ap]["values"])
            self.assertEqual(expected[ap]["sampling"]["draws"], reordered[ap]["sampling"]["draws"])

    def generate(self, name, options=(), success=True):
        out = self.root / name
        result = subprocess.run([sys.executable, "-B", str(GENERATOR), "planar_n_pts",
                                 "--outdir", str(out), "--seed", "17", "--runs", "2",
                                 "--edge-m", "random", "--wind-horizontal-magnitude-m-s", "random",
                                 *options], text=True, capture_output=True, cwd=ROOT)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return out

    def test_generator_replay_and_existing_streams(self):
        plain = self.generate("plain")
        with_gains = self.generate("gains", ["--controller-gains", str(EXAMPLE)])
        selected = self.generate("selected", ["--controller-gains", str(EXAMPLE), "--start-run-id", "1", "--runs", "1"])
        for run in (0, 1):
            a = yaml.safe_load((plain / f"run_{run:03d}/scenario.yaml").read_text())
            b = yaml.safe_load((with_gains / f"run_{run:03d}/scenario.yaml").read_text())
            self.assertEqual(a["common"], b["common"])
            for ap in GAIN_GROUPS:
                self.assertNotIn("controller_gains", a["autopilots"][ap])
                parse_controller_parameters(b["autopilots"][ap]["controller_gains"], ap)
        self.assertEqual((selected / "run_001/scenario.yaml").read_text(), (with_gains / "run_001/scenario.yaml").read_text())
        metadata = yaml.safe_load((with_gains / "metadata.yaml").read_text())
        self.assertIn("controller_gain_sampling", metadata["dataset"])
        self.assertIn("baseline_sha256", metadata["flight_stacks"]["px4"]["controller_gains"])
        bad = self.root / "bad.yaml"
        bad.write_text("px4: {baseline: {}}")
        out = self.generate("invalid", ["--controller-gains", str(bad)], success=False)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
