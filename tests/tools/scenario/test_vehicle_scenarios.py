"""Run: python3 -B -m unittest discover -s tests/tools/scenario -v."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/scenario"))
sys.path.insert(0, str(ROOT / "tools/launcher"))

from vehicle_model_builder import VehicleConfig, generate_vehicle_model, prepare_ardupilot_world
import run_ardupilot_gz_sitl as ardupilot
import run_px4_gz_sitl as px4

SOURCE = ROOT / "gz/FIRE_moonshot_gazebo/models/fire_px4vision/model.sdf"
WORLD = ROOT / "gz/FIRE_moonshot_gazebo/worlds/default_fire_px4vision.sdf"
GENERATOR = ROOT / "tools/scenario/scenario_generator.py"


class VehicleScenarioTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def generate(self, name="runs", options=(), pattern="planar_n_pts", success=True):
        out = self.root / name
        result = subprocess.run(
            [sys.executable, str(GENERATOR), pattern, "--outdir", str(out), *options],
            cwd=ROOT, text=True, capture_output=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out.exists(), "Invalid input must fail before creating a dataset")
        return out

    def scenario(self, out, run_id=0):
        return yaml.safe_load((out / f"run_{run_id:03d}/scenario.yaml").read_text())

    def test_mission_only_wind_policy_is_written_and_validated(self):
        expected = {"mode": "mission_only", "ramp_s": 2.0,
                    "min_settle_s": 5.0, "settle_timeout_s": 30.0}
        for spelling in ("mission_only", "mission-only"):
            with self.subTest(spelling=spelling):
                out = self.generate(spelling, ["--seed", "1", "--wind-policy", spelling,
                                               "--wind-ramp-s", "2"])
                self.assertEqual(self.scenario(out)["common"]["scenario"]["wind_policy"], expected)
                metadata = yaml.safe_load((out / "metadata.yaml").read_text())
                self.assertEqual(metadata["simulation"]["wind"]["policy"], expected)
                run = out / "run_000"
                self.assertEqual(ardupilot._load_scenario_yaml_ardupilot(run).wind_policy.mode, "mission_only")
                self.assertEqual(px4._load_scenario_yaml_px4(run).wind_policy.mode, "mission_only")
                self.generate(f"bad_wind_{spelling}",
                              ["--wind-policy", spelling, "--wind-ramp-s", "nan"], success=False)
                self.generate(f"no_land_{spelling}",
                              ["--wind-policy", spelling, "--land", "false"], success=False)

    def test_gust_and_turbulence_options_reach_both_launchers(self):
        options = ["--seed", "41", "--gust-magnitude-m-s", "3", "--gust-start-s", "2",
                   "--gust-duration-s", "4", "--gust-period-s", "8", "--gust-azimuth-deg", "90",
                   "--gust-elevation-deg", "-30", "--turbulence-magnitude-m-s", ".6",
                   "--turbulence-length-scale-m", "40", "30", "20",
                   "--turbulence-reference-speed-m-s", "8", "--turbulence-azimuth-deg", "45",
                   "--turbulence-sample-period-s", ".05", "--turbulence-seed", "123"]
        for pattern in ("planar_n_pts", "three_d_n_pts"):
            out = self.generate(pattern, options, pattern)
            scenario = self.scenario(out)["common"]["scenario"]
            self.assertEqual(scenario["gust"], dict(magnitude_m_s=3, start_s=2, duration_s=4,
                                                    period_s=8, azimuth_deg=90, elevation_deg=-30))
            for loader in (px4._load_scenario_yaml_px4, ardupilot._load_scenario_yaml_ardupilot):
                cfg = loader(out / "run_000")
                self.assertEqual(cfg.gust.magnitude_m_s, 3)
                self.assertEqual(cfg.turbulence.magnitude_m_s, .6)
                self.assertEqual(cfg.turbulence.length_scale_m, (40, 30, 20))
                self.assertEqual(cfg.turbulence.reference_speed_m_s, 8)
                self.assertEqual(cfg.turbulence.azimuth_deg, 45)
                self.assertEqual(cfg.turbulence.sample_period_s, .05)
                self.assertEqual(cfg.turbulence.seed, 123)
            metadata = yaml.safe_load((out / "metadata.yaml").read_text())["simulation"]["wind"]
            self.assertEqual(metadata["gust"]["model"], "1-minus-cosine")
            self.assertEqual(metadata["turbulence"]["model"], "dryden")
            self.assertEqual(metadata["turbulence"]["seed"], 123)

    def test_disturbance_random_streams_are_reproducible_and_independent(self):
        common = ["--seed", "15", "--runs", "3", "--edge-m", "random",
                  "--wind-horizontal-magnitude-m-s", "random"]
        options = [*common, "--gust-magnitude-m-s", "random", "--gust-magnitude-m-s-range", "1", "2",
                   "--turbulence-magnitude-m-s", "random", "--turbulence-magnitude-m-s-range", ".3", ".8"]
        first = self.generate("first", options)
        second = self.generate("second", options)
        calm = self.generate("calm", common)
        selected = self.generate("selected", [*options, "--start-run-id", "2", "--runs", "1"])
        seeds = []
        for run in range(3):
            a = self.scenario(first, run)
            self.assertEqual(a, self.scenario(second, run))
            scenario = a["common"]["scenario"]
            self.assertTrue(1 <= scenario["gust"]["magnitude_m_s"] <= 2)
            self.assertTrue(.3 <= scenario["turbulence"]["magnitude_m_s"] <= .8)
            seeds.append(scenario["turbulence"]["seed"])
            baseline = self.scenario(calm, run)["common"]["scenario"]
            self.assertEqual(baseline["gust"]["magnitude_m_s"], 0)
            self.assertEqual(baseline["turbulence"]["magnitude_m_s"], 0)
            for key in ("gust", "turbulence"):
                scenario.pop(key)
                baseline.pop(key)
            self.assertEqual(scenario, baseline)
        self.assertEqual(len(set(seeds)), 3)
        self.assertEqual(self.scenario(first, 2), self.scenario(selected, 2))

    def test_invalid_disturbances_fail_before_writing_or_loading(self):
        options = [["--gust-magnitude-m-s", "nan"], ["--gust-duration-s", "0"],
                   ["--gust-period-s", "1"], ["--gust-azimuth-deg", "361"],
                   ["--gust-elevation-deg", "91"], ["--gust-magnitude-m-s-range", "-1", "2"],
                   ["--turbulence-magnitude-m-s", "-1"], ["--turbulence-magnitude-m-s-range", "0", "inf"],
                   ["--turbulence-seed", "-1"], ["--turbulence-reference-speed-m-s", "0"],
                   ["--turbulence-length-scale-m", "1", "0", "2"], ["--turbulence-sample-period-s", "0"]]
        for i, args in enumerate(options):
            self.generate(f"bad_disturbance_{i}", args, success=False)
        out = self.generate("valid", ["--seed", "1"])
        path = out / "run_000/scenario.yaml"
        original = self.scenario(out)
        for key in ("gust", "turbulence"):
            data = yaml.safe_load(yaml.safe_dump(original))
            data["common"]["scenario"][key]["magnitude_m_s"] = float("nan")
            path.write_text(yaml.safe_dump(data))
            for loader in (px4._load_scenario_yaml_px4, ardupilot._load_scenario_yaml_ardupilot):
                with self.assertRaises(ValueError):
                    loader(path.parent)

    def test_fixed_scales_and_legacy_defaults(self):
        legacy = self.scenario(self.generate("legacy", ["--seed", "1"]))
        self.assertNotIn("vehicle", legacy["common"])
        self.assertEqual(legacy["autopilots"]["ardupilot"]["sim"]["world"], "iris_runway")
        self.assertEqual(legacy["autopilots"]["px4"]["sim"]["frame"], "gz_x500")
        for pattern in ("planar_n_pts", "three_d_n_pts"):
            with self.subTest(pattern=pattern):
                data = self.scenario(self.generate(pattern, ["--base-mass-scale", "1.2"], pattern))
                inertial = data["common"]["vehicle"]["inertial"]
                self.assertEqual(inertial["mass_scale"], 1.2)
                self.assertEqual(inertial["inertia_scale"], 1)
                self.assertEqual(data["autopilots"]["ardupilot"]["sim"]["world"], "default_fire_px4vision")
                self.assertEqual(data["autopilots"]["px4"]["sim"]["frame"], "gz_fire_px4vision")
                self.assertIsInstance(data["meta"]["seed"], int)

    def test_random_ranges_seed_and_independent_streams(self):
        mission = ["--seed", "260930", "--runs", "4", "--edge-m", "random",
                   "--wind-horizontal-magnitude-m-s", "random"]
        options = [*mission, "--base-mass-scale", "random", "--base-mass-scale-range", "0.7", "1.3",
                   "--base-inertia-scale", "random", "--base-inertia-scale-range", "1.4", "1.7"]
        first = self.generate("first", options)
        second = self.generate("second", options)
        legacy = self.generate("legacy", mission)
        selected = self.generate("selected", [*options, "--start-run-id", "2", "--runs", "1"])
        values = []
        for run_id in range(4):
            a, b = self.scenario(first, run_id), self.scenario(second, run_id)
            self.assertEqual(a, b)
            self.assertEqual(a["common"]["scenario"], self.scenario(legacy, run_id)["common"]["scenario"])
            inertial = a["common"]["vehicle"]["inertial"]
            self.assertTrue(0.7 <= inertial["mass_scale"] <= 1.3)
            self.assertTrue(1.4 <= inertial["inertia_scale"] <= 1.7)
            values.append(inertial["mass_scale"])
        self.assertEqual(self.scenario(first, 2), self.scenario(selected, 2))
        self.assertGreater(len(set(values)), 1)
        metadata = yaml.safe_load((first / "metadata.yaml").read_text())
        self.assertEqual(metadata["simulation"]["vehicle"]["inertial"]["mass_scale"]["range"], [0.7, 1.3])

    def test_invalid_inputs_fail_before_writing(self):
        options = [
            ["--base-mass-scale", "0"], ["--base-inertia-scale", "-1"],
            ["--base-mass-scale", "nan"], ["--base-inertia-scale", "inf"],
            ["--base-mass-scale-range", "1.2", "0.8"],
            ["--base-inertia-scale-range", "0", "1"],
            ["--base-mass-scale-range", "0.8", "inf"], ["--seed", "-1"],
            ["--base-mass-scale", "random", "--px4-frame", "gz_x500"],
        ]
        for index, args in enumerate(options):
            with self.subTest(args=args):
                self.generate(str(index), args, success=False)

    def test_model_scales_only_base_and_keeps_assets_and_source(self):
        original = SOURCE.read_bytes()
        config = VehicleConfig("fire_px4vision", mass_scale=1.2, inertia_scale=1.1)
        result = generate_vehicle_model(config, SOURCE, self.root)
        model = ET.parse(result).getroot().find("model")
        base = model.find("link[@name='base_link']/inertial")
        self.assertAlmostEqual(float(base.findtext("mass")), 1.8)
        self.assertAlmostEqual(float(base.findtext("inertia/ixx")), 0.029125 * 1.1)
        self.assertAlmostEqual(float(base.findtext("inertia/iyy")), 0.029125 * 1.1)
        self.assertAlmostEqual(float(base.findtext("inertia/izz")), 0.055225 * 1.1)
        source_model = ET.fromstring(original).find("model")
        for index in range(4):
            path = f"link[@name='rotor_{index}']/inertial"
            for term in ("mass", "inertia/ixx", "inertia/iyy", "inertia/izz"):
                self.assertEqual(float(model.findtext(f"{path}/{term}")),
                                 float(source_model.findtext(f"{path}/{term}")))
        self.assertEqual([ET.canonicalize(ET.tostring(p, encoding="unicode"), strip_text=True) for p in source_model.findall("plugin")],
                         [ET.canonicalize(ET.tostring(p, encoding="unicode"), strip_text=True) for p in model.findall("plugin")])
        for uri in model.findall(".//mesh/uri"):
            self.assertTrue(Path(uri.text).is_file(), uri.text)
        manifest = yaml.safe_load((self.root / "generated/resolved_vehicle.yaml").read_text())
        self.assertAlmostEqual(manifest["total_mass_kg"], 1.82)
        self.assertAlmostEqual(manifest["source_total_mass_kg"], 1.52)
        first = result.read_bytes()
        generate_vehicle_model(config, SOURCE, self.root)
        self.assertEqual(first, result.read_bytes(), "Repeated preparation must not compound scales")
        self.assertEqual(original, SOURCE.read_bytes())

    def test_world_preserves_adapter_plugins_and_entity_name(self):
        original = WORLD.read_bytes()
        result = prepare_ardupilot_world(VehicleConfig("fire_px4vision"), str(WORLD), self.root)
        include = ET.parse(result).getroot().find("world/include")
        source_include = ET.fromstring(original).find("world/include")
        self.assertEqual(include.findtext("name"), source_include.findtext("name"))
        self.assertEqual(include.findtext("pose"), source_include.findtext("pose"))
        self.assertTrue(Path(include.findtext("uri")).is_absolute())
        self.assertTrue(Path(include.findtext("uri")).is_file())
        self.assertEqual([ET.canonicalize(ET.tostring(p, encoding="unicode"), strip_text=True) for p in include.findall("plugin")],
                         [ET.canonicalize(ET.tostring(p, encoding="unicode"), strip_text=True) for p in source_include.findall("plugin")])
        self.assertEqual(original, WORLD.read_bytes())

    def test_mismatched_world_and_invalid_tensor_are_rejected(self):
        world = self.root / "wrong.sdf"
        world.write_text('<sdf><world name="wrong"><include><uri>model://iris</uri></include></world></sdf>')
        with self.assertRaisesRegex(ValueError, "exactly once"):
            prepare_ardupilot_world(VehicleConfig("fire_px4vision"), str(world), self.root)
        tree = ET.parse(SOURCE)
        tree.getroot().find("model/link/inertial/inertia/ixx").text = "0.0233"
        invalid = self.root / "invalid.sdf"
        tree.write(invalid)
        with self.assertRaisesRegex(ValueError, "triangle inequality"):
            generate_vehicle_model(VehicleConfig("fire_px4vision"), invalid, self.root)
        self.assertFalse((self.root / "generated").exists())

    def test_prepare_only_and_launcher_use_generated_world(self):
        out = self.generate(options=["--base-mass-scale", "1.2", "--base-inertia-scale", "1.1"])
        argv = ["launcher", "--run-root", str(out), "--prepare-only"]
        with contextlib.redirect_stdout(io.StringIO()), patch.object(sys, "argv", argv), \
             patch.object(ardupilot.signal, "signal"), patch.object(ardupilot, "_popen") as launch:
            self.assertEqual(ardupilot.main(), 0)
            launch.assert_not_called()
        self.assertFalse((out / "run_000/ardu_logs").exists())
        runner = Mock()
        runner.is_done.return_value = True
        runner.get_status.return_value.success = True
        # Exercise main -> prepare -> run_once -> gz command, without processes.
        with contextlib.redirect_stdout(io.StringIO()), patch.object(sys, "argv", argv[:-1]), \
             patch.object(ardupilot.signal, "signal"), patch.object(ardupilot, "_ensure_ardupilot_built"), \
             patch.object(ardupilot, "_popen", return_value=Mock()) as launch, \
             patch.object(ardupilot, "_finalize_proc"), patch.object(ardupilot.time, "sleep"), \
             patch.object(ardupilot, "ArduPilotMissionRunner", return_value=runner), \
             patch.object(ardupilot, "set_world_wind") as wind, \
             patch.object(ardupilot, "_check_ardupilot_logs", return_value=True):
            self.assertEqual(ardupilot.main(), 0)
        command = launch.call_args_list[0].args[1]
        self.assertEqual(command[-1], str(out / "run_000/generated/world_ardupilot.sdf"))
        self.assertEqual(wind.call_args.args, ("default_fire_px4vision", (0.0, 0.0, 0.0)))

    def test_px4_spawns_generated_model(self):
        out = self.generate(options=["--base-mass-scale", "1.2"])
        run = out / "run_000"
        cfg = px4._load_scenario_yaml_px4(run)
        runner = Mock()
        runner.is_done.return_value = True
        runner.get_status.return_value.success = True
        handle = Mock()
        handle.proc.poll.return_value = None
        with contextlib.redirect_stdout(io.StringIO()), patch.object(px4, "px4_command", return_value=["px4"]), \
             patch.object(px4, "_popen", return_value=handle), patch.object(px4, "spawn_model") as spawn, \
             patch.object(px4, "_finalize_proc"), patch.object(px4, "PX4MissionRunner", return_value=runner):
            rc = px4.run_once(ROOT / "ap/px4", 0, run / "scenario.yaml", run / "px4_logs", 14550,
                              cfg.mavlink_url, 0, 1, {"flag": False}, {}, str(cfg.vehicle), cfg.frame,
                              cfg.world, "Purdue", True, False, vehicle_config=cfg.vehicle_config)
        self.assertEqual(rc, 0)
        self.assertEqual(spawn.call_args.args[0].model_path, run / "generated/vehicle/model.sdf")


if __name__ == "__main__":
    unittest.main()
