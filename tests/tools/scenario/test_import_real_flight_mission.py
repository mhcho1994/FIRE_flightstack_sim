"""Mission import, geometry, rejection and launcher compatibility without SITL."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from tools.scenario import import_real_flight_mission as importer
from tools.commander import pymavlink_ardupilot_commander as ardu
from tools.commander import pymavlink_px4_commander as px4
from tools.launcher.run_ardupilot_gz_sitl import _load_scenario_yaml_ardupilot
from tools.launcher.run_px4_gz_sitl import _load_scenario_yaml_px4
from tools.launcher.startup_parameters import collect_boot_parameters


def nav(command, lat=40.0, lon=-86.0, alt=10):
    return dict(type="SimpleItem", command=command, frame=3, autoContinue=True,
                Altitude=alt, AltitudeMode=1, params=[0, 0, 0, None, lat, lon, alt])


def plan():
    return dict(fileType="Plan", version=1, geoFence=dict(circles=[], polygons=[]),
                rallyPoints=dict(points=[]), mission=dict(version=2, firmwareType=3, vehicleType=2,
                plannedHomePosition=[40.0, -86.0, 157], hoverSpeed=1, cruiseSpeed=15,
                items=[nav(22, lat=40.0003, alt=2), nav(16, lat=40.0001, lon=-85.9999),
                       nav(16, lat=40.0002, lon=-85.9998, alt=20), nav(21, alt=0)]))


class ImportRealFlightMissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.src = self.root / "input"
        self.src.mkdir()
        self.path = self.save(plan())
        self.home = importer.simulator_home()

    def save(self, data, name="1.plan"):
        path = self.src / name
        path.write_text(json.dumps(data))
        return path

    def run_cli(self, *options, success=True):
        out = self.root / "output"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            args = ["--input", str(self.src), "--outdir", str(out), *options]
            if success:
                self.assertEqual(importer.main(args), 0)
            else:
                with self.assertRaises(SystemExit) as exc:
                    importer.main(args)
                self.assertEqual(exc.exception.code, 2)
        return out

    def test_relative_altitudes_and_local_route_survive_rebase(self):
        result = importer.read_plan(self.path, self.home, autopilots=["ardupilot", "px4"])
        mission = result.mission
        self.assertEqual(mission.command, [22, 16, 16, 21])
        self.assertEqual(mission.takeoff_alt_m, 2)
        self.assertEqual([p[2] if p else None for p in mission.waypoints_lla], [None, 10, 20, None])
        north, east, down = mission.waypoints_ned[1]
        self.assertAlmostEqual(north, 11.104, delta=.01)
        self.assertAlmostEqual(east, 8.540, delta=.01)
        self.assertEqual(down, -10)
        self.assertNotEqual(mission.waypoints_lla[1][:2], (40.0001, -85.9999))
        changes = result.report["normalizations"]
        self.assertTrue(any(c["field"] == "horizontal_position" for c in changes))
        self.assertTrue(any(c["field"] == "acceptance_radius_m" for c in changes))
        self.assertEqual(result.flight_parameters["ardupilot"]["values"], {"WPNAV_SPEED": 100})
        self.assertEqual(result.flight_parameters["px4"]["values"], {"MPC_XY_CRUISE": 1})

    def test_generated_runs_load_and_build_in_both_launchers(self):
        original = self.path.read_bytes()
        (self.src / "1.kml").write_text('<kml><not-a-mission/></kml>')
        out = self.run_cli("--autopilots", "ardupilot", "px4")
        run = out / "run_000"
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual((run / "source.plan").read_bytes(), original)
        data = yaml.safe_load((run / "scenario.yaml").read_text())
        for module, ap, loader in [(ardu, "ardupilot", _load_scenario_yaml_ardupilot),
                                   (px4, "px4", _load_scenario_yaml_px4)]:
            with self.subTest(ap=ap):
                items, altitude, _, land = module._build_items_from_scenario(data)
                self.assertEqual([i["command"] for i in items], [22, 16, 16, 21])
                self.assertEqual(items[1]["p7"], 10)
                self.assertEqual(items[2]["p7"], 20)
                self.assertEqual(items[1]["frame"], 3)
                self.assertEqual(altitude, 2)
                self.assertTrue(land)
                self.assertEqual(loader(run).location, "Purdue")
                self.assertIsNotNone(collect_boot_parameters(data, ap, run)[0])
        report = json.loads((out / "import_report.json").read_text())["runs"][0]
        self.assertFalse(report["kml_used"])
        self.assertEqual(len(report["kml_sidecars"]), 1)
        self.assertEqual(len(list(out.glob("run_*"))), 1)

    def test_source_firmware_is_default_and_batch_order_is_stable(self):
        other = plan(); other["mission"]["firmwareType"] = 12
        self.save(other, "2.PLAN")
        out = self.run_cli("--start-run-id", "5")
        first = yaml.safe_load((out / "run_005/scenario.yaml").read_text())
        second = yaml.safe_load((out / "run_006/scenario.yaml").read_text())
        self.assertEqual(list(first["autopilots"]), ["ardupilot"])
        self.assertEqual(list(second["autopilots"]), ["px4"])

    def test_speed_override_and_firmware_default(self):
        result = importer.read_plan(self.path, self.home, speed_m_s=2.5)
        self.assertEqual(result.flight_parameters["ardupilot"]["values"]["WPNAV_SPEED"], 250)
        data = plan(); del data["mission"]["hoverSpeed"]
        self.save(data)
        out = self.run_cli("--use-firmware-speed")
        scenario = yaml.safe_load((out / "run_000/scenario.yaml").read_text())
        self.assertNotIn("flight_parameters", scenario["autopilots"]["ardupilot"])
        self.assertEqual(scenario["common"]["scenario"]["command"], [22, 16, 16, 21])

    def test_explicit_speed_commands_keep_their_sequence_and_units(self):
        data = plan()
        data["mission"]["items"].insert(2, dict(type="SimpleItem", command=178, frame=2,
            autoContinue=True, params=[1, 3.5, -1, 0, 0, 0, 0]))
        self.save(data)
        out = self.run_cli("--autopilots", "ardupilot", "px4")
        scenario = yaml.safe_load((out / "run_000/scenario.yaml").read_text())
        for module in (ardu, px4):
            items = module._build_items_from_scenario(scenario)[0]
            self.assertEqual(items[2]["command"], 178)
            self.assertEqual(items[2]["p2"], 3.5)
            self.assertEqual(items[2]["p1"], 1)

    def test_unsupported_or_ambiguous_missions_fail_without_output(self):
        variants = []
        for field, value in [("command", 20), ("frame", 0), ("AltitudeMode", 2),
                             ("autoContinue", False), ("type", "ComplexItem"), ("Altitude", 999)]:
            data = plan(); data["mission"]["items"][1][field] = value; variants.append(data)
        for index, value in [(0, 2), (1, 5), (3, 90), (6, -1)]:
            data = plan(); data["mission"]["items"][1]["params"][index] = value; variants.append(data)
        data = plan(); data["geoFence"]["polygons"] = [{"polygon": []}]; variants.append(data)
        data = plan(); data["mission"]["hoverSpeed"] = 0; variants.append(data)
        for i, data in enumerate(variants):
            with self.subTest(case=i):
                self.save(data)
                self.assertFalse(self.run_cli(success=False).exists())

    def test_whole_batch_is_validated_before_publication(self):
        data = plan(); data["mission"]["items"][1]["command"] = 999
        self.save(data, "2.plan")
        self.assertFalse(self.run_cli(success=False).exists())

    def test_dry_run_and_no_overwrite(self):
        self.assertFalse(self.run_cli("--dry-run").exists())
        out = self.run_cli()
        original = (out / "run_000/scenario.yaml").read_bytes()
        self.run_cli(success=False)
        self.assertEqual((out / "run_000/scenario.yaml").read_bytes(), original)

    def test_kml_only_input_is_rejected(self):
        self.path.unlink()
        (self.src / "1.kml").write_text('<kml/>')
        self.assertFalse(self.run_cli(success=False).exists())

    def test_gain_profile_and_speed_boot_values_are_combined(self):
        profile = importer.ROOT / "tools/scenario/fixtures/default_controller_gain.yaml"
        out = self.run_cli("--controller-gain-profile", str(profile), "--autopilots", "ardupilot", "px4")
        run = out / "run_000"
        data = yaml.safe_load((run / "scenario.yaml").read_text())
        for ap, speed_name in (("ardupilot", "WPNAV_SPEED"), ("px4", "MPC_XY_CRUISE")):
            content, values, sources = collect_boot_parameters(data, ap, run)
            self.assertIn(speed_name, values)
            self.assertGreater(len(values), 10)
            self.assertEqual(len(sources), 2)
            self.assertIn(speed_name.encode(), content)


if __name__ == "__main__":
    unittest.main()
