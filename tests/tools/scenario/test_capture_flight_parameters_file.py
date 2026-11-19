"""Decode real-format parameter records and round-trip them through boot profiles."""

from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.commander.fc_parameter_profiles import GAIN_GROUPS, parameter_names
from tools.scenario import capture_flight_parameters as capture
from tools.scenario.create_params_sets import create_params_sets, resolve_params_sets
from tools.launcher.startup_parameters import validate_parameter_file


def values(autopilot):
    return {name: (index + 1) / 32 for index, name in
            enumerate(parameter_names(autopilot, GAIN_GROUPS[autopilot]))}


def write_bin(path, parameters, changes=(), *,
              firmware_messages=("ArduCopter V4.6.2 (abcdef01)",), version_records=()):
    def fmt(kind, length, name, format_string, columns):
        return b"\xa3\x95\x80" + struct.pack("<BB4s16s64s", kind, length,
            name.encode(), format_string.encode(), columns.encode())
    content = fmt(128, 89, "FMT", "BBnNZ", "Type,Length,Name,Format,Columns")
    content += fmt(129, 35, "PARM", "QNff", "TimeUS,Name,Value,Default")
    content += fmt(130, 75, "MSG", "QZ", "TimeUS,Message")
    content += fmt(131, 83, "VER", "QBBBBIZ", "TimeUS,Maj,Min,Pat,FWT,GH,FWS")
    for text in firmware_messages:
        content += b"\xa3\x95\x82" + struct.pack("<Q64s", 1000, text.encode())
    for record in version_records:
        content += b"\xa3\x95\x83" + struct.pack("<QBBBBI64s", 1000,
            record["Maj"], record["Min"], record["Pat"], record["FWT"], record["GH"], record["FWS"].encode())
    for timestamp, name, value in [(1000, k, v) for k, v in parameters.items()] + list(changes):
        # A different Default value ensures extraction uses Value, not Default.
        content += b"\xa3\x95\x81" + struct.pack("<Q16sff", timestamp, name.encode(), value, 99.0)
    path.write_bytes(content)
    return path


def write_ulog(path, parameters, changes=(), *, firmware=None):
    def msg(kind, payload):
        return struct.pack("<HB", len(payload), ord(kind)) + payload
    def param(name, value):
        key = ("float " + name).encode()
        return struct.pack("<B", len(key)) + key + struct.pack("<f", value)
    content = b"ULog\x01\x12\x35\x01" + struct.pack("<Q", 1000)
    content += msg("B", bytes(40))
    if firmware is None:
        firmware = dict(sys_name="PX4", ver_sw_release=0x010f04ff, ver_sw="abcdef0123456789")
    for name, value in firmware.items():
        if isinstance(value, int):
            key, data = f"uint32_t {name}".encode(), struct.pack("<I", value)
        else:
            data = value.encode()
            key = f"char[{len(data)}] {name}".encode()
        content += msg("I", struct.pack("<B", len(key)) + key + data)
    content += msg("F", b"test:uint64_t timestamp;float value;")
    for name, value in parameters.items():
        content += msg("P", param(name, value))
        content += msg("Q", b"\x01" + param(name, 99.0))
    content += msg("A", struct.pack("<BH", 0, 0) + b"test")
    content += msg("D", struct.pack("<HQf", 0, 1500, 0))
    for timestamp, name, value in changes:
        content += msg("D", struct.pack("<HQf", 0, timestamp, 0))
        content += msg("P", param(name, value))
    path.write_bytes(content)
    return path


class CaptureFlightParametersFileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="real flight parameters ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_both_formats_preserve_initial_final_and_changes_across_all_groups(self):
        for ap, suffix, writer in (("ardupilot", ".BIN", write_bin), ("px4", ".ulg", write_ulog)):
            with self.subTest(autopilot=ap):
                baseline = values(ap)
                names = (GAIN_GROUPS[ap]["attitude"][0], GAIN_GROUPS[ap]["velocity"][0])
                changes = [(2000, names[0], baseline[names[0]]),
                           (3000, names[0], 2.0), (4000, names[0], baseline[names[0]]),
                           (5000, names[1], 3.0)]
                path = writer(self.root / (ap + suffix), baseline, changes)
                initial, report = capture.read_flight_parameters(path)
                final, _ = capture.read_flight_parameters(path, snapshot="final")
                self.assertEqual(initial[ap]["baseline"], baseline)
                self.assertEqual(final[ap]["baseline"], dict(baseline, **{names[1]: 3.0}))
                self.assertEqual(initial[ap]["groups"], list(GAIN_GROUPS[ap]))
                self.assertEqual(initial[ap]["parameters"], {})
                self.assertEqual(report["log_format"], "bin" if ap == "ardupilot" else "ulog")
                self.assertEqual(report["firmware_identity"]["autopilot"], ap)
                self.assertEqual(report["firmware_identity"]["version"], "4.6.2" if ap == "ardupilot" else "1.15.4")
                self.assertTrue(report["firmware_identity"]["evidence"])
                if ap == "px4":
                    self.assertEqual(report["firmware_identity"]["version_type"], 255)
                    self.assertEqual(report["firmware_identity"]["git_hash"], "abcdef0123456789")
                self.assertEqual([event["timestamp_us"] for event in report["history"]["changes"]],
                                 [3000, 4000, 5000])
                self.assertEqual(report["history"]["changes"][0]["before"], baseline[names[0]])

    def test_bin_ver_identifies_firmware_without_msg(self):
        record = dict(FWS="ArduCopter V4.6.3 (3fc7011a)", Maj=4, Min=6, Pat=3,
                      FWT=128, GH=0x3fc7011a)
        path = write_bin(self.root / "ver_only.BIN", values("ardupilot"),
                         firmware_messages=(), version_records=[record])
        profile, report = capture.read_flight_parameters(path)
        self.assertEqual(profile["ardupilot"]["baseline"], values("ardupilot"))
        self.assertEqual(report["history"]["firmware"], [])
        identity = report["firmware_identity"]
        self.assertEqual(identity["version"], "4.6.3")
        self.assertEqual(identity["version_type"], 128)
        self.assertEqual(identity["git_hash"], "3fc7011a")
        self.assertIn(dict(field="VER", value=record), identity["evidence"])

    def test_px4_converted_bin_uses_px4_registry_and_requires_all_selected_gains(self):
        path = write_bin(self.root / "converted.BIN", values("px4"),
                         firmware_messages=["PX4 flight.ulg via ulog_to_bin.py"])
        profile, report = capture.read_flight_parameters(path, parameter_schema="px4-mc-v1")
        self.assertEqual(profile["px4"]["baseline"], values("px4"))
        self.assertEqual(report["log_format"], "bin")
        self.assertEqual(report["firmware_identity"]["autopilot"], "px4")
        self.assertIsNone(report["firmware_identity"]["version"])
        with self.assertRaisesRegex(ValueError, "not ardupilot"):
            capture.read_flight_parameters(path, autopilot="ardupilot")
        missing = values("px4")
        del missing["MPC_XY_P"]
        write_bin(path, missing, firmware_messages=["PX4 v1.15.4"])
        with self.assertRaisesRegex(ValueError, "px4: missing initial.*MPC_XY_P"):
            capture.read_flight_parameters(path)
        partial, _ = capture.read_flight_parameters(path, groups=["attitude"])
        self.assertEqual(set(partial["px4"]["baseline"]), set(GAIN_GROUPS["px4"]["attitude"]))

    def test_header_selects_decoder_even_with_misleading_suffix(self):
        for ap, writer, suffix, kind in (("px4", write_ulog, ".BIN", "ulog"),
                                         ("ardupilot", write_bin, ".ulg", "bin")):
            with self.subTest(ap=ap):
                path = writer(self.root / ("renamed" + suffix), values(ap))
                profile, report = capture.read_flight_parameters(path)
                self.assertEqual(profile[ap]["baseline"], values(ap))
                self.assertEqual(report["log_format"], kind)
                opposite = "px4" if ap == "ardupilot" else "ardupilot"
                with self.assertRaisesRegex(ValueError, f"not {opposite}"):
                    capture.read_flight_parameters(path, autopilot=opposite)

    def test_missing_unknown_or_conflicting_identity_preserves_existing_capture(self):
        record = dict(FWS="ArduCopter V4.6.3 (3fc7011a)", Maj=4, Min=6, Pat=3,
                      FWT=128, GH=0x3fc7011a)
        paths = [
            write_bin(self.root / "no_identity.BIN", values("ardupilot"), firmware_messages=[]),
            write_bin(self.root / "board_only.BIN", values("ardupilot"), firmware_messages=["PX4v2 board"]),
            write_ulog(self.root / "no_identity.ulg", values("px4"),
                       firmware=dict(ver_sw_release=0x010f04ff, ver_hw="PX4_FMU_V6C")),
            write_ulog(self.root / "unknown.ulg", values("px4"),
                       firmware=dict(sys_name="CogniPilot", ver_sw_release=0x010f04ff)),
            write_bin(self.root / "conflicting_stack.BIN", values("px4"),
                      firmware_messages=["PX4 v1.15.4"], version_records=[record]),
            write_bin(self.root / "conflicting_version.BIN", values("ardupilot"),
                      version_records=[record]),
        ]
        output = self.root / "existing.yaml"
        meta = output.with_suffix(".meta.json")
        output.write_text("previous profile")
        meta.write_text("previous metadata")
        for path in paths:
            with self.subTest(path=path):
                error = io.StringIO()
                with redirect_stdout(io.StringIO()), redirect_stderr(error):
                    status = capture.main(["--from-file", str(path), "--output", str(output), "--overwrite"])
                self.assertEqual(status, 1)
                self.assertRegex(error.getvalue(), "Cannot identify autopilot|Conflicting firmware")
                self.assertEqual(output.read_text(), "previous profile")
                self.assertEqual(meta.read_text(), "previous metadata")
        with self.assertRaisesRegex(ValueError, "Cannot identify autopilot"):
            capture.read_flight_parameters(paths[0], autopilot="ardupilot")
        with self.assertRaisesRegex(ValueError, "Cannot identify autopilot"):
            capture.read_flight_parameters(paths[2], autopilot="px4")

    def test_identified_firmware_without_release_does_not_invent_a_version(self):
        path = write_ulog(self.root / "no_release.ulg", values("px4"),
                          firmware=dict(sys_name="PX4", ver_sw="abcdef0123456789"))
        profile, report = capture.read_flight_parameters(path, parameter_schema="px4-mc-v1")
        self.assertEqual(profile["px4"]["baseline"], values("px4"))
        self.assertIsNone(report["firmware_identity"]["version"])
        self.assertEqual(report["firmware_identity"]["git_hash"], "abcdef0123456789")

    def test_other_stack_and_unselected_groups_do_not_affect_selected_gains(self):
        baseline = dict(values("px4"), ATC_ANG_RLL_P=float("nan"), MC_ROLLRATE_P=float("nan"))
        path = write_ulog(self.root / "selected.ulg", baseline)
        profile, report = capture.read_flight_parameters(path, groups=["position"])
        expected = {name: baseline[name] for name in GAIN_GROUPS["px4"]["position"]}
        self.assertEqual(profile["px4"]["baseline"], expected)
        self.assertEqual(report["history"]["initial"], expected)
        with self.assertRaisesRegex(ValueError, "MC_ROLLRATE_P: non-finite"):
            capture.read_flight_parameters(path)

    def test_ardupilot_47_names_and_values_remain_native(self):
        from tools.commander.fc_parameter_registry import ARDU47
        logged = {name: (i + 1) / 32 for i, name in enumerate(
            name for names in ARDU47.values() for name in names)}
        path = write_bin(self.root / "new.bin", logged, [(6000, "PSC_D_ACC_P", .25)],
                         firmware_messages=["ArduCopter V4.7.0 (1511f271)"])
        profile, report = capture.read_flight_parameters(path, snapshot="final")
        self.assertEqual(profile["ardupilot"]["baseline"], dict(logged, PSC_D_ACC_P=.25))
        self.assertEqual(profile["ardupilot"]["parameter_schema"], "arducopter-4.7")
        self.assertEqual(profile["ardupilot"]["firmware"]["version"], "4.7.0")
        self.assertEqual(report["conversions"], {})
        self.assertEqual(report["history"]["initial"], logged)
        resolved = resolve_params_sets(profile, seed=42, run_id=0, run_dir=self.root)
        content = (self.root / resolved['ardupilot']['parameter_file']).read_text()
        self.assertIn('PSC_D_ACC_P 0.25', content)
        self.assertNotIn('PSC_ACCZ_', content)

    def test_missing_values_never_use_defaults_and_complete_groups_can_be_selected(self):
        baseline = values("ardupilot")
        del baseline["PSC_VELXY_D"]
        path = write_bin(self.root / "partial.BIN", baseline)
        with self.assertRaisesRegex(ValueError, "missing initial.*PSC_VELXY_D"):
            capture.read_flight_parameters(path)
        profile, _ = capture.read_flight_parameters(path, groups=["attitude", "position", "acceleration"])
        self.assertEqual(len(profile["ardupilot"]["baseline"]), 8)
        for groups in ([], ["position", "position"], ["unsupported"]):
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, "groups"):
                capture.read_flight_parameters(path, groups=groups)

    def test_ulog_late_parameter_is_not_promoted_into_initial_snapshot(self):
        baseline = values("px4")
        del baseline["MPC_XY_P"]
        path = write_ulog(self.root / "late.ulg", baseline, [(3000, "MPC_XY_P", 1.0)])
        with self.assertRaisesRegex(ValueError, "missing initial.*MPC_XY_P"):
            capture.read_flight_parameters(path)
        profile, report = capture.read_flight_parameters(path, snapshot="final")
        self.assertEqual(profile["px4"]["baseline"]["MPC_XY_P"], 1.0)
        self.assertNotIn("MPC_XY_P", report["history"]["initial"])
        self.assertIsNone(report["history"]["changes"][0]["before"])

    def test_context_values_and_changes_are_preserved_without_applying_them(self):
        baseline = values("px4")
        context = {name: 1.0 for name in capture.CONTEXT_PARAMETERS["px4"]}
        path = write_ulog(self.root / "context.ulg", dict(baseline, **context),
                          [(2000, "MC_ROLLRATE_K", 2.0)])
        profile, report = capture.read_flight_parameters(path, snapshot="final")
        self.assertEqual(profile["px4"]["baseline"], baseline)
        self.assertEqual(report["context_parameters"]["MC_ROLLRATE_K"], 2.0)
        self.assertEqual(report["missing_context_parameters"], [])
        self.assertEqual(report["history"]["changes"][0]["parameter"], "MC_ROLLRATE_K")

    def test_firmware_schema_selects_native_names_and_invalid_formats_are_rejected(self):
        path = write_bin(self.root / "ambiguous.BIN", dict(values("ardupilot"), PSC_NE_POS_P=1.0))
        profile, _ = capture.read_flight_parameters(path)
        self.assertEqual(profile["ardupilot"]["baseline"], values("ardupilot"))
        with self.assertRaisesRegex(ValueError, "not px4"):
            capture.read_flight_parameters(path, autopilot="px4")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            capture.read_flight_parameters(path, snapshot="middle")
        for filename, content in (("trajectory.csv", b"x,y,z"), ("eeprom.bin", bytes(128)),
                                  ("empty.ulg", b"")):
            with self.subTest(filename=filename):
                bad = self.root / filename
                bad.write_bytes(content)
                with self.assertRaises(ValueError):
                    capture.read_flight_parameters(bad)

    def test_saved_profiles_round_trip_to_native_files_and_metadata_paths_are_relative(self):
        for ap, suffix, writer in (("ardupilot", ".BIN", write_bin), ("px4", ".ulg", write_ulog)):
            with self.subTest(autopilot=ap):
                source = writer(self.root / (ap + suffix), values(ap))
                original = source.read_bytes()
                output = self.root / "nested outputs" / f"{ap}.yaml"
                profile = capture.capture_flight_parameters(from_file=source, output=output)
                metadata_path = output.with_suffix(".meta.json")
                report = json.loads(metadata_path.read_text())
                entry = report["autopilots"][ap]
                self.assertEqual(report["schema_version"], 2)
                self.assertFalse(Path(entry["source_log"]).is_absolute())
                self.assertEqual((metadata_path.parent / entry["source_log"]).resolve(), source)
                self.assertEqual(report["paths_relative_to"], "metadata_directory")
                self.assertEqual(entry["source_sha256"], hashlib.sha256(original).hexdigest())
                self.assertEqual(entry["firmware_identity"]["autopilot"], ap)
                self.assertEqual(entry["log_format"], "bin" if ap == "ardupilot" else "ulog")
                self.assertEqual(report["profile_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(create_params_sets(output), profile)
                run_dir = self.root / "export" / ap
                resolved = resolve_params_sets(profile, seed=42, run_id=0, run_dir=run_dir)
                validate_parameter_file(resolved[ap], ap, run_dir)
                self.assertEqual(resolved[ap]["values"], values(ap))
                with self.assertRaises(FileExistsError):
                    capture._capture_from_file(source, output)
                capture._capture_from_file(source, output, overwrite=True)

    def test_failed_capture_does_not_replace_existing_outputs(self):
        path = self.root / "bad.BIN"
        output = self.root / "existing.yaml"
        meta = output.with_suffix(".meta.json")
        output.write_text("previous profile")
        meta.write_text("previous metadata")
        for value in (float("nan"), float("inf"), -1.0):
            with self.subTest(value=value):
                write_bin(path, dict(values("ardupilot"), ATC_ANG_RLL_P=value))
                with self.assertRaises(ValueError):
                    capture._capture_from_file(path, output, overwrite=True)
                self.assertEqual(output.read_text(), "previous profile")
                self.assertEqual(meta.read_text(), "previous metadata")

    def test_changing_source_is_rejected_before_publishing(self):
        path = write_bin(self.root / "appending.BIN", values("ardupilot"))
        output = self.root / "never.yaml"
        with patch.object(capture, "sha256_file", side_effect=["before", "after"]):
            with self.assertRaisesRegex(ValueError, "changed during capture"):
                capture._capture_from_file(path, output)
        self.assertFalse(output.exists())
        self.assertFalse(output.with_suffix(".meta.json").exists())


    def test_merge_adds_and_refreshes_one_stack_preserving_other_profile_and_provenance(self):
        ap = write_bin(self.root / "first.BIN", values("ardupilot"))
        px4 = write_ulog(self.root / "px4.ulg", values("px4"), [(2000, "MPC_XY_P", 2.0)])
        output = self.root / "outputs" / "combined.yaml"
        meta = output.with_suffix(".meta.json")
        initial = capture._capture_from_file(ap, output, merge=True)
        ap_info = json.loads(meta.read_text())["autopilots"]["ardupilot"]
        ap.unlink()  # Previous logs are not required to merge an existing capture.
        merged = capture._capture_from_file(px4, output, snapshot="final", merge=True)
        report = json.loads(meta.read_text())
        self.assertEqual(merged["ardupilot"], initial["ardupilot"])
        self.assertEqual(report["autopilots"]["ardupilot"], ap_info)
        self.assertEqual(merged["px4"]["baseline"]["MPC_XY_P"], 2.0)
        self.assertEqual(report["autopilots"]["px4"]["snapshot"], "final")
        px4_info = report["autopilots"]["px4"]
        updated_ap = write_bin(self.root / "second.BIN", values("ardupilot"), [(4000, "PSC_POSXY_P", 3.0)])
        refreshed = capture._capture_from_file(updated_ap, output, snapshot="final", merge=True)
        report = json.loads(meta.read_text())
        self.assertEqual(refreshed["px4"], merged["px4"])
        self.assertEqual(report["autopilots"]["px4"], px4_info)
        self.assertEqual(refreshed["ardupilot"]["baseline"]["PSC_POSXY_P"], 3.0)
        self.assertEqual(report["autopilots"]["ardupilot"]["snapshot"], "final")
        self.assertNotEqual(report["autopilots"]["ardupilot"]["source_sha256"], ap_info["source_sha256"])
        self.assertEqual(report["profile_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(create_params_sets(output), refreshed)
        for ap, entry in report["autopilots"].items():
            self.assertFalse(Path(entry["source_log"]).is_absolute())
            expected = updated_ap if ap == "ardupilot" else px4
            self.assertEqual((meta.parent / entry["source_log"]).resolve(), expected)
        run_dir = self.root / "native"
        resolved = resolve_params_sets(refreshed, seed=42, run_id=0, run_dir=run_dir)
        for ap, config in resolved.items():
            validate_parameter_file(config, ap, run_dir)
        # A subsequent partial capture replaces that stack's whole group selection.
        partial = capture._capture_from_file(updated_ap, output,
            groups=["attitude", "position"], merge=True)
        self.assertEqual(partial["ardupilot"]["groups"], ["attitude", "position"])
        self.assertEqual(partial["px4"], merged["px4"])

    def test_merge_upgrades_legacy_metadata_and_retains_rules_for_the_other_stack(self):
        from tools.scenario.create_params_sets import save_params_sets
        ap = write_bin(self.root / "legacy.BIN", values("ardupilot"))
        px4 = write_ulog(self.root / "next.ulg", values("px4"))
        for absolute in (False, True):
            with self.subTest(legacy_absolute_source=absolute):
                output = self.root / str(absolute) / "legacy.yaml"
                meta = output.with_suffix(".meta.json")
                profiles = capture._capture_from_file(ap, output)
                profiles["ardupilot"]["parameters"] = {
                    "ATC_ANG_RLL_P": dict(operation="scale", value=1.1)}
                save_params_sets(profiles, output)
                report = json.loads(meta.read_text())
                legacy = report["autopilots"]["ardupilot"]
                expected_entry = dict(legacy)
                if absolute:
                    legacy["source_log"] = str(ap)
                legacy.update(paths_relative_to="metadata_directory",
                              profile_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
                meta.write_text(json.dumps(legacy))
                merged = capture._capture_from_file(px4, output, merge=True)
                report = json.loads(meta.read_text())
                self.assertEqual(report["schema_version"], 2)
                self.assertEqual(merged["ardupilot"], profiles["ardupilot"])
                self.assertEqual(report["autopilots"]["ardupilot"], expected_entry)
                self.assertEqual(set(report["autopilots"]), {"ardupilot", "px4"})

    def test_merge_rejects_incomplete_or_inconsistent_pairs_without_writing(self):
        ap = write_bin(self.root / "original.BIN", values("ardupilot"))
        px4 = write_ulog(self.root / "incoming.ulg", values("px4"))
        output = self.root / "existing.yaml"
        meta = output.with_suffix(".meta.json")
        cases = ("missing_yaml", "missing_metadata", "hash_mismatch", "bad_json", "bad_yaml",
                 "wrong_autopilots", "wrong_baseline", "unsupported_schema", "incomplete_entry")
        for case in cases:
            with self.subTest(case=case):
                capture._capture_from_file(ap, output, overwrite=True)
                report = json.loads(meta.read_text())
                if case == "missing_yaml":
                    output.unlink()
                elif case == "missing_metadata":
                    meta.unlink()
                elif case == "hash_mismatch":
                    output.write_bytes(output.read_bytes() + b"# edited\n")
                elif case == "bad_json":
                    meta.write_text("{")
                elif case == "bad_yaml":
                    output.write_text("ardupilot: [")
                else:
                    if case == "wrong_autopilots":
                        report["autopilots"]["px4"] = report["autopilots"].pop("ardupilot")
                    elif case == "wrong_baseline":
                        report["autopilots"]["ardupilot"]["baseline_sha256"] = "0" * 64
                    elif case == "unsupported_schema":
                        report["schema_version"] = 999
                    else:
                        del report["autopilots"]["ardupilot"]["history"]
                    meta.write_text(json.dumps(report))
                before = [path.read_bytes() if path.exists() else None for path in (output, meta)]
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    status = capture.main(["--from-file", str(px4), "--output", str(output), "--merge"])
                self.assertEqual(status, 1)
                after = [path.read_bytes() if path.exists() else None for path in (output, meta)]
                self.assertEqual(after, before)

    def test_failed_incoming_capture_preserves_merge_pair(self):
        ap = write_bin(self.root / "valid.BIN", values("ardupilot"))
        px4_values = values("px4")
        del px4_values["MPC_XY_P"]
        px4 = write_ulog(self.root / "incomplete.ulg", px4_values)
        output = self.root / "existing.yaml"
        meta = output.with_suffix(".meta.json")
        capture._capture_from_file(ap, output)
        before = (output.read_bytes(), meta.read_bytes())
        with self.assertRaisesRegex(ValueError, "missing initial.*MPC_XY_P"):
            capture._capture_from_file(px4, output, merge=True)
        self.assertEqual((output.read_bytes(), meta.read_bytes()), before)

    def test_cli_merge_keeps_independent_snapshots_and_overwrite_still_replaces_all(self):
        ap = write_bin(self.root / "ardu.BIN", values("ardupilot"))
        px4 = write_ulog(self.root / "px4.ulg", values("px4"))
        output = self.root / "both.yaml"
        meta = output.with_suffix(".meta.json")
        capture._capture_from_file(px4, output, snapshot="final")
        text = io.StringIO()
        with patch.object(capture, "DEFAULT_FILE_OUTPUT", output), redirect_stdout(text):
            status = capture.main(["--from-file", str(ap), "--merge"])
        self.assertEqual(status, 0)
        self.assertIn("px4: firmware version=1.15.4, format=ulog", text.getvalue())
        self.assertIn("ardupilot: firmware version=4.6.2, format=bin", text.getvalue())
        self.assertIn("px4: 20 gains, snapshot=final", text.getvalue())
        self.assertIn("ardupilot: 23 gains, snapshot=initial", text.getvalue())
        self.assertEqual(set(create_params_sets(output)), {"ardupilot", "px4"})
        before = (output.read_bytes(), meta.read_bytes())
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            capture._capture_from_file(ap, output, merge=True, overwrite=True)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            capture.main(["--from-file", str(ap), "--output", str(output), "--merge", "--overwrite"])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual((output.read_bytes(), meta.read_bytes()), before)
        capture._capture_from_file(px4, output, overwrite=True)
        self.assertEqual(set(create_params_sets(output)), {"px4"})
        self.assertEqual(set(json.loads(meta.read_text())["autopilots"]), {"px4"})


if __name__ == "__main__":
    unittest.main()
