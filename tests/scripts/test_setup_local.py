"""Local setup dispatch tests without installing packages or changing the host.

Run from the project root:
    python3 -B -m unittest discover -s tests/scripts -v
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

FAKE_HOST_COMMAND = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
state_file = Path(os.environ["LOCAL_TEST_STATE"])
state = json.loads(state_file.read_text())
if name == "id":
    print("test-user" if args == ["-un"] else " ".join(state["memberships"]))
    sys.exit(0)
if name == "getent":
    lookup = args[1]
    for group, gid in state["groups"].items():
        if lookup == group or lookup == str(gid):
            print(f"{group}:x:{gid}:")
            sys.exit(0)
    sys.exit(2)
with open(os.environ["LOCAL_TEST_CALLS"], "a") as stream:
    stream.write(json.dumps({"name": name, "args": args}) + "\n")
if args and args[0] == "groupadd":
    gid = args[args.index("--gid") + 1] if "--gid" in args else 999 - len(state["groups"])
    state["groups"][args[-1]] = int(gid)
if args and args[0] == "usermod":
    state["memberships"].extend(args[args.index("-G") + 1].split(","))
state_file.write_text(json.dumps(state))
'''

FAKE_HELPER = r'''#!/usr/bin/env bash
python3 - "$0" "$@" <<'PY'
import json
import os
from pathlib import Path
import sys
name = Path(sys.argv[1]).stem
args = sys.argv[2:]
with open(os.environ["LOCAL_TEST_CALLS"], "a") as stream:
    stream.write(json.dumps({"name": name, "args": args, "cwd": os.getcwd()}) + "\n")
phase = args[args.index("--phase") + 1] if "--phase" in args else ""
if name == os.environ.get("LOCAL_FAIL_HELPER") and phase == os.environ.get("LOCAL_FAIL_PHASE", ""):
    sys.exit(23)
PY
'''


class LocalCommands(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fire local test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project with spaces"
        (self.project / "scripts").mkdir(parents=True)
        (self.project / "install").mkdir()
        self.script = self.project / "scripts/setup_local.sh"
        shutil.copy2(ROOT / "scripts/setup_local.sh", self.script)
        self.helper_names = ("base", "ros2", "gazebo", "autopilot", "extra", "usersetup", "clean")
        for name in self.helper_names:
            path = self.project / "install" / f"{name}.sh"
            path.write_text(FAKE_HELPER)
            path.chmod(0o644)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("sudo", "getent", "id", "rosdep"):
            path = self.bin / name
            path.write_text(FAKE_HOST_COMMAND)
            path.chmod(0o755)
        self.calls = self.root / "calls.jsonl"
        self.calls.touch()
        self.state = self.root / "state.json"
        self.set_state(groups={"sudo": 27, "plugdev": 46, "dialout": 20,
                               "input": 107, "render": 110, "video": 44},
                       memberships=["test-user"])
        self.env = os.environ.copy()
        for key in ("GID_INPUT", "GID_RENDER", "BASH_ENV", "LOCAL_FAIL_HELPER", "LOCAL_FAIL_PHASE"):
            self.env.pop(key, None)
        self.env.update(PATH=f"{self.bin}:{os.environ['PATH']}",
                        LOCAL_TEST_STATE=str(self.state), LOCAL_TEST_CALLS=str(self.calls),
                        TMPDIR=str(self.root))

    def set_state(self, **state):
        self.state.write_text(json.dumps(state))

    def events(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def phases(self):
        result = []
        for event in self.events():
            if event["name"] not in self.helper_names:
                continue
            args = event["args"]
            phase = args[args.index("--phase") + 1] if "--phase" in args else ""
            result.append((event["name"], phase))
        return result

    def run_script(self, *args, code=0):
        result = subprocess.run(["bash", str(self.script), *args], cwd=self.root,
                                env=self.env, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result

    def test_help_and_invalid_arguments_do_not_change_host(self):
        for args in ((), ("help",), ("--help",), ("build", "--help")):
            self.run_script(*args)
        for args in (("open",), ("run",), ("stop",), ("remove",),
                     ("build", "--new-terminal"), ("build", "--no-fetch"),
                     ("env", "--upgrade"), ("install", "--unknown")):
            with self.subTest(args=args):
                self.run_script(*args, code=1)
        self.assertEqual(self.events(), [])

    def test_build_only_calls_prepared_source_build_from_project_root(self):
        (self.project / "install/base.sh").unlink()
        self.run_script("build")
        self.assertEqual(self.phases(), [("autopilot", "build")])
        event = self.events()[0]
        self.assertEqual(event["cwd"], str(self.project))
        self.assertIn("--with-ardupilot", event["args"])
        self.assertEqual(event["args"][event["args"].index("--project-root") + 1], str(self.project))

    def test_fetch_only_prepares_sources_and_artifacts(self):
        self.run_script("fetch")
        self.assertEqual(self.phases(), [("autopilot", "fetch"), ("extra", "fetch")])
        self.assertFalse(any(e["name"] == "sudo" for e in self.events()))

    def test_env_only_dispatches_environment_helpers(self):
        self.run_script("env")
        self.assertEqual(self.phases(), [("gazebo", "env"), ("autopilot", "env"),
                                        ("extra", "env"), ("usersetup", "")])

    def test_dependencies_install_binary_gazebo_without_building_sources(self):
        self.run_script("deps")
        self.assertEqual(self.phases(), [("base", ""), ("ros2", ""),
                                        ("gazebo", "deps"), ("gazebo", "build"),
                                        ("autopilot", "deps"), ("extra", "deps")])
        for event in self.events():
            if event["name"] == "gazebo":
                self.assertEqual(event["args"][event["args"].index("--install") + 1], "binary")
                self.assertEqual(event["args"][event["args"].index("--ros-gz") + 1], "binary")
        sudo_commands = [e["args"] for e in self.events() if e["name"] == "sudo"]
        self.assertFalse(any("upgrade" in args or "autoremove" in args or args[0] in ("chown", "chmod") for args in sudo_commands))
        for name in self.helper_names:
            self.assertEqual((self.project / "install" / f"{name}.sh").stat().st_mode & 0o777, 0o644)

    def test_install_orders_all_phases_and_no_fetch_is_scoped(self):
        self.run_script("install")
        phases = self.phases()
        self.assertEqual(phases, [("base", ""), ("ros2", ""), ("gazebo", "deps"),
                                 ("gazebo", "build"), ("autopilot", "deps"), ("extra", "deps"),
                                 ("autopilot", "fetch"), ("extra", "fetch"), ("autopilot", "build"),
                                 ("gazebo", "env"), ("autopilot", "env"), ("extra", "env"),
                                 ("usersetup", "")])
        self.assertNotIn(("clean", ""), phases)
        self.calls.write_text("")
        self.run_script("install", "--no-fetch")
        self.assertEqual(self.phases(), [phase for phase in phases if phase[1] != "fetch"])

    def test_upgrade_is_explicit_and_debug_reaches_helpers(self):
        self.run_script("deps", "--upgrade", "--debug")
        self.assertEqual(sum(e["name"] == "sudo" and "upgrade" in e["args"] for e in self.events()), 1)
        self.assertTrue(all("--debug" in e["args"] for e in self.events() if e["name"] in self.helper_names))

    def test_missing_groups_get_os_ids_and_existing_memberships_are_preserved(self):
        self.set_state(groups={"sudo": 27, "plugdev": 46, "dialout": 20, "video": 44},
                       memberships=["test-user", "sudo", "plugdev", "dialout", "video"])
        self.run_script("deps")
        groupadd = [e["args"] for e in self.events() if e["name"] == "sudo" and e["args"][0] == "groupadd"]
        self.assertEqual(groupadd, [["groupadd", "--system", "input"], ["groupadd", "--system", "render"]])
        usermod = [e["args"] for e in self.events() if e["name"] == "sudo" and e["args"][0] == "usermod"]
        self.assertEqual(usermod, [["usermod", "-a", "-G", "input,render", "test-user"]])
        self.calls.write_text("")
        self.run_script("deps")
        self.assertFalse(any(e["name"] == "sudo" and e["args"][0] in ("groupadd", "usermod") for e in self.events()))

    def test_gid_override_validation_precedes_host_changes(self):
        self.set_state(groups={"other-device": 110}, memberships=["test-user"])
        for value in ("invalid", "110"):
            self.env["GID_INPUT"] = value
            self.run_script("deps", code=1)
            self.assertEqual(self.events(), [])
        self.env["GID_INPUT"] = "2001"
        self.run_script("deps")
        self.assertTrue(any(e["args"] == ["groupadd", "--system", "--gid", "2001", "input"] for e in self.events()))

    def test_missing_helpers_fail_before_installation(self):
        (self.project / "install/usersetup.sh").unlink()
        self.run_script("install", code=1)
        self.assertEqual(self.events(), [])

    def test_helper_failure_stops_later_phases(self):
        self.env.update(LOCAL_FAIL_HELPER="autopilot", LOCAL_FAIL_PHASE="fetch")
        self.run_script("install", code=23)
        self.assertEqual(self.phases()[-1], ("autopilot", "fetch"))
        self.assertNotIn(("autopilot", "build"), self.phases())
        self.assertNotIn(("usersetup", ""), self.phases())

    def test_sourcing_preserves_shell_state_and_propagates_failures(self):
        command = r'''
set +e
set +u
set +o pipefail
trap ':' EXIT
PROJECT_ROOT=caller_value
log() { :; }
before_options=$(set +o)
before_traps=$(trap -p)
before_functions=$(declare -f)
before_vars=$(declare -p PROJECT_ROOT)
before_pwd=$PWD
source "$1" build >/dev/null || exit 10
source "$1" invalid >/dev/null 2>&1
[[ $? == 1 ]] || exit 11
[[ $(set +o) == "$before_options" ]] || exit 12
[[ $(trap -p) == "$before_traps" ]] || exit 13
[[ $(declare -f) == "$before_functions" ]] || exit 14
[[ $(declare -p PROJECT_ROOT) == "$before_vars" ]] || exit 15
[[ $PWD == "$before_pwd" ]] || exit 16
[[ ! -v THIS_DIR && ! -v COMMAND ]] || exit 17
'''
        result = subprocess.run(["bash", "-c", command, "test", str(self.script)],
                                cwd=self.root, env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.env.update(LOCAL_FAIL_HELPER="autopilot", LOCAL_FAIL_PHASE="build")
        result = subprocess.run(["bash", "-c", 'if source "$1" build; then exit 99; else exit "$?"; fi',
                                 "test", str(self.script)], env=self.env, capture_output=True)
        self.assertEqual(result.returncode, 23)

    def test_repeated_ardupilot_env_preserves_user_content_without_new_duplicates(self):
        env_file = self.root / "ardupilot-env"
        actual = (ROOT / "install/autopilot.sh").read_text()
        actual = actual.replace('local env_file="${HOME}/.ardupilot_env"',
                                'local env_file="${LOCAL_TEST_AP_ENV}"')
        (self.project / "install/autopilot.sh").write_text(actual)
        self.env["LOCAL_TEST_AP_ENV"] = str(env_file)
        custom = '# User settings\nexport CUSTOM_SETTING=keep\n'
        env_file.write_text(custom)
        for _ in range(2):
            result = subprocess.run(["bash", str(self.project / "install/autopilot.sh"),
                                     "--phase", "env", "--no-px4", "--with-ardupilot",
                                     "--project-root", str(self.project)],
                                    env=self.env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            contents = env_file.read_text()
            self.assertTrue(contents.startswith(custom))
            self.assertEqual(contents.count('PATH="$HOME/.local/bin:$PATH"'), 1)
        # The legacy unmarked block should be recognized, not appended again.
        before = env_file.read_bytes()
        result = subprocess.run(["bash", str(self.project / "install/autopilot.sh"),
                                 "--phase", "env", "--no-px4", "--with-ardupilot"],
                                env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(env_file.read_bytes(), before)

    def test_repeated_usersetup_keeps_one_managed_bashrc_block(self):
        bashrc = self.root / "test-bashrc"
        bashrc.write_text('# User aliases\nalias keep_me=true\n')
        script = self.project / "install/usersetup.sh"
        script.write_text((ROOT / "install/usersetup.sh").read_text().replace(
            'BASHRC="/home/${USER}/.bashrc"', 'BASHRC="${LOCAL_TEST_BASHRC}"'))
        self.env["LOCAL_TEST_BASHRC"] = str(bashrc)
        contents = None
        for _ in range(2):
            result = subprocess.run(["bash", str(script), "--project-root", str(self.project)],
                                    env=self.env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            updated = bashrc.read_text()
            self.assertEqual(updated.count('# >>> flightstack_sim usersetup >>>'), 1)
            self.assertIn('alias keep_me=true', updated)
            if contents is not None:
                self.assertEqual(updated, contents)
            contents = updated


if __name__ == "__main__":
    unittest.main()
