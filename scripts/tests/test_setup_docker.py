"""Docker lifecycle regression tests; no daemon, image build, or GUI is used.

Run: python3 -m unittest discover -s scripts/tests -v
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

FAKE_DOCKER = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
state_file = Path(os.environ["FAKE_STATE"])
state = json.loads(state_file.read_text())
with open(os.environ["FAKE_CALLS"], "a") as stream:
    stream.write(json.dumps(args) + "\n")

def finish(code=0):
    state_file.write_text(json.dumps(state))
    sys.exit(code)

if args == ["info"]:
    finish(1 if state.get("daemon_down") else 0)
if args[:2] == ["image", "inspect"]:
    finish(0 if state.get("image") else 1)
if args[:2] == ["container", "inspect"]:
    if not state.get("status"):
        finish(1)
    if "--format" in args:
        fmt = args[args.index("--format") + 1]
        if "State.Status" in fmt:
            print(state["status"])
        elif "State.ExitCode" in fmt:
            print(state.get("exit_code", 0))
        elif "State.StartedAt" in fmt:
            print(state.get("started_at", "2026-10-01T15:05:00.000000000Z"))
        else:
            print(state.get("env", ""))
    finish()
if args[:2] == ["container", "start"]:
    state["status"] = "exited" if state.get("startup_error") else "running"
    finish()
if args[:2] == ["container", "stop"]:
    state["status"] = "exited"
    finish()
if args[:2] == ["container", "rm"]:
    state.pop("status", None)
    finish()
if args[0] == "run":
    state["status"] = "exited" if state.get("startup_error") else "running"
    finish()
if args[0] == "build":
    if state.get("build_error"):
        finish(17)
    state["image"] = True
    finish()
if args[0] == "logs":
    if state.get("logs_error"):
        print("Error response from daemon: log driver is unavailable", file=sys.stderr)
        finish(1)
    if "--since" not in args:
        print(state.get("previous_logs", ""))
    print(state.get("logs", ""))
    if state.pop("ready_after_logs", False):
        state["ready"] = True
    if state.pop("exit_after_logs", False):
        state.update(status="exited", exit_code=2)
    finish()
if args[0] == "exec":
    if "test" in args:
        finish(0 if state.get("ready") else 1)
    finish()
raise SystemExit("Unexpected docker command: " + repr(args))
'''


class DockerCommands(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fire docker test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project with spaces"
        (self.project / "scripts").mkdir(parents=True)
        (self.project / "install").mkdir()
        (self.project / ".docker_home").mkdir()
        self.script = self.project / "scripts/setup_docker.sh"
        shutil.copy2(ROOT / "scripts/setup_docker.sh", self.script)
        for name in ("autopilot.sh", "extra.sh"):
            (self.project / "install" / name).write_text("#!/bin/bash\necho fetched\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.executable("docker", FAKE_DOCKER)
        self.state = self.root / "state.json"
        self.calls = self.root / "calls.jsonl"
        self.calls.touch()
        self.set_state(image=True)
        self.env = os.environ.copy()
        for key in ("DISPLAY", "WAYLAND_DISPLAY", "WSL_DISTRO_NAME", "XAUTHORITY",
                    "IMAGE_NAME", "IMAGE_TAG", "CONTAINER_NAME", "BASH_ENV"):
            self.env.pop(key, None)
        self.env.update(PATH=f"{self.bin}:{os.environ['PATH']}",
                        FAKE_STATE=str(self.state), FAKE_CALLS=str(self.calls),
                        TMPDIR=str(self.root))

    def executable(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)
        return path

    def set_state(self, **state):
        self.state.write_text(json.dumps(state))

    def docker_calls(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def run_script(self, *args, code=0):
        result = subprocess.run(["bash", str(self.script), *args], cwd=self.root,
                                env=self.env, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result

    def test_help_and_invalid_options_do_not_contact_docker(self):
        self.run_script()
        self.run_script("help")
        for args in (("unknown",), ("run", "--new-terminal"), ("open", "--recreate"),
                     ("build", "--image"), ("build", "--no-fetch", "--fetch-only")):
            with self.subTest(args=args):
                self.run_script(*args, code=1)
        self.assertEqual(self.docker_calls(), [])

    def test_build_cache_and_custom_identity(self):
        self.run_script("build", "--no-fetch", "--image", "custom", "--tag", "dev")
        build = next(c for c in self.docker_calls() if c[0] == "build")
        self.assertNotIn("--no-cache", build)
        self.assertIn("custom:dev", build)
        self.assertIn(f"UID_USER={os.getuid()}", build)
        self.assertEqual(build[-1], str(self.project))
        self.run_script("build", "--no-fetch", "--no-cache")
        self.assertIn("--no-cache", self.docker_calls()[-1])

    def test_fetch_only_needs_no_daemon(self):
        self.set_state(daemon_down=True)
        result = self.run_script("build", "--fetch-only")
        self.assertEqual(result.stdout.count("fetched"), 2)
        self.assertEqual(self.docker_calls(), [])

    def test_build_and_fetch_failures_propagate(self):
        self.set_state(build_error=True)
        self.run_script("build", "--no-fetch", code=17)
        self.calls.write_text("")
        (self.project / "install/autopilot.sh").write_text("exit 23\n")
        self.run_script("build", code=23)
        self.assertFalse(any(c[0] == "build" for c in self.docker_calls()))

    def test_new_container_is_detached_and_resets_setup_from_any_directory(self):
        sentinel = self.project / ".docker_home/.setup_done"
        sentinel.touch()
        self.run_script("run", "--gpu", "--no-x11")
        run = next(c for c in self.docker_calls() if c[0] == "run")
        self.assertIn("--detach", run)
        self.assertIn("--init", run)
        self.assertIn("--gpus", run)
        self.assertIn(f"{self.project}:/home/user/FIRE_flightstack_sim:rw", run)
        self.assertEqual(run[-3:], ["--", "sleep", "infinity"])
        self.assertFalse(sentinel.exists())
        self.assertFalse(any(c[0] in ("build", "attach", "logs") or "--interactive" in c
                             for c in self.docker_calls()))

    def test_running_and_stopped_containers_are_reused(self):
        sentinel = self.project / ".docker_home/.setup_done"
        sentinel.touch()
        for status in ("running", "exited", "created"):
            with self.subTest(status=status):
                self.calls.write_text("")
                self.set_state(status=status)
                self.run_script("run")
                calls = self.docker_calls()
                self.assertFalse(any(c[0] in ("run", "build", "logs") or "--interactive" in c
                                     or c[:2] == ["container", "rm"] for c in calls))
                self.assertEqual(any(c[:2] == ["container", "start"] for c in calls), status != "running")
                self.assertTrue(sentinel.exists())

    def test_recreate_checks_image_before_removing_container(self):
        self.set_state(status="running", image=False)
        self.run_script("run", "--recreate", code=1)
        self.assertFalse(any(c[:2] == ["container", "rm"] for c in self.docker_calls()))
        self.set_state(status="running", image=True)
        self.run_script("run", "--recreate")
        calls = self.docker_calls()
        removal = next(i for i, c in enumerate(calls) if c[:2] == ["container", "rm"])
        creation = next(i for i, c in enumerate(calls) if c[0] == "run")
        self.assertLess(removal, creation)

    def test_open_requires_running_and_initialized_container(self):
        for state in ({}, {"status": "exited"}, {"status": "running", "ready": False}):
            with self.subTest(state=state):
                self.set_state(**state)
                self.run_script("open", code=1)
        self.set_state(status="running", ready=True)
        self.run_script("open")
        shell = self.docker_calls()[-1]
        self.assertEqual(shell[0], "exec")
        self.assertIn("--interactive", shell)
        self.assertIn("--tty", shell)
        self.assertEqual(shell[shell.index("--user") + 1], "user")
        self.assertEqual(shell[-1], "/bin/bash")

    def test_run_explains_background_initialization_and_ready_state(self):
        result = self.run_script("run", "--container", "custom-dev")
        self.assertIn("returns without waiting", result.stdout)
        self.assertIn("several minutes or longer", result.stdout)
        self.assertIn("docker logs -f custom-dev", result.stdout)
        self.assertIn("Setup completed.", result.stdout)
        self.assertNotIn("[ERROR]", result.stderr)
        self.assertFalse(any(c[0] == "logs" for c in self.docker_calls()))

        self.set_state(status="running", ready=True)
        result = self.run_script("run")
        self.assertIn("Ready to open a shell", result.stdout)
        self.assertNotIn("still in progress", result.stdout)

    def test_open_during_initialization_explains_how_to_follow_logs(self):
        for options in ((), ("--new-terminal",)):
            with self.subTest(options=options):
                self.set_state(status="running", logs="[ENTRYPOINT] Performing one-time setup...")
                result = self.run_script("open", *options, "--container", "custom-dev", code=1)
                self.assertIn("[INFO] Container initialization is still in progress", result.stdout)
                self.assertIn("docker logs -f custom-dev", result.stdout)
                self.assertIn("retry open", result.stdout)
                self.assertNotIn("[ERROR]", result.stderr)
                self.assertFalse(any("--interactive" in c for c in self.docker_calls()))

    def test_open_reports_fatal_entrypoint_failure_with_recent_logs(self):
        failure = "[entrypoint.sh] ERROR line=109 cmd=bash install/autopilot.sh"
        self.set_state(status="running", logs="Build failed\n" + failure)
        result = self.run_script("open", code=1)
        self.assertIn("[ERROR] Container initialization failed", result.stderr)
        self.assertIn(failure, result.stderr)
        self.assertIn("docker logs fire_flightstack_sim", result.stderr)
        self.assertNotIn("still in progress", result.stdout)
        self.assertFalse(any("--interactive" in c for c in self.docker_calls()))

    def test_open_ignores_old_failures_and_nonfatal_dependency_errors(self):
        started_at = "2026-10-01T16:00:00.123456789Z"
        self.set_state(status="running", started_at=started_at,
                       previous_logs="[entrypoint.sh] ERROR line=109 cmd=old-build",
                       logs="ERROR: unresolved rosdep keys\nContinuing build...")
        result = self.run_script("open", code=1)
        self.assertIn("still in progress", result.stdout)
        self.assertNotIn("[ERROR]", result.stderr)
        log_calls = [c for c in self.docker_calls() if c[0] == "logs"]
        self.assertEqual(log_calls, [["logs", "--since", started_at, "--tail", "80",
                                     "fire_flightstack_sim"]])

    def test_open_connects_if_initialization_finishes_while_reading_logs(self):
        self.set_state(status="running", ready_after_logs=True,
                       logs="[ENTRYPOINT] Setup completed.")
        result = self.run_script("open")
        self.assertIn("--interactive", self.docker_calls()[-1])
        self.assertNotIn("still in progress", result.stdout)

    def test_open_detects_exit_while_reading_logs(self):
        self.set_state(status="running", exit_after_logs=True, logs="Build failed")
        result = self.run_script("open", code=1)
        self.assertIn("exited with code 2", result.stderr)
        self.assertIn("Build failed", result.stderr)
        self.assertIn("[ERROR] Container startup failed", result.stderr)
        self.assertNotIn("still in progress", result.stdout)
        self.assertFalse(any("--interactive" in c for c in self.docker_calls()))

    def test_open_reports_log_read_failure_without_claiming_setup_failed(self):
        self.set_state(status="running", logs_error=True)
        result = self.run_script("open", code=1)
        self.assertIn("[ERROR] Could not read initialization logs", result.stderr)
        self.assertIn("log driver is unavailable", result.stderr)
        self.assertNotIn("Container initialization failed", result.stderr)
        self.assertNotIn("still in progress", result.stdout)

    def test_ready_container_opens_without_reading_old_logs(self):
        self.set_state(status="running", ready=True,
                       previous_logs="[entrypoint.sh] ERROR line=109 cmd=old-build")
        self.run_script("open")
        self.assertIn("--interactive", self.docker_calls()[-1])
        self.assertFalse(any(c[0] == "logs" for c in self.docker_calls()))

    def test_stop_remove_and_missing_container_are_idempotent(self):
        self.set_state(status="running", image=True)
        self.run_script("stop")
        self.run_script("stop")
        self.run_script("remove")
        self.run_script("remove")
        self.run_script("stop")
        calls = self.docker_calls()
        self.assertEqual(sum(c[:2] == ["container", "stop"] for c in calls), 1)
        self.assertEqual(sum(c[:2] == ["container", "rm"] for c in calls), 1)
        self.assertTrue(json.loads(self.state.read_text())["image"])

    def test_sourcing_preserves_caller_and_returns_failure(self):
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
source "$1" help >/dev/null || exit 10
source "$1" invalid >/dev/null 2>&1
[[ $? == 1 ]] || exit 11
[[ $(set +o) == "$before_options" ]] || exit 12
[[ $(trap -p) == "$before_traps" ]] || exit 13
[[ $(declare -f) == "$before_functions" ]] || exit 14
[[ $(declare -p PROJECT_ROOT) == "$before_vars" ]] || exit 15
[[ $PWD == "$before_pwd" ]] || exit 16
[[ ! -v SCRIPT_PATH && ! -v COMMAND ]] || exit 17
'''
        result = subprocess.run(["bash", "-c", command, "test", str(self.script)],
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.set_state(build_error=True)
        result = subprocess.run(["bash", "-c", 'if source "$1" build --no-fetch; then exit 99; else exit "$?"; fi',
                                 "test", str(self.script)], env=self.env, capture_output=True)
        self.assertEqual(result.returncode, 17)

    def test_linux_and_wsl_terminal_argument_boundaries(self):
        capture = self.root / "terminal.json"
        launcher = '#!/usr/bin/env python3\nimport json, os, sys\nfrom pathlib import Path\nPath(os.environ["TERMINAL_CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
        self.executable("x-terminal-emulator", launcher)
        self.executable("wt.exe", launcher)
        self.env.update(DISPLAY=":0", TERMINAL_CAPTURE=str(capture))
        self.set_state(status="running", ready=True)
        self.run_script("open", "--new-terminal", "--container", "custom-dev")
        args = json.loads(capture.read_text())
        self.assertEqual(args, ["-e", "bash", str(self.script), "open", "--container", "custom-dev"])
        self.env["WSL_DISTRO_NAME"] = "Ubuntu Test"
        self.run_script("open", "--new-terminal")
        args = json.loads(capture.read_text())
        self.assertEqual(args[:8], ["-w", "0", "new-tab", "wsl.exe", "--distribution", "Ubuntu Test", "--exec", "bash"])
        self.assertIn(str(self.script), args)

    def test_headless_and_failed_terminal_launch_report_errors(self):
        self.set_state(status="running", ready=True)
        self.run_script("open", "--new-terminal", code=1)
        self.env["DISPLAY"] = ":0"
        self.executable("x-terminal-emulator", "#!/bin/bash\nexit 4\n")
        result = self.run_script("open", "--new-terminal", code=1)
        self.assertIn("Terminal launch failed", result.stderr)

    def test_x11_permission_outlives_run_and_headless_avoids_xhost(self):
        self.executable("xhost", '#!/bin/bash\nprintf "%s\\n" "$@" >> "$XHOST_CALLS"\n')
        xhost_calls = self.root / "xhost.txt"
        self.env["XHOST_CALLS"] = str(xhost_calls)
        self.run_script("run")
        self.assertFalse(xhost_calls.exists())
        self.set_state(image=True)
        self.env["DISPLAY"] = ":99"
        self.run_script("run")
        self.assertEqual(xhost_calls.read_text().splitlines(), [f"+si:localuser:{os.environ['USER']}"])
        run = next(c for c in self.docker_calls() if c[0] == "run" and "DISPLAY=:99" in c)
        self.assertIn("DISPLAY=:99", run)
        self.set_state(status="exited", env="DISPLAY=:99")
        self.run_script("run")
        self.assertEqual(len(xhost_calls.read_text().splitlines()), 2)

    def test_container_build_mounts_preserve_native_outputs(self):
        roots = ("ros2/px4_ros_uxrce_dds_ws", "ros2/px4_msgs_ws", "gz/harmonic_ws")
        paths = ["build", "gz/env", "gz/ardupilot_gazebo/build", "ap/px4/build", "ap/ardupilot/build"]
        for root in roots:
            (self.project / root / "src").mkdir(parents=True)
            paths.extend(f"{root}/{output}" for output in ("build", "install", "log"))
        for relative in paths:
            path = self.project / relative
            path.mkdir(parents=True, exist_ok=True)
            (path / "native-output").write_text("keep host build")
        self.run_script("run")
        run = next(c for c in self.docker_calls() if c[0] == "run")
        for relative in paths:
            with self.subTest(relative=relative):
                isolated = self.project / ".docker_home/artifacts/fire_flightstack_sim" / relative
                self.assertIn(f"{isolated}:/home/user/FIRE_flightstack_sim/{relative}:rw", run)
                self.assertEqual(list(isolated.iterdir()), [])
                self.assertEqual((self.project / relative / "native-output").read_text(), "keep host build")
        self.set_state(image=True)
        self.run_script("run", "--container", "another-dev")
        custom_run = [c for c in self.docker_calls() if c[0] == "run"][-1]
        self.assertTrue(any(".docker_home/artifacts/another-dev/" in arg for arg in custom_run))
        self.run_script("run", "--container", "../invalid", code=1)

    def test_failed_start_and_open_report_exit_code_and_logs(self):
        for command in ("run", "open"):
            with self.subTest(command=command):
                self.set_state(status="exited", image=True, startup_error=True,
                               exit_code=2, logs="CMakeCache.txt belongs to a different directory")
                result = self.run_script(command, code=1)
                self.assertIn("exited with code 2", result.stderr)
                self.assertIn("CMakeCache.txt belongs to a different directory", result.stderr)
                self.assertIn("docker logs fire_flightstack_sim", result.stderr)

    def test_entrypoint_preserves_command_arguments_and_clears_stale_readiness(self):
        entrypoint = self.project / "install/entrypoint.sh"
        marker = self.root / "ready"
        script = (ROOT / "install/entrypoint.sh").read_text()
        script = script.replace('WORKSPACE="/home/${USER_NAME}/FIRE_flightstack_sim"',
                                f'WORKSPACE="{self.project}"')
        script = script.replace('READY_FILE="/tmp/fire_flightstack_sim.ready"',
                                f'READY_FILE="{marker}"')
        entrypoint.write_text(script)
        self.executable("id", '#!/bin/bash\nif [[ $# == 1 ]]; then echo 0; else echo 1000; fi\n')
        self.executable("sudo", '#!/bin/bash\nshift 3\nexec "$@"\n')
        capture = self.root / "capture.py"
        capture.write_text('import json, sys\nprint(json.dumps(sys.argv[1:]))\n')
        env = dict(self.env, HOST_UID="1000", HOST_GID="1000")
        arguments = ["two words", "$(touch should-not-exist)", 'double"quote', "single'quote", ""]
        result = subprocess.run(["bash", str(entrypoint), "--skip-setup", "--",
                                 shutil.which("python3"), str(capture), *arguments],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout.splitlines()[-1]), arguments)
        self.assertTrue(marker.exists())
        env["HOST_UID"] = ""
        result = subprocess.run(["bash", str(entrypoint), "--skip-setup"],
                                env=env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
