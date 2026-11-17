# FIRE Flightstack Simulator
A unified simulation environment for **PX4, ArduPilot, and Gazebo (SITL)**, designed for **Motion-based Tactical Identification of Firmware (MOTIF)**.
This repository provides a **reproducible, containerized development environment** for:

- Multi-autopilot simulation (PX4 + ArduPilot) hosed in Gazebo (Harmonic) simulator
- Integration with mavlink-based custom launcher/commander and analysis pipelines

---

Python tools use explicit package imports, for example
`from tools.commander.pymavlink_px4_commander import PX4MissionRunner`.
Run commands from the repository root with `python3 -m tools.<package>.<module>`;
no editable install or `PYTHONPATH` override is required. The command-line tools
also retain direct file execution (`python3 tools/.../script.py`) by adding only
the repository root to the import path. Package `__init__.py` files do not start
simulators or import optional tool dependencies.

## PX4 standalone Gazebo with FIRE PX4Vision

Run these commands from the repository root. The standalone and batch launchers
automatically run `make px4_sitl` in the selected PX4 directory if the binary or
startup script is missing, in both Docker and local environments. Existing build
outputs are reused; after changing firmware, rebuild explicitly:

```bash
make -C ap/px4 px4_sitl
```

The launchers build before configuring Gazebo resource paths, PX4 sensor plugins,
and the server configuration. A build failure stops the batch without flight retries.

Start Gazebo and spawn `fire_px4vision_0` in terminal 1:

```bash
python3 tools/launcher/px4_gz_standalone.py gazebo --headless
```

Omit `--headless` for the Gazebo GUI. The launcher starts the server, confirms
vehicle creation, then opens the GUI. This prevents the Gazebo 8 GUI from missing
the vehicle's creation update while loading its initial state.
After `[READY]`, attach PX4 in terminal 2:

```bash
python3 tools/launcher/px4_gz_standalone.py px4
```

Defaults are `--world default_fire --model fire_px4vision --instance 0` and
PX4 airframe `--vehicle 4006`. Both commands must use the same `--partition`
(default: `$GZ_PARTITION`, or `fire_px4_standalone`) and `--instance`.
The Gazebo command accepts `--pose X Y Z ROLL PITCH YAW` (metres/radians).
PX4 uses `PX4_GZ_STANDALONE=1` and attaches to the existing entity; it does not
spawn another vehicle. Airframe 4006 provides the PX4 control parameters.

Connect QGroundControl, or start a headless GCS heartbeat in terminal 3:

```bash
python3 -m tools.commander.fake_gcs_heartbeat --connect udp:127.0.0.1:14550
```

Parameters and ULogs are stored under `logs/px4_standalone/0`; use `--work-dir`
to select a different directory. Stop PX4 first, then Gazebo, with Ctrl+C.
`default_fire.sdf` is the empty FIRE environment. `default_fire_px4vision.sdf`
contains the ArduPilot interface and is not a PX4 world.

For automated missions, the batch launcher starts Gazebo, spawns the model,
attaches PX4, runs the mission, and stops its own processes. Do not start the
manual commands above at the same time. Generate a short example scenario and run:

```bash
export FLIGHTSTACK_SIM_ROOT="$PWD"
python3 -m tools.scenario.scenario_generator planar_n_pts \
  --outdir data/px4_fire_standalone --runs 1 \
  --px4-vehicle 4006 --px4-frame gz_fire_px4vision --px4-world default_fire \
  --n 2 --edge-m 2 --vertex-deg 0 --alt-m 3 --speed-m-s 1 \
  --landing-alt-m 2 --land true
python3 -m tools.launcher.run_px4_gz_sitl \
  --run-root data/px4_fire_standalone --headless \
  --startup-delay-s 2 --max-run-s 120 --max-retries 0
```

For existing scenarios, set `autopilots.px4.sim.vehicle: 4006`,
`frame: gz_fire_px4vision`, and `world: default_fire`. Logs are collected in
`run_XXX/px4_logs/`, with PX4 runtime state in its `rootfs/` subdirectory.
The batch launcher uses a separate Gazebo partition for each attempt.
With the GUI enabled, `gazebo.log` contains server output and `gazebo_gui.log`
contains GUI output; both processes use the same resource paths and partition.
Both launchers accept `--inter-run-delay SECONDS` (default: `0`).
`scripts/run_sitl_batch.sh` sets it to `3` for both autopilots. The delay applies
before every actual attempt, including the first attempt, retries, and
transitions to the next scenario. For subsequent attempts, the previous attempt
has shut down and its logs have been collected or archived before the wait.
There is no delay after the last attempt or for skipped scenarios. Ctrl+C cancels
the wait. This is separate from `--startup-delay-s`, which controls startup waits.

Both batch launchers apply `common.scenario.wind_m_s: [east, north, up]` in m/s
before starting the mission by default (`wind_policy.mode: continuous`). The selected world must load `WindEffects`; the FIRE
defaults provide it through PX4's `server.config` or the ArduPilot world SDF.
Each attempt publishes the target using Gazebo's environment and partition,
including an explicit `[0, 0, 0]`, and waits up to 30 seconds for `wind_info` to
acknowledge it. Failure stops the attempt before the mission starts. A missing
`wind_m_s` field preserves the world's wind settings; invalid or nonfinite vectors
are rejected when loading the scenario. `px4_logs/wind.json` or `ardu_logs/wind.json`
records the requested and acknowledged target in ENU, and is archived with failed
attempts. This acknowledgement is not a measurement of the filtered wind velocity.

For calm takeoff/landing with wind only during the airborne mission, add:

```yaml
common:
  scenario:
    wind_m_s: [2.0, 1.0, 0.5]  # ENU, m/s; use [0, 0, 0] for the calm control
    wind_policy:
      mode: mission_only
      ramp_s: 3.0
      min_settle_s: 5.0
      settle_timeout_s: 30.0
```

The scenario generator accepts `--wind-policy mission-only`, `--wind-ramp-s`,
`--wind-min-settle-s`, and `--wind-settle-timeout-s`. Existing YAML without a policy
retains continuous wind. Mission-only runs require one initial TAKEOFF, airborne
waypoints, and one final LAND. The original waypoint geometry/speeds are retained;
indefinite holds are inserted after TAKEOFF and before LAND (at the final airborne
waypoint). Both autopilots use these holds, including zero-wind control runs.

The launcher verifies actual calm wind before starting the commander. At the
first hold the vehicle settles, wind ramps up, and the vehicle settles again
before the measurement segment starts. At the final hold the segment ends, wind
ramps to zero, and landing is released only after calm wind and stable flight.
Hold arrival uses position, altitude, and velocity telemetry; selecting a mission
item alone does not count as arrival. Windy hover may have nonzero equilibrium
tilt. Calm holds additionally require tilt below 5 degrees. Other defaults are
1.5 m horizontal error, 0.5 m altitude error, 0.3 m/s speed, and 0.1 m/s wind error;
these can be set as `position_tolerance_m`, `altitude_tolerance_m`,
`speed_tolerance_m_s`, `wind_tolerance_m_s`, and `calm_tilt_deg` in `wind_policy`.
The checkpoint timeout includes arrival, ramping, and settling (the takeoff
checkpoint settles both before and after ramping). Ramp/settling durations use
simulation time; process/telemetry failures and a
stalled clock fail the attempt. Allow extra time in `--max-run-s` (e.g. 300).

Mission-only mode and dynamic wind models require Gazebo Harmonic Python bindings (`gz.transport13`,
`gz.msgs10`) in the launcher's Python environment and the world's SceneBroadcaster
state service. A private worker inherits the server environment/partition and
checks the Wind entity's `WorldLinearVelocity`, in addition to target acknowledgement.
Aerodynamic plugins remain enabled throughout the flight.

Each attempt also records `wind_events.jsonl` (phase boundaries, simulation time,
MAVLink boot time and telemetry age), `wind_samples.jsonl` (actual/target wind), and
`mission_sequence_map.json` (uploaded sequence to original scenario command index).
These files are archived with failed attempts. Event boot times use the latest
GLOBAL_POSITION_INT sample, so boundary precision is limited by telemetry latency.

Time-varying wind is the sum of the ENU mean `wind_m_s`, a finite
1-minus-cosine gust, and three-axis Dryden turbulence. For example, append these
options to a `scenario_generator.py planar_n_pts` or `three_d_n_pts` command:

```bash
--wind-policy mission-only \
--gust-magnitude-m-s 3 --gust-start-s 2 --gust-duration-s 4 \
--gust-period-s 10 --gust-azimuth-deg 90 --gust-elevation-deg 0 \
--turbulence-magnitude-m-s 0.5 --turbulence-length-scale-m 30 30 10 \
--turbulence-reference-speed-m-s 5 --turbulence-seed 42
```

| Option | Default | Meaning |
|---|---|---|
| `--gust-magnitude-m-s` | `0` | Peak added gust speed; zero disables gusts |
| `--gust-start-s` | `0` | Delay after disturbance activation, in simulation seconds |
| `--gust-duration-s` | `5` | Full pulse duration; the peak occurs halfway through |
| `--gust-period-s` | `0` | Start-to-start repeat interval; zero gives one pulse; otherwise at least the duration |
| `--gust-azimuth-deg` | `0` | Direction toward which the gust blows: North=0, East=90 |
| `--gust-elevation-deg` | `0` | Angle above horizontal, from -90 (down) to +90 (up) |
| `--turbulence-magnitude-m-s` | `0` | Standard deviation sigma of **each** Dryden u/v/w component; zero disables turbulence |
| `--turbulence-length-scale-m LU LV LW` | `30 30 10` | Longitudinal, lateral, vertical scale lengths in metres |
| `--turbulence-reference-speed-m-s` | `5` | Fixed advection speed V, giving time scales L/V; independent of measured vehicle speed |
| `--turbulence-azimuth-deg` | `0` | Longitudinal u-axis toward North=0, East=90; v is horizontal right and w is up |
| `--turbulence-sample-period-s` | `0.1` | Dryden filter interval in simulation seconds |
| `--turbulence-seed` | derived per run | Noise seed; an explicit seed repeats the same noise realization for equal settings |

Both magnitude options also accept `random`, with
`--gust-magnitude-m-s-range MIN MAX` (default `0 5`) and
`--turbulence-magnitude-m-s-range MIN MAX` (default `0 2`). These sample an
amplitude **once per run**. Separate random streams preserve mission geometry,
mean wind, and vehicle parameters. The dataset seed and run ID reproduce the
resolved amplitudes and turbulence seed in `scenario.yaml`; retries and both
autopilots read those same values.

Settings are stored as `common.scenario.gust` and `common.scenario.turbulence`
mappings; CLI suffixes become keys with underscores, e.g. `gust.magnitude_m_s`
and `turbulence.length_scale_m: [30, 30, 10]`. Missing mappings disable the models.
A dynamic-only YAML without `wind_m_s` uses zero mean wind. Dataset
`metadata.yaml` and each attempt's `wind.json` also record the settings.

In `mission_only`, disturbances activate at `mission_segment_start`, after the
mean-wind settling hold. They switch off at `mission_segment_end` before the
mean wind ramps down; landing still requires measured calm wind and stable flight.
In `continuous`, activation occurs when the wind controller initializes before
the commander starts, so a single gust can finish before takeoff; set
`--gust-start-s` or `--gust-period-s` as needed. Both use simulation time.
During active disturbances, monitoring checks command acknowledgement (allowing
one publication of transport latency); the constant-wind tolerance applies to
settling holds. Actual filtered wind remains recorded throughout.

The gust follows `A/2 * (1 - cos(2*pi*t/T))` for `0 <= t <= T`, and is zero
outside the pulse, equivalent to the spatial profile with `T=2H/V` described in
[FAA AC 25.341-1](https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_25_341-1.pdf).
Dryden uses the longitudinal first-order and lateral/vertical second-order
[MIL-F-8785C velocity spectra](https://www.mathworks.com/help/aeroblks/drydenwindturbulencemodelcontinuous.html),
with exact Gaussian state updates and stationary initialization. This implements
translational wind with user-selected fixed L, V, and equal component sigma;
it does not calculate altitude-dependent parameters or angular-rate gusts.
The vector RMS is `sqrt(3)*sigma`; sigma is not a peak bound.

`wind_samples.jsonl` separates `base`, `gust`, `turbulence`, summed `target`,
`acknowledged`, and measured `actual`, with `command_sim_s` recording the model
sample time. The worker publishes at approximately 10 Hz wall time; faster
simulation or smaller filter intervals can skip publications, although internal
Dryden ticks remain reproducible. Gazebo WindEffects can further filter the
published signal, so requested and measured spectra need not be identical.

Both plotting CLIs accept `--segment mission` to exclude takeoff, settling and
landing. This requires both recorded segment boundaries and raw ULog/BIN boot
timestamps (or CSV `timestamp`/`TimeUS`/`time_us` columns in microseconds). All
signals are cropped before alignment and share the recorded segment-start origin.
Missing/incomplete boundaries are reported instead of guessing from altitude.

```bash
python -m tools.plotting.plot_sitl_run_comparison \
  --run-dirs data/sitl_wind_test/run_000 data/sitl_wind_test/run_001 \
  --autopilot ardupilot --segment mission \
  --output-dir results/plotting/sitl_wind_test/ardu_mission_compare
```


Both batch launchers interpret `--max-retries 0` as one attempt. Successful
flights remain in `px4_logs/` or `ardu_logs/`, with their outcome recorded in
`result.json`. A first-attempt success creates no attempts directory.

Failed, timed-out, interrupted, and missing-log attempts are saved under
`px4_attempts/session_<UTC timestamp>_<unique ID>/attempt_NNN/` or the equivalent
`ardu_attempts/` path. Each archive contains the available flight records,
process logs, runtime state, and `result.json`; its session also includes a
snapshot of `scenario.yaml`, native parameter files, and generated vehicle files.
New attempts freeze their inputs under `<autopilot>_logs/run_inputs/`, so a later
scenario edit cannot relabel old logs. Retries retain runtime parameters but
clear archived flight records so a later success cannot pick up
an earlier attempt's log. Earlier failures remain available after a retry succeeds.
If archiving fails, the batch stops and leaves the active evidence in place.

`--force` removes the active logs and reruns without deleting attempt archives.
An unfinished attempt is archived before its output can be overwritten on a
later invocation. A run is skipped only when `result.json` records success,
its input fingerprint matches the current scenario and launch configuration,
and its nonempty collected flight logs still match their recorded sizes and
SHA256 hashes. The input fingerprint includes the contents of `scenario.yaml`,
validated native boot parameters, and effective launch settings such as the
startup delay, timeout, and headless mode. Retry count and inter-run delay do not
invalidate a completed flight. Native parameter files are validated before any
skip decision; stale or inconsistent files cause an error without altering logs.

Changed inputs, missing/invalid results, old results without fingerprints, and
missing/empty/modified flight logs cause archival and a new run. Historical logs
without an input snapshot are marked `input_snapshot_missing`; the current
scenario is never attached as if it described that old flight. Log hashes detect
changes after collection; this check does not parse the complete BIN/ULog format.

Editing a baseline/profile YAML requires regenerating the run scenarios and
native parameter files before relaunching. With the default gain environment
policy (`off`), firmware binaries are not fingerprinted; use `--force` after
changing them. Enabled policies also include the current firmware binary and
selected startup files in the run fingerprint. External model assets and launcher
source versions are not fingerprinted under any policy; use `--force` after
changing those. Default plotting reads the usual log directories and does not
include attempt archives.

Run the launcher regression tests without starting SITL:

```bash
python3 -B -m unittest discover -s tests/tools/launcher -v
```

The environment snippet for other Gazebo tools can be loaded separately with
`source gz/env/px4_gz_env.sh`; its generator is `install/autopilot.sh --phase env`.

## Plotting an archived flight attempt

`plot_sitl_autopilot_response.py` accepts an explicit BIN, ULog, or CSV with
`--log-file`. Select exactly one autopilot; `--log-file` and `--run-dir` are
mutually exclusive. This also works for timed-out or interrupted flights that
have a readable log:

```bash
python3 -m tools.plotting.plot_sitl_autopilot_response \
  --log-file path/to/session/attempt_002/logs/00000001.BIN \
  --autopilots ardu --prefix run_001_attempt_002 \
  --output-dir results/plotting/attempt_002
```

The output contains the 3D trajectory, position, velocity, attitude, and body-rate
histories. Use distinct prefixes or output directories for different attempts,
since archived BIN filenames may be identical. The default `--run-dir` mode
continues to read the active autopilot log directories.

## Comparing wind conditions across runs

Use one autopilot across multiple runs of the same mission:

```bash
python3 -m tools.plotting.plot_sitl_run_comparison \
  --run-dirs data/sitl_wind_test/run_000 data/sitl_wind_test/run_001 \
  --autopilot px4 --output-dir results/plotting/sitl_wind_test
```

Use `--autopilot ardu` for ArduPilot. The script produces horizontal/3D trajectory
overlays plus position, velocity, attitude, and body-rate histories.
Legends show the scenario wind target in ENU; `wind.json` records acknowledgement.
Position and velocity use ENU, retaining each log's local origin; use runs with
the same local reference. Velocity includes the speed magnitude. Attitude
(roll/pitch/yaw) and body rates (p/q/r) retain the logged axes and are shown in
degrees and degrees/second. The existing loader uses PX4 estimated
local position and ArduPilot SIM2 position (falling back to XKF1/NKF1).

By default, time zero is the first sample with absolute vertical velocity at
least 0.2 m/s, an approximate takeoff reference. `--align none` retains time since
the first sample of each series. All histories retain each run's full duration
and use the same run colors and wind labels. Settings other than wind that
differ trigger a warning. Non-success `result.json` states are rejected unless
`--include-incomplete` is explicitly supplied; partial flights are then labelled.
Historical logs without `result.json` remain supported.

`--run-dirs` also accepts individual archived `attempt_NNN` directories alongside
normal runs. The attempt's own `result.json` is checked, and wind/mission metadata
comes from its parent session's `scenario.yaml` snapshot. Failed attempts still
require `--include-incomplete`:

```bash
python3 -m tools.plotting.plot_sitl_run_comparison \
  --run-dirs data/sitl_wind_test/run_000 \
    data/sitl_wind_test/run_001/ardu_attempts/session_20261020T030632Z_bceha83i/attempt_001 \
  --autopilot ardupilot --include-incomplete \
  --output-dir results/plotting/sitl_wind_test/ardu_compare
```

## Per-scenario mass and inertia scaling

Generate FIRE PX4Vision scenarios with fixed `base_link` multipliers:

```bash
python3 -m tools.scenario.scenario_generator planar_n_pts \
  --outdir data/fire_inertial_fixed --runs 1 --seed 260930 \
  --base-mass-scale 1.2 --base-inertia-scale 1.0
```

Or sample each multiplier independently from a uniform range for each run:

```bash
python3 -m tools.scenario.scenario_generator planar_n_pts \
  --outdir data/fire_inertial_random --runs 10 --seed 260930 \
  --base-mass-scale random --base-mass-scale-range 0.8 1.2 \
  --base-inertia-scale random --base-inertia-scale-range 0.9 1.1
```

These options work for both `planar_n_pts` and `three_d_n_pts`. A scale must be
finite and strictly positive. The default scale is 1.0; the default random range
is `[0.8, 1.2]`. All six inertia tensor components use the same inertia multiplier.
Rotor masses, rotor inertias, geometry, centre of mass and motor settings are
unchanged. With the current model, a mass scale of 1.2 gives a 1.8 kg body and
1.82 kg total mass. Set both fixed scales to the same value to scale body mass
and inertia together; two `random` options are sampled independently.

Either scale option enables `common.vehicle` and selects these defaults unless
they are explicitly supplied:

- ArduPilot: `gazebo-px4vision`, `JSON`, `default_fire_px4vision`.
- PX4: airframe `4006`, `gz_fire_px4vision`, `default_fire`.

`--vehicle-model fire_px4vision` enables generated SDFs at unit scales.
Without these options, the previous iris / x500 defaults and original SDFs are
used. Incompatible explicitly selected frame/model settings are rejected.

`scenario.yaml` stores the sampled numeric scales under
`common.vehicle.inertial`, with `mode: scale` and `target_link: base_link`.
`metadata.yaml` records the distributions and seed. Omit `--seed` to generate
and record a new seed. Separate per-run streams for mission, wind, mass and
inertia preserve the same mission and wind when scaling options change.
The same seed, run ID and input options reproduce the samples; `--start-run-id`
can regenerate a particular run.

The batch launchers generate the actual SDFs **before simulation starts**.
Prepare ArduPilot SDFs without starting processes or modifying logs:

```bash
python3 -m tools.launcher.run_ardupilot_gz_sitl \
  --run-root data/fire_inertial_random --prepare-only
```

Run the scenarios with ArduPilot:

```bash
source gz/env/ardupilot_gz_env.sh
python3 -m tools.launcher.run_ardupilot_gz_sitl \
  --run-root data/fire_inertial_random --headless
```

The PX4 batch launcher also applies the same `common.vehicle` scales:

```bash
python3 -m tools.launcher.run_px4_gz_sitl \
  --run-root data/fire_inertial_random --headless
```

Run each autopilot batch separately. Per-run artifacts are:

```text
run_000/
  scenario.yaml
  generated/
    vehicle/model.sdf
    resolved_vehicle.yaml
    world_ardupilot.sdf            # ArduPilot preparation only
    resolved_ardupilot_world.yaml  # ArduPilot preparation only
  ardu_logs/
  px4_logs/
```

Each preparation scales the original model, so scales never accumulate.
ArduPilot's generated world references the run's model by absolute path while
preserving the include's entity name, pose and adapter plugins. PX4 spawns the
generated model directly in its standalone world. Missing target links, invalid
inertia tensors, and mismatched world/model selections fail before launch.

The manifests record resolved link masses, inertia tensors, total mass, and
source/generated SDF paths and SHA-256 hashes. Source SDFs are never edited.
Mesh assets are referenced through absolute paths to the original model; these
artifacts depend on that checkout. A new preparation uses the current source
SDF, so keep its recorded version when reproducing an older experiment.
The scaling options do not retune or reset flight-controller parameters.

Check the implementation without flying:

```bash
python3 -B -m unittest discover -s tests/tools -v
gz sdf -k data/fire_inertial_random/run_000/generated/world_ardupilot.sdf
```

### 🚀 Features

- Unified workspace for **PX4 + ArduPilot**
- ROS 2 Humble + DDS integration
- Gazebo (Harmonic) simulation support
- Container-based reproducible environment
- Modular structure for trajectory generation and testing

---

### 📁 Repository Structure
```
FIRE_flightstack_sim/
├── ap/
│ ├── px4/                    # PX4-Autopilot (cloned during installation)
│ └── ardupilot/              # ArduPilot (cloned during installation)
│
├── data/
│ ├── run_XXX/                # Data folder (generated by scenario generator)
│ │     ├── ardu_logs         # Ardupilot SITL logs (generated by Ardupilot SITL launcher)
│ │     └── px4_logs          # PX4 SITL logs (generated by PX4 SITL launcher)
│ └── run_OOO/ 
│
├── ros2/
│ └── px4_ros_uxrce_dds_ws/   # ROS 2 + Micro XRCE-DDS workspace
│
├── tools/
│ ├── commander/              # MAVLink-based commander
│ ├── launcher/               # SITL launcher / orchestration tools
│ ├── scenario/               # Scenario generation utilities
│ ├── QGC/                    # QGroundControl related tools
│ └── misc/ # Miscellaneous utilities
│
├── install/
│ ├── base.sh # Base dependency installation
│ ├── autopilot.sh # PX4 / ArduPilot setup
│ ├── ros2.sh # ROS2 setup
│ ├── gazebo.sh # Gazebo setup
│ ├── extra.sh # Additional tools and dependencies
│ ├── usersetup.sh # User environment setup
│ ├── entrypoint.sh # Docker entrypoint
│ └── clean.sh # Cleanup script
│
├── scripts/
│ ├── get_src.sh # Clone external repositories
│ ├── setup_docker.sh # Build, run, open, stop, and remove the Docker container
│ ├── setup_local.sh # Local environment setup
│ └── setup_ml.sh # ML environment setup (optional)
│
├── ws/
│ ├── data/ # Processed dataset / features
│ ├── drone_classifier/ # ML models / classifiers
│ └── pyproject.toml # Python project config
│
├── docker/
│ ├── Dockerfile # Container build definition
│ └── compose.yaml # Docker compose configuration
│
├── logs/ # Runtime logs
├── test_mission.plan # Example mission file
├── README.md
└── LICENSE
```

---

## ⚙️ Installation Options

Two installation workflows are supported:

### 1️⃣ Container-based Setup (Recommended)

This is the **recommended method** for reproducibility and dependency isolation.

Run these commands from the repository root. Both `source` and `bash` are
supported; sourcing leaves the host shell's options, variables, and traps intact.

```bash
source scripts/setup_docker.sh build   # Fetch sources/artifacts and build the image
source scripts/setup_docker.sh run     # Start in the background; return the terminal
source scripts/setup_docker.sh open    # Enter an independent container Bash session
```

`run` reuses an existing container and restarts it if stopped. It requires an
image built with `build`; it does not automatically build or replace containers.
`run` starts the container in the background and returns without waiting for
initialization. On the first run or after recreation, the initial build and
environment setup may take several minutes or longer. Wait until initialization
finishes before connecting with `open`. The script displays this reminder and
the log command while initialization is pending:

```bash
docker logs -f fire_flightstack_sim
```

When `[ENTRYPOINT] Setup completed.` appears, retry `open`. On subsequent
starts, setup may be skipped with `[ENTRYPOINT] Setup already done or skipped.`:

```bash
source scripts/setup_docker.sh open
```

Pressing `Ctrl+C` while following these logs only ends log monitoring; the
container continues running. `open` checks the readiness marker before connecting.
If initialization is still in progress, it prints an `[INFO]` message and the log
command, then returns a nonzero status without opening a shell. It checks logs
from the current container start and reports an `[ERROR]` with recent logs when
the entrypoint records a fatal failure or the container exits with a nonzero
code. Errors from earlier starts and nonfatal dependency messages are not
classified as current initialization failures.

Exit the shell with `exit`; the container keeps running, and multiple terminals
can open independent sessions.

```bash
source scripts/setup_docker.sh open --new-terminal  # Linux window or WSL Windows Terminal tab
source scripts/setup_docker.sh stop                # Keep the stopped container
source scripts/setup_docker.sh remove              # Remove the container, keeping the image
```

Container build outputs (`build`, ROS workspace `install`/`log`, and generated
`gz/env` files) are stored under `.docker_home/artifacts/<container-name>` and
mounted over their usual paths inside the container. The host's native build
outputs are preserved. This prevents CMake caches containing host paths from
being reused at the container's different project path. Containers created
before this isolation was added need `run --recreate` to receive the new mounts.
Changing Docker's image build cache with `build --no-cache` does not fix caches
inside a bind-mounted source workspace.

`open` uses the current terminal by default. `--new-terminal` needs a graphical
Linux terminal or Windows Terminal with WSL; over SSH, use `open` in another
terminal instead.

`build` always runs a Docker build and reuses cached layers, so no separate
`rebuild` command is needed. After changing the Dockerfile or image dependencies:

```bash
source scripts/setup_docker.sh build --no-fetch  # Skip fetching sources again
source scripts/setup_docker.sh run --recreate   # Replace the container with the updated image
```

Use `build --no-cache` only when a fresh build without cached layers is needed.
`build --fetch-only` runs the source/artifact fetch steps without building an image.
`run --recreate` reruns project setup. Both `remove` and `run --recreate` delete
files stored only inside the container; bind-mounted project files are retained.

X11 forwarding is enabled when `DISPLAY` is set; `run --no-x11` disables it.
Use `run --gpu` for NVIDIA GPU access. Creation options (including `--gpu`, X11,
image tags, and networking) take effect on a new container; repeat the desired
options with `run --recreate` to change an existing one. Run
`source scripts/setup_docker.sh help` for all options.

### 2️⃣ Local Setup (Advanced)

The local setup targets Ubuntu 22.04 (including WSL), ROS 2 Humble, and binary
Gazebo Harmonic. Run these commands from the repository root. Both `source`
and `bash` are supported; sourcing preserves the calling shell's state.

```bash
source scripts/setup_local.sh install  # Initial setup: deps -> fetch -> build -> env
```

For subsequent work, select only the needed phase:

```bash
source scripts/setup_local.sh deps     # System packages, ROS/Gazebo, locale/device groups
source scripts/setup_local.sh fetch    # Source checks, auxiliary repos, QGroundControl
source scripts/setup_local.sh build    # Incremental DDS/px4_msgs/Gazebo plugin builds
source scripts/setup_local.sh env      # Environment files, QGC access, and ~/.bashrc
source scripts/setup_local.sh help
```

`build` uses existing sources and local build outputs; it does not run the full
installation or fetch phase. Its underlying helper still performs rosdep
checks and may install missing declared dependencies. PX4 and ArduPilot
firmware builds are separate (see below). `fetch` expects the main source
submodules to be present; it checks them and fetches auxiliary sources/artifacts.
Gazebo/ROS-GZ binary package installation is included in `deps`.

```bash
source scripts/setup_local.sh install --no-fetch  # Sources/artifacts already prepared
source scripts/setup_local.sh deps --upgrade     # Also explicitly run apt-get upgrade
```

Whole-system upgrades and package cleanup are not automatic. `--no-fetch` is
valid only for `install`; `--upgrade` is valid for `install` and `deps`.
No command defaults to an installation: an invocation without arguments prints
help. Use `--debug` with any command to trace its helpers.

`env` updates configuration files and QGroundControl access; it expects fetched
artifacts to exist. Apply the shell configuration in the current terminal with:

```bash
source ~/.bashrc
```

New device-group memberships require logging out and back in. Existing group
IDs are retained; missing input/render groups use OS-assigned IDs unless
`GID_INPUT` or `GID_RENDER` is explicitly set. `deps`, `build`, and `env` can invoke sudo
through their helpers. There is no development-shell command.

🛠️ Build Instructions

Inside the container:

PX4
cd ap/px4
make px4_sitl
ArduPilot
cd ap/ardupilot
./waf configure --board sitl
./waf copter
▶️ Running Simulations
PX4 SITL (Gazebo)
cd ap/px4
make px4_sitl gz
ArduPilot SITL
cd ap/ardupilot
sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map
🔌 ROS 2 Integration

Build ROS 2 workspace:

cd ros2/px4_ros_uxrce_dds_ws
colcon build --symlink-install

Source environment:

source install/setup.bash





🧪 Research Use Cases

This environment is designed for:

CPS vulnerability analysis (e.g., sensor attacks, actuator faults)

Multi-fidelity simulation pipelines

Controller validation (offboard / onboard)

DDS / MAVLink communication experiments

Mixed-reality and hardware-in-the-loop extensions


⚠️ Notes

This repository does not pin exact upstream commits by default.
For reproducibility, consider locking versions manually.

ROS 2 environment should be clean (avoid multiple workspace overlays).

Docker environment is preferred to avoid dependency conflicts.



📚 Related Work

PX4 Offboard Control:
https://github.com/mhcho1994/px4_ros_autopilot

Control Algorithms (Go / PD):
https://github.com/Balt-AA/balt_go_pd

Mixed Reality / Simulation References:
https://github.com/kpant14/px4-gz-docker

https://github.com/CogniPilot/mixed_sense


📄 License

This repository is intended for research and experimental use.


Repository for PX4(v1.15.4)-ROS(Humble)-Gazebo(Harmonic) SiTL simulator \
This repository is created for personal research and for testing control algorithms. \
For low-level PX4 offboard control algorithm with safety filter, please refer to another repositories: https://github.com/mhcho1994/px4_ros_autopilot and https://github.com/Balt-AA/balt_go_pd. 
This is an updated version of the original repository with mixed-reality, \
please refer to the following links: https://github.com/kpant14/px4-gz-docker and https://github.com/CogniPilot/mixed_sense. \

### Installation and Launch
Two options are available for installing the simulator: installing at local Linux host or running a development container

#### 1. Local installation and running simulations at Linux host computer (WSL2)
If you try to set up PX4-ROS-Gazebo in your local environment,

```bash
source scripts/setup_local.sh install
```
#### 2. Local installation and running simulations at Linux host computer (WSL2)
Build the dockerfile and launch simulation environment
Note that this is not the finalized version of the container for SITL.
The current version uses the classic Gazebo, it will be replaced to the ignition Gazebo soon.

1. To get the sources required (run only once in the first time),

```bash
source ./scripts/get_src.sh
```

2. To build the Docker image and start the container in the background,

```bash
source scripts/setup_docker.sh build
source scripts/setup_docker.sh run
```

3. To enter into the container,

```bash
source scripts/setup_docker.sh open
```
you can use terminator instead of the bash terminal.

4. Build PX4-Autopilot inside the container,
(Check you are inside the container as user id 'user')

```bash
cd px4 && make px4_sitl
```

Run command below to launch a single drone simulation.

```bash
./Tools/simulation/gazebo-classic/sitl_multiple_run.sh -n 1 -m iris
```


  /home/user/ws/flightstack_sim/
  ap/
    px4/                  # PX4-Autopilot git repo
    ardupilot/            # ArduPilot git repo
  gz/
    harmonic_ws/          # colcon ws: src/ build/ install/ log/
  ros/
    sim_ws/               # colcon ws for your ROS pkgs + px4_msgs
    ros_gz_ws/            # optional colcon ws for ros_gz from source
  scripts/
  docker/


## Capture flight controller parameters

`tools/scenario/capture_flight_parameters.py` is the entrypoint for both capture
sources. Select exactly one source; the argument is the input path:

| Source | Input | Example output (`--output`) |
|---|---|---|
| `--from-file LOG` | Original ArduPilot `.BIN`/`.bin` or PX4 `.ulg`, from real flight or SITL | `data/parameters/default_controller_gains.yaml` |
| `--from-sitl SCENARIO` | Generated `scenario.yaml`; boot fresh SITL instances and read startup parameters | `data/parameters/default_controller_gains.yaml` |

Both modes write reusable profile YAML and adjacent `.meta.json` metadata.
The examples below explicitly use
`--output data/parameters/default_controller_gains.yaml` for either source, with
metadata saved to `data/parameters/default_controller_gains.meta.json`.
Run them from the repository root: a relative `--output` is resolved from the
current working directory, and `data/parameters/` is created on capture.
Omitting `--output` still uses the tool's separate defaults for each mode.
These are alternative capture workflows sharing one output path; use `--overwrite`
to replace an existing capture when switching sources.
`--overwrite` works in either mode. `--autopilot`, `--snapshot`, `--groups`, and
`--merge` apply to file capture; `--autopilots`, `--work-dir`, and `--timeout` apply
to SITL capture. Options for the other mode fail before reading or launching.
The implementations retain their existing provenance formats: log history for file
capture, and firmware/startup/session information for SITL capture. `--merge`
combines existing file captures; it does not merge SITL boot captures.

### Capture from a fresh SITL boot

`capture_flight_parameters.py --from-sitl scenario.yaml` boots ArduPilot and PX4 sequentially
in new runtime directories and reads the controller parameters without starting a
mission runner or ground station. It uses the scenario's simulator configuration;
waypoints and wind commands are not executed. The output is a profile containing
all supported groups (ArduPilot: 23 gains; PX4: 20 gains), with `parameters: {}`.
These are startup values for the scenario's selected firmware, vehicle/frame,
airframe and startup parameter files. Changing that configuration can change the
captured values. Changing simulated mass or inertia alone does not automatically
retune controller gains. The examples save to
`data/parameters/default_controller_gains.yaml`; choose another `--output` path to
keep separate profiles for different configurations.
The checked-in `tools/scenario/fixtures/default_controller_gain.yaml` was captured using
`gazebo-px4vision` and PX4 airframe `4006`. When available, the adjacent
`default_controller_gain.meta.json` identifies the actual binaries and startup
configuration. Randomized profiles used by unit tests live under
`tests/tools/scenario/fixtures/`.

Run the following from the repository root, in the same environment used for normal
SITL/Gazebo runs. Gazebo, the ArduPilot plugin, FIRE plugins, and pymavlink must be
available. Existing firmware builds are reused; missing builds are prepared by the
existing build helpers. Rebuild firmware explicitly after source changes. Stop
other SITL instances using the same ports before capture.

1. Generate one basic scenario **without `--controller-gains`**:

   ```bash
   export FLIGHTSTACK_SIM_ROOT="$PWD"
   python3 -m tools.scenario.scenario_generator planar_n_pts \
     --outdir data/default_capture --runs 1 --seed 42 \
     --ardupilot-vehicle ArduCopter --ardupilot-frame gazebo-px4vision \
     --ardupilot-model JSON --ardupilot-world default_fire_px4vision \
     --px4-vehicle 4006 --px4-frame gz_fire_px4vision --px4-world default_fire \
     --n 2 --edge-m 2 --vertex-deg 0 --alt-m 3 --speed-m-s 1 \
     --landing-alt-m 2 --land true \
     --wind-horizontal-magnitude-m-s 0 --wind-vertical-magnitude-m-s 0
   ```

2. Start both SITLs, read their configured startup parameters, and stop the capture processes:

   ```bash
   python3 -m tools.scenario.capture_flight_parameters \
     --from-sitl data/default_capture/run_000/scenario.yaml \
     --output data/parameters/default_controller_gains.yaml \
     --timeout 60 --overwrite
   ```

   `--overwrite` explicitly refreshes the existing SITL capture. Omit it when
   writing a new output path. `--autopilots ardupilot` or `--autopilots px4` captures
   just one stack; use a separate output filename for a partial profile. The tool
   never merges partial captures with an earlier profile. No separate batch
   launcher, MAVProxy, QGroundControl, or commander should be started for this step.

3. Export boot files from the captured profile without creating another mission:

   ```bash
   python3 -m tools.scenario.create_params_sets \
     --profile data/parameters/default_controller_gains.yaml \
     --params-dir data/parameters/sitl_boot --seed 42 --run-id 0
   ```

   This explicitly reads the new SITL capture. Output includes
   `parameters/ardupilot.parm`, `parameters/px4.bson`, and
   `controller_gains.resolved.yaml` under `data/parameters/sitl_boot/`. Since the
   captured profile has no overrides, its gain values are independent of the seed.
   To use them in a flight scenario, repeat step 1 with a new `--outdir` and add
   `--controller-gains data/parameters/default_controller_gains.yaml`, then run the
   normal batch launcher on that dataset. Omitting `--profile` in
   `create_params_sets.py`, or giving `--controller-gains` without a path, selects
   the repository's `tools/scenario/fixtures/default_controller_gain.yaml` baseline.

Each capture creates `parameter_captures/capture_*/` beside `scenario.yaml`;
`--work-dir` changes this parent directory. Every invocation has fresh EEPROM/PX4
rootfs state; it never deletes or reuses the original run's logs or saved parameters.
ArduPilot starts directly with the selected frame's default files and `-w`.
PX4 uses its normal airframe startup with fresh storage; inherited `PX4_PARAM_*`,
`PX4_SIM_SPEED_FACTOR`, and `PX4_ESTIMATOR` overrides are excluded.

Capture rejects scenarios with configured gains, an armed/wrong autopilot,
missing/non-REAL32/nonfinite parameters, or values that change between two reads.
It publishes the profile only after all requested stacks succeed. Failure leaves
the previous output intact and retains the session's `capture.json` and process
logs. Ctrl+C stops the capture's processes. `--timeout` bounds connection/readback
per stack after launch; PX4 Gazebo startup has a separate bounded readiness wait.

The adjacent `default_controller_gains.meta.json` records the profile/scenario
hashes, binary hash, source commit and tracked-dirty status, startup file hashes,
airframe, time, and session directory. Filesystem paths are relative to the directory
containing each JSON file (`paths_relative_to: metadata_directory`), including
`scenario`, `session_dir`, firmware paths, startup-file keys, and `world`. The same
rule applies to the session's `capture.json`, where `session_dir` is `.`. Paths
outside that directory can contain `..`; keep the relative directory layout when
moving captures. Recorded `command` paths and `command_cwd` use this same base;
command paths are normalized for inspection, not a shell replay command. Runtime
process arguments retain their original paths. Profile values are self-contained.
Compare the metadata's `profile_sha256` to the YAML when moving or replacing these files. PX4's `MC_*RATE_K` and `MC_YAW_WEIGHT`
are recorded as context, not included as adjustable gains in the generated profile.
The capture scope is the supported controller gains, not every firmware parameter.

Python callers can use `capture_flight_parameters(from_sitl=scenario, output=output, ...)` from
`tools.scenario.capture_flight_parameters`. The lower-level
`read_controller_parameters(connection, autopilot)` reads an existing MAVLink
connection; the fresh-boot workflow captures startup values for the selected SITL configuration.

### Capture from existing BIN/ULog files

`capture_flight_parameters.py --from-file LOG` reads an existing flight-controller
log offline, including logs produced by SITL. Use an original ArduPilot `.BIN`/`.bin`
or PX4 `.ulg`, such as `FIRE_moonshot_classifier` files under
`data/flight_test/<dataset>/prepared/flight/fc_logs/` or `data/sitl/<dataset>/run_*/`.
Processed trajectory CSV/NPZ files do not preserve the parameter records. This
command does not launch SITL or connect to an autopilot. BIN decoding requires
`pymavlink`; ULog decoding requires `pyulog`.

From the repository root, for example:

```bash
python3 -m tools.scenario.capture_flight_parameters \
  --from-file ../FIRE_moonshot_classifier/data/flight_test/260910_flight_logs/prepared/flight/fc_logs/00000004.BIN \
  --output data/parameters/default_controller_gains.yaml

# Replace this input with an original PX4 real-flight or SITL log path.
python3 -m tools.scenario.capture_flight_parameters \
  --from-file /path/to/flight.ulg \
  --output data/parameters/default_controller_gains.yaml \
  --merge
```

The file header selects the BIN or ULog decoder, independently of its extension.
`--autopilot auto` (the default) identifies the flight stack from the logged
firmware: BIN `MSG.Message` / `VER.FWS` banners such as `ArduCopter V4.7.0`, or
ULog `sys_name` such as `PX4`. Version information comes from the banner / `VER`
fields or ULog `ver_sw_release`; ULog `ver_sw` records the firmware revision.
The CLI prints the identified stack and available version.

Missing/unsupported firmware identity, conflicting stack/version records, or a
mismatch with an explicit `--autopilot ardupilot|px4` fails before output is
replaced. An explicit stack does not bypass firmware verification. If the stack
is identified but its release version was not recorded, the version is saved as
`null`; it is not guessed from the filename, hardware name, or gain names.

A BIN converted from PX4 ULog can therefore be identified as PX4 if its firmware
banner is retained. It must also contain every required gain for the selected
groups; conversion does not recover parameter records that were omitted.
Known parameter-name/unit mappings are applied to the actual logged names, not
chosen speculatively from a firmware version.

Each invocation creates two files:

- `<name>.yaml`: a profile for the detected autopilot, with all supported groups,
  inline `baseline` values, and `parameters: {}`. These are logged flight settings,
  not factory defaults. ArduPilot has 23 supported gains and PX4 has 20.
- `<name>.meta.json`: `schema_version: 2`, the complete YAML's `profile_sha256`,
  and an `autopilots` mapping. Each autopilot entry retains its source hash,
  relative `source_log` path, `log_format`, and `firmware_identity` (stack, version,
  version type, revision and identification evidence), plus original firmware
  records, gain/context values, first-observation timestamps, observed changes, conversions,
  snapshot selection, and capture time. Paths are relative to the JSON directory.

To add PX4 to an existing ArduPilot capture, use the **same output path** with
`--merge` (replace the input with your original ULog):

```bash
python3 tools/scenario/capture_flight_parameters.py \
  --from-file /path/to/flight.ulg \
  --output data/parameters/default_controller_gains.yaml \
  --merge
```

Run captures sequentially. `--merge` creates the output when neither file exists;
otherwise it adds or replaces the incoming autopilot's entire profile and its
metadata entry, preserving the other autopilot's groups, values, rules, and
provenance. For example, importing another ArduPilot BIN refreshes only `ardupilot`.
A selected subset of groups replaces that autopilot's previous group selection;
old groups are not retained. Each autopilot keeps its own `snapshot` and
`captured_at`; `updated_at` describes the most recent file update.

Existing captures from the original tool, including a single autopilot's metadata
at the JSON root, are upgraded automatically during a merge. New captures always
write the `autopilots.ardupilot` / `autopilots.px4` structure, even for a single stack.
Both the existing YAML and its metadata must be present and consistent, including
the YAML hash and each baseline hash. A missing or mismatched pair fails without
changing either file. The previous source logs need not still exist. Source paths
remain relative to the JSON directory after migration.

Without either flag, existing output still raises an error. `--overwrite` replaces
the entire YAML and metadata with the new capture; it cannot be combined with
`--merge`. A merged profile works with `create_params_sets.py` and
`scenario_generator.py` exactly like any profile containing both autopilots.

`--snapshot initial` is the default. For BIN, this means the first observed
`PARM.Value` for each parameter, **not** its `Default` field or a simultaneous boot
snapshot. For ULog, it means the initial parameter section; a parameter first seen
later is unavailable in the initial snapshot. `--snapshot final` selects the last
logged values. Both choices preserve observed gain/context changes in metadata,
including changes that later return to the initial value. One static profile does
not replay changes during flight. ULog change timestamps come from the preceding
data sample, so they are approximate; the decoder reads all topics to retain them.

The output uses the parameter names in `fc_parameter_profiles.py`. For ArduPilot
4.7 logs, the supported `PSC_NE_*` / `PSC_D_*` gains are mapped to this checkout's
`PSC_POSXY_*`, `PSC_POSZ_*`, `PSC_VELXY_*`, `PSC_VELZ_*`, and `PSC_ACCZ_*` names.
Position and velocity PID values retain their scale; `PSC_D_ACC_{P,I,D}` values
are multiplied by 10 when exported as `PSC_ACCZ_{P,I,D}`, reversing the
[documented ArduPilot migration](https://github.com/ArduPilot/ardupilot/blob/1511f271/libraries/AC_AttitudeControl/AC_PosControl.cpp#L1621-L1701).
Metadata preserves the original names/values and every applied multiplier. This is
a conversion of the supported gain subset, not a complete firmware migration or
a guarantee of identical aircraft behavior.

PX4 `MC_*RATE_K` and `MC_YAW_WEIGHT` are saved as context when available, including
their changes. They are not applied by the current gain-profile pipeline; compare
this context when reproducing a flight setup. Filters, feed-forward terms, limits,
and other flight parameters are outside the current profile scope.

Missing required gains fail without substituting SITL defaults or silently dropping
a group. For a partial log, explicitly choose complete groups, for example
`--groups attitude position velocity`. Unknown names/units are not inferred.
Existing output files are updated only with `--merge` or `--overwrite`; decoding
and validation finish before replacement. Use completed logs: a source file that changes
during capture is rejected. YAML is published last, with its hash in the sidecar;
check that hash if publication was interrupted.

The saved profile works with the existing native boot-file path:

```bash
python3 -m tools.scenario.create_params_sets \
  --profile data/parameters/default_controller_gains.yaml \
  --params-dir data/parameters/flight_test_boot --seed 42 --run-id 0
```

To generate missions with those settings, add
`--controller-gains data/parameters/default_controller_gains.yaml` to the scenario
creation command above, using a new `--outdir`. Only the autopilot present in the
profile receives custom gains. The scenario generator writes the native parameter
file for startup; the normal launcher then applies it. The repository baseline at
`tools/scenario/fixtures/default_controller_gain.yaml` is not changed by this workflow.

Python callers can merge both profiles and their provenance with the same API:

```python
from tools.scenario.capture_flight_parameters import capture_flight_parameters

output = "data/parameters/default_controller_gains.yaml"
capture_flight_parameters(from_file="/path/to/flight.BIN", output=output, merge=True)
profiles = capture_flight_parameters(from_file="/path/to/flight.ulg", output=output,
                                     snapshot="final", merge=True)
```

The return value contains all profiles saved in the combined YAML. The lower-level
`read_flight_parameters(...)` still returns `(profile, metadata)` for a single input
without writing files, with an absolute source path in memory. The capture function
serializes that path relatively under the corresponding autopilot entry. If you
combine profiles yourself with `create_params_sets()` / `save_params_sets()`, the
default is to write only profile YAML. For a derived profile, obtain provenance
with `tools.scenario.gain_provenance.load_capture_provenance(source, profiles)` and
pass it to `save_params_sets(profiles, output, provenance=provenance)` to preserve
its capture metadata. The builder CLI handles this with `--preserve-gain-provenance`.

## Per-run controller gains

Both SITL launchers support boot-time rate, attitude, position, and velocity gains.
ArduPilot also supports the vertical acceleration PID loop. Select the loops with
`groups` in each autopilot's profile. The supported parameters match the firmware
sources in this checkout:

| Group | ArduPilot | PX4 |
|---|---|---|
| `rate` (9) | `ATC_RAT_{RLL,PIT,YAW}_{P,I,D}` | `MC_{ROLL,PITCH,YAW}RATE_{P,I,D}` |
| `attitude` (3) | `ATC_ANG_{RLL,PIT,YAW}_P` | `MC_{ROLL,PITCH,YAW}_P` |
| `position` (2) | `PSC_POSXY_P`, `PSC_POSZ_P` | `MPC_XY_P`, `MPC_Z_P` |
| `velocity` (6) | `PSC_VEL{XY,Z}_{P,I,D}` | `MPC_{XY,Z}_VEL_{P,I,D}_ACC` |
| `acceleration` (3, ArduPilot only) | `PSC_ACCZ_{P,I,D}` | No separate supported acceleration PID group |

Attitude and position loops have P gains. Horizontal position/velocity parameters
are shared by X and Y in these controllers; they cannot be sampled separately by
axis. All listed parameters support independent `set`, `scale`, or `offset`
rules with constant values or uniform sampling. Filters, FF, limits, and PX4's
additional rate multipliers and yaw weight are not sampled by the profile.

The generator resolves rules to absolute values in
`autopilots.<ardupilot|px4>.controller_gains` and writes native parameter files:

- ArduPilot: `run_XXX/parameters/ardupilot.parm` (`NAME VALUE` defaults).
- PX4: `run_XXX/parameters/px4.bson` (native BSON with float32-rounded gains).

Each scenario records the relative file path and its SHA256. Launchers validate
that the file matches the resolved values, then copy it into the attempt logs.
Missing, modified, or stale files fail before simulation processes start.
ArduPilot receives `--add-param-file` **and `--wipe-eeprom`**: existing stored
values override defaults, so gain-enabled attempts reset their own run's SITL
EEPROM and rebuild settings from the frame defaults plus the gain file. This
also resets saved parameters outside the selected gain groups to frame defaults.
PX4 receives a per-attempt startup script through `-s`; it runs `param import`
after airframe defaults and before dataman, simulator, and controller startup.
The original firmware/build startup script is not edited. An unrecognized rcS
layout or a failed import stops startup; PX4's stored nonselected parameters
continue to follow its normal airframe/autoconfiguration behavior.

The commander sends **no MAVLink PARAM_SET for gains**. It reads the requested
gains before readiness checks and ARM, rejecting missing parameters, wrong types,
timeouts, or value mismatches. MAVLink remains in use for readback and mission
control. Scenarios without a gain block use the original startup path.

Create a reusable YAML profile with the new CLI:

```bash
python3 -m tools.scenario.create_params_sets \
  --output tools/scenario/my_controller_gains.yaml \
  --groups rate --baseline-only \
  --random-scale MC_ROLLRATE_P 0.9 1.1 \
  --offset MC_ROLLRATE_I 0.01 \
  --set MC_ROLLRATE_D 0.0015 \
  --random-scale ATC_RAT_RLL_P 0.9 1.1
```

`--profile` selects an existing input YAML (default: the captured default profile).
`--autopilots px4` or `--autopilots ardupilot` selects one stack. `--baseline-only`
clears input rules before adding overrides; otherwise the input rules remain.
`--set`, `--scale`, and `--offset` accept `PARAM VALUE`; their `--random-*`
forms accept `PARAM LOW HIGH`. Each parameter can have its own operation/range.
To export native files without generating a mission, add
`--params-dir data/parameter_trial --seed 42 --run-id 0`. This also writes
`controller_gains.resolved.yaml`, mapping autopilots to frozen scenario gain blocks.

The same API is available to Python callers and is used by the scenario generator:

```python
from pathlib import Path
from tools.scenario.create_params_sets import (
    create_params_sets, save_params_sets, resolve_params_sets,
)

profiles = create_params_sets(
    autopilots=["px4"], groups=["rate"],
    parameters={"px4": {
        "MC_ROLLRATE_P": {"operation": "scale", "uniform": [0.9, 1.1]},
        "MC_ROLLRATE_I": {"operation": "offset", "value": 0.01},
        "MC_ROLLRATE_D": {"operation": "set", "value": 0.0015},
    }},
)
save_params_sets(profiles, Path("tools/scenario/my_controller_gains.yaml"))
resolved = resolve_params_sets(profiles, seed=42, run_id=0,
                               run_dir=Path("data/parameter_trial"))
# resolved["px4"] is a controller_gains block for autopilots.px4.
```

`create_params_sets()` returns a regular dictionary in the profile YAML schema;
it does not write files or sample. Per-stack `parameters` arguments replace that
stack's rules. `resolve_params_sets()` samples once and optionally writes files.
The generator also accepts `--controller-gains` without a path to call the shared
profile builder with the captured default profile. Omitting the `--controller-gains` option
entirely leaves gain configuration disabled and produces no gain parameter files.
The default profile includes all supported groups for each stack and has
`parameters: {}`, so it preserves the captured baseline. Random sampling occurs
only when an input profile or API/CLI overrides explicitly contain `uniform` rules.
An explicit YAML path keeps experiments independent of future default refreshes.

Start with [the captured controller defaults](tools/scenario/fixtures/default_controller_gain.yaml):

```bash
export FLIGHTSTACK_SIM_ROOT="$PWD"
python3 -m tools.scenario.scenario_generator planar_n_pts \
  --outdir data/controller_gain_trials --runs 3 --seed 42 \
  --controller-gains tools/scenario/fixtures/default_controller_gain.yaml \
  --ardupilot-frame gazebo-px4vision --ardupilot-world default_fire_px4vision \
  --px4-vehicle 4006 --px4-frame gz_fire_px4vision --px4-world default_fire \
  --n 2 --edge-m 2 --vertex-deg 0 --alt-m 3 --speed-m-s 1 \
  --landing-alt-m 2 --land true
python3 -m tools.launcher.run_px4_gz_sitl \
  --run-root data/controller_gain_trials --headless --startup-delay-s 2 --max-run-s 180
python3 -m tools.launcher.run_ardupilot_gz_sitl \
  --run-root data/controller_gain_trials --headless --startup-delay-s 2 --max-run-s 180
```

The profile has an `ardupilot` and/or `px4` mapping. Each contains `groups`,
`baseline`, and optional `parameters` rules. **Baseline must include every gain
in each selected group**, even when only one gain is varied. You can select a
single group without providing gains for other loops. For example:

```yaml
px4:
  groups: [attitude, position]
  baseline:
    MC_ROLL_P: 6.5
    MC_PITCH_P: 6.5
    MC_YAW_P: 2.8
    MPC_XY_P: 0.95
    MPC_Z_P: 1.0
  parameters:
    MC_ROLL_P:
      operation: scale
      uniform: [0.95, 1.05]
    MPC_XY_P:
      operation: offset
      uniform: [-0.05, 0.05]
    MPC_Z_P:
      operation: set
      value: 1.05
```

Alternatively, `baseline: baselines/mav.parm` reads a MAVProxy `NAME VALUE`
snapshot relative to the profile. Only the selected groups are extracted; all
other parameters in the snapshot are ignored. Inline mappings must contain
exactly the selected groups. Unknown groups, missing baseline values, and rules
for unselected parameters are rejected before generating output.

For baseline value `K0`, the operations are `set: K = value`,
`scale: K = K0 * value`, and `offset: K = K0 + value`. Every rule uses exactly
one of `value` (constant) or `uniform: [minimum, maximum]` (sampled once per run).
All operations support either form. Offset has the same units as the gain;
scale is dimensionless. A zero baseline can be changed with `offset` or `set`.
Optional `bounds: [minimum, maximum]` limits the **final gain**. Both transformed
range endpoints are checked before generating output; values are never clipped
or resampled. Gains must be finite, non-negative, and representable as float32.
These checks validate input ranges; they do not establish closed-loop stability.
Use a baseline appropriate to the firmware and airframe. The checked-in default
profile contains measured startup values for gazebo-px4vision and PX4 airframe
4006, including their airframe overrides; it contains no experiment rules.

Omitted rules restore the baseline within selected groups. `parameters: {}`
restores all selected groups. ArduPilot's gain-enabled startup resets other saved
settings as described above; PX4 imports the selected gains after its normal
startup defaults. Removing the whole gain block disables this file import path;
it does not independently reset existing SITL parameter storage.

Sampling is independent for every parameter and autopilot. The dataset seed,
run ID, autopilot, and parameter name determine each draw. Adding/reordering
rules or groups does not change other draws or mission, wind, and inertial
streams. `metadata.yaml` records the groups, baseline, its hash, and rules. Each
`scenario.yaml` embeds the baseline, sampled inputs, and final values, so it no
longer depends on the original profile/snapshot. Keep its `parameters/` directory
with it. Replays/retries reuse the same absolute values. Failed-flight archives
include the native input files, generated PX4 startup script, and configuration.

For manual configuration, add a resolved block under `autopilots.ardupilot` or
`autopilots.px4`. For example, under the PX4 entry:

```yaml
controller_gains:
  groups: [position]
  baseline:
    MPC_XY_P: 0.95
    MPC_Z_P: 1.0
  values:
    MPC_XY_P: 1.05
```

`values` contains absolute numbers; omitted entries use `baseline`. Runtime
scenarios do not accept unresolved `parameters` rules or baseline file paths.
The generator writes every selected gain into `values`. After editing values,
regenerate the native files with `write_parameter_files()` or remove both
`parameter_file` and `parameter_file_sha256` fields. Scenarios without these file
fields (including older generated scenarios) are rendered into per-attempt native
files by the launcher; their frozen values are never resampled. If editing a
baseline, remove/recompute its `baseline_sha256`; stale hashes are rejected.
Update/remove `sampling` provenance when manually replacing generated values.

Each attempt writes `ardu_logs/controller_gains.json` or
`px4_logs/controller_gains.json`, with `method: startup_file`, the requested
configuration, readbacks, and any failure. `boot_parameters.json` records the
actual native file's hash, source, and storage policy; `boot_parameters.parm` or
`boot_parameters.bson` is the boot input. PX4 also writes `boot_parameters.rcS`.
PX4 readback reports also
record `MC_*RATE_K` when rate is selected and `MC_YAW_WEIGHT` when attitude is
selected. Reports are archived with unsuccessful attempts. Edited scenarios are
rerun automatically when their input fingerprint changes; `--force` also reruns
unchanged scenarios.

The scenario generator accepts `--gain-environment-policy {off,record,require-match}`:

| Policy | Behavior |
| --- | --- |
| `off` (default) | Does not read capture metadata, inspect firmware, or compare environments. Existing gain application, readback, and scenario/parameter hash checks still apply. |
| `record` | Records capture provenance and the planned/current environment. Differences or missing environment information do not block execution. |
| `require-match` | Requires an environment match. Known differences or insufficient capture information cause an error; unavailable target build artifacts can remain `pending` at generation, but must match after build preparation before launch. |

For example, add `--gain-environment-policy record` to a generation command that
already uses `--controller-gains`. Enabled policies require a gain profile. The
selected policy and a self-contained provenance snapshot are saved in
`autopilots.<stack>.controller_gain_environment` in each scenario, and the planning
report is also stored in `metadata.yaml`. Both launchers honor the saved policy;
scenarios without this block behave as `off`. The generator only inspects files;
it does not build or boot SITL. Neither policy automatically captures new gains
or replaces the baseline. Gain sampling and the frozen `values` are unchanged.

The comparison covers the autopilot, vehicle/frame, firmware commit when available,
binary SHA256, and the startup files recorded by SITL capture: PX4's `rcS`,
`rc.mc_defaults`, and selected airframe; ArduPilot's selected frame defaults.
Inherited PX4 parameter/estimator/speed overrides are also reported against the
clean capture environment. This does not establish compatibility of every flight
parameter, model asset, or transitive startup dependency. BIN/ULog captures normally
lack SITL binary/startup hashes, so `require-match` rejects them as unverified even
if their firmware version agrees. Use `record` or `off` to intentionally use
historical flight gains in a different environment.

Enabled modes verify that the adjacent `.meta.json` belongs to the profile using
its profile and baseline hashes. Missing metadata is recorded as unknown;
malformed metadata or inconsistent hashes are errors under both enabled modes.
When producing an intermediate profile with `create_params_sets.py`, add
`--preserve-gain-provenance` to preserve the original capture in a derived
`.meta.json` alongside `--output`. This flag takes no value and is disabled by
default: omitting it skips capture metadata reads and writes. It validates
profile/metadata consistency and preserves provenance; it does not inspect
firmware or compare SITL environments. The builder no longer accepts
`--gain-environment-policy`.

The scenario generator independently accepts `--gain-environment-policy` with
`off` (default), `record`, and `require-match`; it and the launchers perform
target comparison. In `scripts/run_sitl_batch.sh`, if the generator consumes an
intermediate profile and enables environment checks, add
`--preserve-gain-provenance` to the builder command. If it consumes a captured
profile directly, it reads that profile's own `.meta.json`.

At launch, enabled policies inspect the prepared build again before checking for
reusable results and before flight. Each attempt writes `gain_environment.json`
in its autopilot log directory, with the check also included in `result.json`'s
inputs. A changed execution environment invalidates a previous successful result
under `record`; `require-match` rejects a mismatch before altering previous logs.
`--force` does not bypass the policy. Launches use the saved provenance snapshot,
so the original profile, log, or capture checkout need not remain available.

Profile validation and sampling live in `tools/commander/fc_parameter_profiles.py`,
including `parse_controller_parameters()` for resolved scenario values;
`tools/scenario/create_params_sets.py` provides profile construction and native
file export. Both commanders use the read-only `verify_controller_parameters()` in
`mavlink_parameters.py`. The old `--rate-gains` CLI spelling remains an alias.
Use [the captured controller defaults](tools/scenario/fixtures/default_controller_gain.yaml)
for all groups, or select only rate with `create_params_sets(groups=["rate"])`
or the profile builder's `--groups rate` option. Existing rate profiles and
scenarios that omit `groups` still select only `rate`. All groups now use the
same `controller-gains-v1` sampling namespace. Regenerating scenarios changes
random rate-gain samples relative to the former `rate-gains-v1` namespace, even
with the same seed/run ID; already generated scenarios keep their frozen values.
The `v1` identifies the sampling recipe (seed derivation and random draw method).
Changing a baseline, seed, or variation range does not require a new version;
changing the recipe should use a new version to distinguish experiment provenance.
New metadata uses `controller_gain_sampling`
and `parameters: controller_gain_profile` for the expanded feature.
