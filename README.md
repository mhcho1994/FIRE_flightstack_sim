# FIRE Flightstack Simulator
A unified simulation environment for **PX4, ArduPilot, and Gazebo (SITL)**, designed for **Motion-based Tactical Identification of Firmware (MOTIF)**.
This repository provides a **reproducible, containerized development environment** for:

- Multi-autopilot simulation (PX4 + ArduPilot) hosed in Gazebo (Harmonic) simulator
- Integration with mavlink-based custom launcher/commander and analysis pipelines

---

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
python3 tools/commander/fake_gcs_heartbeat.py --connect udp:127.0.0.1:14550
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
python3 tools/scenario/scenario_generator.py planar_n_pts \
  --outdir data/px4_fire_standalone --runs 1 \
  --px4-vehicle 4006 --px4-frame gz_fire_px4vision --px4-world default_fire \
  --n 2 --edge-m 2 --vertex-deg 0 --alt-m 3 --speed-m-s 1 \
  --landing-alt-m 2 --land true
python3 tools/launcher/run_px4_gz_sitl.py \
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
python tools/plotting/plot_sitl_run_comparison.py \
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
snapshot of `scenario.yaml` and generated vehicle files. Retries retain runtime
parameters but clear archived flight records so a later success cannot pick up
an earlier attempt's log. Earlier failures remain available after a retry succeeds.
If archiving fails, the batch stops and leaves the active evidence in place.

`--force` removes the active logs and reruns without deleting attempt archives.
An unfinished attempt is archived before its output can be overwritten on a
later invocation. Runs with `result.json` are skipped only after success and
when a flight log is present; historical logs without this file retain the
previous skip behavior. Default plotting reads the usual log directories and
does not include attempt archives.

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
python3 tools/plotting/plot_sitl_autopilot_response.py \
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
python3 tools/plotting/plot_sitl_run_comparison.py \
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
python3 tools/plotting/plot_sitl_run_comparison.py \
  --run-dirs data/sitl_wind_test/run_000 \
    data/sitl_wind_test/run_001/ardu_attempts/session_20261020T030632Z_bceha83i/attempt_001 \
  --autopilot ardupilot --include-incomplete \
  --output-dir results/plotting/sitl_wind_test/ardu_compare
```

## Per-scenario mass and inertia scaling

Generate FIRE PX4Vision scenarios with fixed `base_link` multipliers:

```bash
python3 tools/scenario/scenario_generator.py planar_n_pts \
  --outdir data/fire_inertial_fixed --runs 1 --seed 260930 \
  --base-mass-scale 1.2 --base-inertia-scale 1.0
```

Or sample each multiplier independently from a uniform range for each run:

```bash
python3 tools/scenario/scenario_generator.py planar_n_pts \
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
python3 tools/launcher/run_ardupilot_gz_sitl.py \
  --run-root data/fire_inertial_random --prepare-only
```

Run the scenarios with ArduPilot:

```bash
source gz/env/ardupilot_gz_env.sh
python3 tools/launcher/run_ardupilot_gz_sitl.py \
  --run-root data/fire_inertial_random --headless
```

The PX4 batch launcher also applies the same `common.vehicle` scales:

```bash
python3 tools/launcher/run_px4_gz_sitl.py \
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
