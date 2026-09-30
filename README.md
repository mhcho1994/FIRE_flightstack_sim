# FIRE Flightstack Simulator
A unified simulation environment for **PX4, ArduPilot, and Gazebo (SITL)**, designed for **Motion-based Tactical Identification of Firmware (MOTIF)**.
This repository provides a **reproducible, containerized development environment** for:

- Multi-autopilot simulation (PX4 + ArduPilot) hosed in Gazebo (Harmonic) simulator
- Integration with mavlink-based custom launcher/commander and analysis pipelines

---

## PX4 standalone Gazebo with FIRE PX4Vision

Run these commands from the repository root. Build PX4 once (and rebuild after
changing firmware). The standalone scripts configure Gazebo resource paths,
PX4 sensor plugins, and the server configuration themselves.

```bash
make -C ap/px4 px4_sitl
```

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
`--max-retries 0` runs once; `--force` explicitly removes the old PX4 logs and reruns.

The environment snippet for other Gazebo tools can be loaded separately with
`source gz/env/px4_gz_env.sh`; its generator is `install/autopilot.sh --phase env`.

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
python3 -m unittest discover -s tools/scenario/tests -v
python3 -m unittest discover -s tools/launcher/tests -v
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
├── script/
│ ├── get_src.sh # Clone external repositories
│ ├── run_dev.sh # Launch development container
│ ├── setup_docker.sh # Docker setup
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

#### Step 1. Clone sources
```bash
source ./script/get_src.sh
Step 2. Build and launch container
source ./run_dev.sh
Step 3. Enter container
docker exec -u user -it flightstack_sim bash
2️⃣ Local Setup (Advanced)

⚠️ Recommended only if you need native performance or custom system integration.

Run:

source ./script/install/base.sh
source ./install/autopilot.sh --mode all
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
source ./script/run_lnx.sh
```
#### 2. Local installation and running simulations at Linux host computer (WSL2)
Build the dockerfile and launch simulation environment
Note that this is not the finalized version of the container for SITL.
The current version uses the classic Gazebo, it will be replaced to the ignition Gazebo soon.

1. To get the sources required (run only once in the first time),

```bash
source ./script/get_src.sh
```

2. To build the dockerfile as an image or run the container (the command will build the image if the container does not exist), 

```bash
source run_dev.sh
```

3. To enter into the container,

```bash
docker exec -u user -it drones bash
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