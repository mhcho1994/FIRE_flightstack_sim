#!/usr/bin/env bash
python3 tools/scenario/scenario_generator.py three_d_n_pts \
--outdir ./data/sitl_logs --runs 1000 \
--ardupilot-vehicle ArduCopter \
--ardupilot-frame gazebo-px4vision \
--ardupilot-model JSON \
--ardupilot-world default_fire_px4vision \
--ardupilot-location Purdue \
--px4-vehicle 4006 \
--px4-frame gz_fire_px4vision \
--px4-world default_fire \
--px4-location Purdue \
--n 10 \
--edge-m random --edge-m-range 5 20 \
--vertex-deg random --vertex-deg-range 0 360 \
--speed-m-s random --speed-m-s-range 1 20 \
--alt-m random --alt-m-range 2 20 \
--takeoff-alt-m 10 \
--landing-alt-m 5 \
--base-mass-scale random --base-mass-scale-range 0.75 1.25 \
--base-inertia-scale random --base-inertia-scale-range 0.85 1.15 \
--wind-horizontal-magnitude-m-s 0 \
--wind-horizontal-azimuth-deg 0 \
--wind-vertical-magnitude-m-s 0 \
--wind-vertical-direction 1

python3 tools/launcher/run_px4_gz_sitl.py --run-root ./data/sitl_logs --startup-delay 2 --inter-run-delay 3 --max-run-s 300 --verbose --headless
python3 tools/launcher/run_ardupilot_gz_sitl.py --run-root ./data/sitl_logs --startup-delay 2 --inter-run-delay 3 --max-run-s 300 --verbose --headless
