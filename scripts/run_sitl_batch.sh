#!/usr/bin/env bash
# python3 tools/scenario/create_params_sets.py \
# --profile ./tools/scenario/fixtures/purdue_test_controller_gain.yaml \
# --output ./data/sitl_logs/sitl_batch_controller_gains.yaml \
# --baseline-only \
# --random-scale ATC_RAT_RLL_P 0.9 1.1 \
# --random-scale ATC_RAT_RLL_I 0.9 1.1 \
# --random-scale ATC_RAT_RLL_D 0.9 1.1 \
# --random-scale ATC_RAT_PIT_P 0.9 1.1 \
# --random-scale ATC_RAT_PIT_I 0.9 1.1 \
# --random-scale ATC_RAT_PIT_D 0.9 1.1 \
# --random-scale ATC_RAT_YAW_P 0.9 1.1 \
# --random-scale ATC_RAT_YAW_I 0.9 1.1 \
# --random-scale ATC_RAT_YAW_D 0.9 1.1 \
# --random-scale ATC_ANG_RLL_P 0.9 1.1 \
# --random-scale ATC_ANG_PIT_P 0.9 1.1 \
# --random-scale ATC_ANG_YAW_P 0.9 1.1 \
# --random-scale PSC_POSXY_P 0.9 1.1 \
# --random-scale PSC_POSZ_P 0.9 1.1 \
# --random-scale PSC_VELXY_P 0.9 1.1 \
# --random-scale PSC_VELXY_I 0.9 1.1 \
# --random-scale PSC_VELXY_D 0.9 1.1 \
# --random-scale PSC_VELZ_P 0.9 1.1 \
# --random-scale PSC_VELZ_I 0.9 1.1 \
# --random-scale PSC_VELZ_D 0.9 1.1 \
# --random-scale PSC_ACCZ_P 0.9 1.1 \
# --random-scale PSC_ACCZ_I 0.9 1.1 \
# --random-scale PSC_ACCZ_D 0.9 1.1 \
# --random-scale MC_ROLLRATE_P 0.9 1.1 \
# --random-scale MC_ROLLRATE_I 0.9 1.1 \
# --random-scale MC_ROLLRATE_D 0.9 1.1 \
# --random-scale MC_PITCHRATE_P 0.9 1.1 \
# --random-scale MC_PITCHRATE_I 0.9 1.1 \
# --random-scale MC_PITCHRATE_D 0.9 1.1 \
# --random-scale MC_YAWRATE_P 0.9 1.1 \
# --random-scale MC_YAWRATE_I 0.9 1.1 \
# --random-scale MC_YAWRATE_D 0.9 1.1 \
# --random-scale MC_ROLL_P 0.9 1.1 \
# --random-scale MC_PITCH_P 0.9 1.1 \
# --random-scale MC_YAW_P 0.9 1.1 \
# --random-scale MPC_XY_P 0.9 1.1 \
# --random-scale MPC_Z_P 0.9 1.1 \
# --random-scale MPC_XY_VEL_P_ACC 0.9 1.1 \
# --random-scale MPC_XY_VEL_I_ACC 0.9 1.1 \
# --random-scale MPC_XY_VEL_D_ACC 0.9 1.1 \
# --random-scale MPC_Z_VEL_P_ACC 0.9 1.1 \
# --random-scale MPC_Z_VEL_I_ACC 0.9 1.1 \
# --random-scale MPC_Z_VEL_D_ACC 0.9 1.1 || exit $?

python3 tools/scenario/scenario_generator.py three_d_n_pts \
--outdir ./data/sitl_logs --runs 1 \
--controller-gains ./tools/scenario/fixtures/purdue_test_controller_gain.yaml \
--gain-environment-policy record \
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
--landing-alt-m 5
# --base-mass-scale random --base-mass-scale-range 0.75 1.25 \
# --base-inertia-scale random --base-inertia-scale-range 0.85 1.15 \
# --wind-horizontal-magnitude-m-s random --wind-horizontal-magnitude-m-s-range 0 1 \
# --wind-horizontal-azimuth-deg random --wind-horizontal-azimuth-deg-range 0 360 \
# --wind-vertical-magnitude-m-s random --wind-vertical-magnitude-m-s-range 0 0.25 \
# --wind-vertical-direction random \
# --wind-policy mission-only \
# --gust-magnitude-m-s 0 --gust-start-s 0 --gust-duration-s 5  --gust-period-s 10 \
# --gust-azimuth-deg 0 --gust-elevation-deg 0 \
# --turbulence-magnitude-m-s random --turbulence-magnitude-m-s-range 0 1.0 \
# --turbulence-length-scale-m 30 30 10 --turbulence-reference-speed-m-s 5 || exit $?

python3 tools/launcher/run_px4_gz_sitl.py --run-root ./data/sitl_logs --startup-delay 2 --inter-run-delay 3 --max-run-s 300 --verbose --headless
python3 tools/launcher/run_ardupilot_gz_sitl.py --run-root ./data/sitl_logs --startup-delay 2 --inter-run-delay 3 --max-run-s 300 --verbose --headless
