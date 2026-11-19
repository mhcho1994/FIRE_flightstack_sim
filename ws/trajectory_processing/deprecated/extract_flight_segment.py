#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pyulog import ULog
from pymavlink import mavutil
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

matplotlib.use("Agg")

TARGET_HZ = 50.0
DT = 1.0 / TARGET_HZ  # 20 ms




# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    BASE_DATA_DIR = Path("data")
    print("[Info] Starting Combined Pipeline: XY / Full Highlights / Segments...\n")

    for i in range(100):
        run_folder = f"run_{i:03d}"
        run_dir = BASE_DATA_DIR / run_folder
        if not run_dir.exists():
            continue

        px4_dir = run_dir / "px4_logs" / "raw"
        ardu_dir = run_dir / "ardu_logs" / "raw" / "logs"

        x_px4, y_px4, t_px4, feat_px4, segments_px4, spans_px4 = (None,) * 6
        x_ardu, y_ardu, t_ardu, feat_ardu, segments_ardu, spans_ardu = (None,) * 6

        if px4_dir.exists():
            for file in os.listdir(px4_dir):
                if file.lower().endswith(".ulg"):
                    px4_result = process_px4_flight_data(str(px4_dir / file))
                    x_px4, y_px4, t_px4, feat_px4, segments_px4, spans_px4 = px4_result
                    break

        if ardu_dir.exists():
            for file in os.listdir(ardu_dir):
                if file.lower().endswith(".bin"):
                    ardu_result = process_ardu_flight_data(str(ardu_dir / file))
                    x_ardu, y_ardu, t_ardu, feat_ardu, segments_ardu, spans_ardu = ardu_result
                    break

        if (x_px4 is not None) or (x_ardu is not None):
            plot_combined_xy_trajectory(
                x_px4,
                y_px4,
                x_ardu,
                y_ardu,
                title=f"Combined X-Y Trajectory ({run_folder})",
                save_path=str(run_dir / f"trajectory_xy_combined_{run_folder}.png"),
            )

        if feat_px4 is not None and len(feat_px4) > 0:
            print(f"[{run_folder}] Generating PX4 trajectory plot...")
            plot_full_trajectory_with_spans(
                t=t_px4,
                features=feat_px4,
                spans=spans_px4,
                title=f"Trajectory [Segment Check]: PX4 ({run_folder})",
                save_path=str(run_dir / f"features_px4_seg_check_{run_folder}.png"),
                line_color="tab:green",
            )

            if segments_px4 and segments_px4["turn"]:
                first_turn = segments_px4["turn"][0]
                plot_turn_segment_features(
                    t=first_turn["time"],
                    features=first_turn["features"],
                    title=f"Trajectory [Isolated Turn]: PX4 ({run_folder})",
                    save_path=str(run_dir / f"features_px4_turn_seg_{run_folder}.png"),
                    line_color="tab:green",
                )

        if feat_ardu is not None and len(feat_ardu) > 0:
            print(f"[{run_folder}] Generating ArduPilot trajectory plot...")
            plot_full_trajectory_with_spans(
                t=t_ardu,
                features=feat_ardu,
                spans=spans_ardu,
                title=f"Trajectory [Segment Check]: ArduPilot ({run_folder})",
                save_path=str(run_dir / f"features_ardupilot_seg_check_{run_folder}.png"),
                line_color="tab:orange",
            )

            if segments_ardu and segments_ardu["turn"]:
                first_turn = segments_ardu["turn"][0]
                plot_turn_segment_features(
                    t=first_turn["time"],
                    features=first_turn["features"],
                    title=f"Trajectory [Isolated Turn]: ArduPilot ({run_folder})",
                    save_path=str(run_dir / f"features_ardupilot_turn_seg_{run_folder}.png"),
                    line_color="tab:orange",
                )

    print("\n[Info] All combined XY and segmentation plots generated successfully!")