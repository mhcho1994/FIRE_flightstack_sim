#!/usr/bin/env python3
"""
Plotting utilities for trajectory visualization and analysis.

What this script does:
---------------------
- TODO: write down the main steps of the pipeline in a concise manner

"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.use('Agg')  

# =====================================================================
# PLOTS
# =====================================================================
def plot_combined_xy_trajectory(
    traj_raw_px4,
    traj_raw_ardu,
    traj_resampled_px4,
    traj_resampled_ardu,
    title="Combined X-Y Trajectory",
    save_figure=False,
    save_path="xy_trajectory.png",
):
    if (traj_raw_px4.size == 0) and (traj_raw_ardu.size == 0):
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    if traj_raw_px4.size > 0:
        ax.plot(traj_raw_px4[:, 1], traj_raw_px4[:, 0], color="tab:green", linewidth=1.5, alpha=0.8, label="PX4 Path (Raw)")
        ax.plot(traj_resampled_px4[:, 1], traj_resampled_px4[:, 0], color="tab:green", linewidth=1.5, linestyle='--', alpha=0.8, label="PX4 Path (Smoothed)")
        ax.plot(traj_raw_px4[0, 1], traj_raw_px4[0, 0], marker="o", color="darkgreen", markersize=6, label="PX4 Start")
        ax.plot(traj_raw_px4[-1, 1], traj_raw_px4[-1, 0], marker="X", color="darkgreen", markersize=6, label="PX4 End")

    if traj_raw_ardu.size > 0:
        ax.plot(traj_raw_ardu[:, 1], traj_raw_ardu[:, 0], color="tab:orange", linewidth=1.5, alpha=0.8, label="ArduPilot Path (Raw)")
        ax.plot(traj_resampled_ardu[:, 1], traj_resampled_ardu[:, 0], color="tab:orange", linewidth=1.5, linestyle='--', alpha=0.8, label="ArduPilot Path (Smoothed)")
        ax.plot(traj_raw_ardu[0, 1], traj_raw_ardu[0, 0], marker="o", color="darkorange", markersize=6, label="ArduPilot Start")
        ax.plot(traj_raw_ardu[-1, 1], traj_raw_ardu[-1, 0], marker="X", color="darkorange", markersize=6, label="ArduPilot End")

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("East (Y) [m]", fontsize=12)
    ax.set_ylabel("North (X) [m]", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    plt.tight_layout()
    if save_figure:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_full_trajectory_with_spans(t, features, feature_names, spans, title, save_path, line_color):
    if features is None or len(features) == 0:
        return

    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    axes_flat = axes.flatten()

    span_colors = {
        "takeoff": "orange",
        "landing": "mediumpurple",
        "straight_const": "mediumseagreen",
        "straight_accel": "dodgerblue",
        "straight_decel": "deepskyblue",
        "turn": "crimson",
        "unknown": "gray",
    }

    for i in range(10):
        ax = axes_flat[i]
        ax.plot(t, features[:, i], color=line_color, linewidth=1.5, zorder=2)
        ax.set_title(feature_names[i], fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5, zorder=1)

        if i == 6:
            ax.set_ylim(np.percentile(features[:, i], 2), np.percentile(features[:, i], 98))
        elif i == 7:
            ax.set_ylim(-1, np.percentile(features[:, i], 98))
        elif i == 8:
            ax.set_ylim(-0.1, np.percentile(features[:, i], 98))

        if spans:
            if spans.get("takeoff"):
                ax.axvspan(*spans["takeoff"], color=span_colors["takeoff"], alpha=0.20, zorder=0)
            if spans.get("landing"):
                ax.axvspan(*spans["landing"], color=span_colors["landing"], alpha=0.20, zorder=0)

            for s, e in spans.get("straight_const", []):
                ax.axvspan(s, e, color=span_colors["straight_const"], alpha=0.18, zorder=0)

            for s, e in spans.get("straight_accel", []):
                ax.axvspan(s, e, color=span_colors["straight_accel"], alpha=0.22, zorder=0)

            for s, e in spans.get("straight_decel", []):
                ax.axvspan(s, e, color=span_colors["straight_decel"], alpha=0.22, zorder=0)

            for s, e in spans.get("turn", []):
                ax.axvspan(s, e, color=span_colors["turn"], alpha=0.30, zorder=0)

            for s, e in spans.get("unknown", []):
                ax.axvspan(s, e, color=span_colors["unknown"], alpha=0.12, zorder=0)

    legend_patches = [
        mpatches.Patch(color=span_colors["takeoff"], alpha=0.20, label="Takeoff"),
        mpatches.Patch(color=span_colors["straight_const"], alpha=0.18, label="Straight Const"),
        mpatches.Patch(color=span_colors["straight_accel"], alpha=0.22, label="Straight Accel"),
        mpatches.Patch(color=span_colors["straight_decel"], alpha=0.22, label="Straight Decel"),
        mpatches.Patch(color=span_colors["turn"], alpha=0.30, label="Turn"),
        mpatches.Patch(color=span_colors["unknown"], alpha=0.12, label="Unknown"),
        mpatches.Patch(color=span_colors["landing"], alpha=0.20, label="Landing"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=4, fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_turn_segment_features(t, features, title, save_path, line_color):
    if features is None or len(features) == 0:
        return

    feature_names = [
        "Altitude (m)",
        "Heading (rad)",
        "Z-Axis Velocity (m/s)",
        "XY-Plane Speed (m/s)",
        "Z-Axis Acceleration (m/s²)",
        "XY-Plane Accel Norm (m/s²)",
        "Z-Axis Jerk (m/s³)",
        "XY-Plane Jerk Norm (m/s³)",
        "Curvature (1/m)",
        "Yaw Rate (rad/s)",
    ]

    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    axes_flat = axes.flatten()

    for i in range(10):
        ax = axes_flat[i]
        ax.plot(t, features[:, i], color=line_color, linewidth=2.0)
        ax.set_title(feature_names[i], fontsize=12, fontweight="bold")
        ax.set_xlabel("Absolute Time (s)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)

        if i == 6:
            ax.set_ylim(np.percentile(features[:, i], 2), np.percentile(features[:, i], 98))
        elif i == 7:
            ax.set_ylim(-1, np.percentile(features[:, i], 98))
        elif i == 8:
            ax.set_ylim(-0.1, np.percentile(features[:, i], 98))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
