#!/usr/bin/env python3
"""Plot SITL autopilot response for ArduPilot, PX4, and CogniPilot."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from .trajectory_loader import (
        AUTOPILOTS,
        AutopilotResponse,
        TimeSeries,
        align_to_vertical_motion_start,
        canonical_autopilot,
        load_response,
        load_run,
    )
except ImportError:
    from trajectory_loader import (
        AUTOPILOTS,
        AutopilotResponse,
        TimeSeries,
        align_to_vertical_motion_start,
        canonical_autopilot,
        load_response,
        load_run,
    )


COLORS = {
    "ardu": "tab:orange",
    "px4": "tab:blue",
    "cogni": "tab:green",
}

LABELS = {
    "ardu": "ArduPilot",
    "px4": "PX4",
    "cogni": "CogniPilot",
}

LINESTYLES = {
    0: "-",
    1: "--",
    2: ":",
}


def plot_3d_trajectory(responses: dict[str, AutopilotResponse], save_path: Path) -> None:
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for autopilot, response in responses.items():
        if response.position is None:
            continue
        pos = response.position.values
        color = COLORS.get(autopilot, None)
        label = LABELS.get(autopilot, autopilot)
        ax.plot(pos[:, 1], pos[:, 0], -pos[:, 2], color=color, linewidth=1.6, label=label)
        ax.scatter(pos[0, 1], pos[0, 0], -pos[0, 2], color=color, marker="o", s=35)
        ax.scatter(pos[-1, 1], pos[-1, 0], -pos[-1, 2], color=color, marker="x", s=45)

    ax.set_title("3D Trajectory")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="best")
    _save(fig, save_path)


def plot_vector_history(
    responses: dict[str, AutopilotResponse],
    attr: str,
    title: str,
    ylabel: str,
    save_path: Path,
    *,
    angle_degrees: bool = False,
    include_norm: bool = False,
) -> None:
    fig, axes = plt.subplots(4 if include_norm else 3, 1, figsize=(13, 10), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    component_labels = None
    for autopilot, response in responses.items():
        series = getattr(response, attr)
        if series is None:
            continue
        component_labels = series.labels
        values = np.rad2deg(series.values) if angle_degrees else series.values
        color = COLORS.get(autopilot, None)
        autopilot_label = LABELS.get(autopilot, autopilot)

        for idx in range(3):
            axes[idx].plot(
                series.t,
                values[:, idx],
                color=color,
                linestyle=LINESTYLES[idx],
                linewidth=1.25,
                label=autopilot_label,
            )

        if include_norm:
            norm = np.linalg.norm(values, axis=1)
            axes[3].plot(series.t, norm, color=color, linewidth=1.4, label=autopilot_label)

    fig.suptitle(title, fontsize=15, fontweight="bold")
    labels = component_labels or ("x", "y", "z")
    for idx in range(3):
        axes[idx].set_ylabel(f"{labels[idx]} {ylabel}")
        axes[idx].grid(True, linestyle="--", alpha=0.45)
        axes[idx].legend(loc="best")

    if include_norm:
        axes[3].set_ylabel(f"norm {ylabel}")
        axes[3].grid(True, linestyle="--", alpha=0.45)
        axes[3].legend(loc="best")

    axes[-1].set_xlabel("Time [s]")
    _save(fig, save_path)


def plot_all(
    responses: dict[str, AutopilotResponse],
    output_dir: Path,
    prefix: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = [
        output_dir / f"{prefix}_trajectory_3d.png",
        output_dir / f"{prefix}_position_history.png",
        output_dir / f"{prefix}_velocity_history.png",
        output_dir / f"{prefix}_attitude_history.png",
        output_dir / f"{prefix}_rate_history.png",
    ]

    plot_3d_trajectory(responses, saved[0])
    plot_vector_history(
        responses,
        "position",
        "Position Time History",
        "[m]",
        saved[1],
    )
    plot_vector_history(
        responses,
        "velocity",
        "Velocity Time History",
        "[m/s]",
        saved[2],
        include_norm=True,
    )
    plot_vector_history(
        responses,
        "attitude",
        "Attitude Time History",
        "[deg]",
        saved[3],
        angle_degrees=True,
    )
    plot_vector_history(
        responses,
        "rate",
        "Body Rate Time History",
        "[deg/s]",
        saved[4],
        angle_degrees=True,
    )
    return saved


def parse_autopilots(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(AUTOPILOTS)
    return [canonical_autopilot(item) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-dir", type=Path, default=Path("data/sitl_logs/run_000"))
    source.add_argument(
        "--log-file", type=Path,
        help="Read a specific ULog/BIN/CSV, including archived attempts; select one --autopilots value.",
    )
    parser.add_argument(
        "--autopilots",
        type=parse_autopilots,
        default=list(AUTOPILOTS),
        help="Comma-separated autopilots to plot: ardu,px4,cogni, or all.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/plotting"))
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to run directory name or log filename stem.")
    parser.add_argument(
        "--align",
        choices=("none", "vertical"),
        default="vertical",
        help="Time alignment mode.",
    )
    parser.add_argument(
        "--start-vertical-threshold",
        type=float,
        default=0.2,
        help="Absolute vertical velocity threshold used by --align vertical.",
    )
    args = parser.parse_args()
    if args.log_file is not None:
        if len(args.autopilots) != 1:
            parser.error("--log-file requires exactly one autopilot, e.g. --autopilots ardu")
        if not args.log_file.is_file():
            parser.error(f"Log file does not exist: {args.log_file}")
    return args


def main() -> None:
    args = parse_args()
    if args.log_file is not None:
        autopilot = args.autopilots[0]
        responses = {autopilot: load_response(args.log_file, autopilot)}
    else:
        responses = load_run(args.run_dir, args.autopilots)

    missing = [ap for ap in args.autopilots if ap not in responses]
    for ap in missing:
        print(f"[WARN] No {LABELS.get(ap, ap)} log found under {args.run_dir}")

    if args.align == "vertical":
        responses = {
            ap: align_to_vertical_motion_start(resp, args.start_vertical_threshold)
            for ap, resp in responses.items()
        }

    responses = {ap: resp for ap, resp in responses.items() if _has_any_plot_data(resp)}
    if not responses:
        raise RuntimeError(f"No plottable logs found in {args.log_file or args.run_dir}")

    prefix = args.prefix or (args.log_file.stem if args.log_file is not None else args.run_dir.name)
    saved = plot_all(responses, args.output_dir, prefix)

    print("[INFO] Loaded sources:")
    for ap, resp in responses.items():
        print(f"  {LABELS.get(ap, ap)}: {resp.source_file}")

    print("[INFO] Saved figures:")
    for path in saved:
        print(f"  {path}")


def _has_any_plot_data(response: AutopilotResponse) -> bool:
    return any(
        series is not None
        for series in (response.position, response.velocity, response.attitude, response.rate)
    )


def _save(fig: plt.Figure, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
