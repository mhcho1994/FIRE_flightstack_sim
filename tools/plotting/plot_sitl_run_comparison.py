#!/usr/bin/env python3
"""Compare one autopilot across SITL runs, with scenario wind labels.

Position and velocity are converted from NED to ENU, retaining local origins.
Attitude and body rates retain their logged axes and are displayed in degrees
and degrees/second. Vertical motion alignment is an approximate takeoff
reference, not a mission-start event.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Sequence
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

if __package__:
    from .trajectory_loader import (
        TimeSeries, align_to_vertical_motion_start, canonical_autopilot, load_run,
        resolve_log_directory, crop_to_mission_segment,
    )
else:
    from trajectory_loader import (
        TimeSeries, align_to_vertical_motion_start, canonical_autopilot, load_run,
        resolve_log_directory, crop_to_mission_segment,
    )


@dataclass(frozen=True)
class RunTrace:
    run_dir: Path
    label: str
    position: TimeSeries  # ENU, in the log's local frame
    velocity: TimeSeries | None = None  # ENU
    attitude: TimeSeries | None = None  # Logged roll/pitch/yaw, radians
    rate: TimeSeries | None = None  # Body p/q/r, radians/second


def _clean_series(series: TimeSeries) -> TimeSeries:
    valid = np.isfinite(series.t) & np.all(np.isfinite(series.values), axis=1)
    t, values = series.t[valid], series.values[valid]
    t, first = np.unique(t, return_index=True)
    if len(t) < 2:
        raise ValueError("Time series needs at least two finite, distinct timestamps")
    return TimeSeries(t, values[first], series.labels)


def _to_enu(series: TimeSeries) -> TimeSeries:
    series = _clean_series(series)
    values = series.values[:, [1, 0, 2]] * np.array([1.0, 1.0, -1.0])
    return TimeSeries(series.t, values, ("East", "North", "Up"))


def load_comparison_runs(
    run_dirs: Sequence[Path],
    autopilot: str,
    *,
    align: str = "vertical",
    vertical_threshold: float = 0.2,
    include_incomplete: bool = False,
    segment: str = "full",
) -> list[RunTrace]:
    """Load one autopilot per run; warn if settings other than wind differ."""
    autopilot = canonical_autopilot(autopilot)
    run_dirs = [Path(path) for path in run_dirs]
    if len(run_dirs) < 2 or len({path.resolve() for path in run_dirs}) != len(run_dirs):
        raise ValueError("Provide at least two distinct run directories")
    if segment not in ("full", "mission"):
        raise ValueError("segment must be full or mission")
    if align not in ("none", "vertical"):
        raise ValueError("align must be 'none' or 'vertical'")
    if not np.isfinite(vertical_threshold) or vertical_threshold <= 0:
        raise ValueError("vertical_threshold must be finite and positive")

    metadata = []
    baseline = None
    for path in run_dirs:
        logs_dir = resolve_log_directory(path, autopilot)
        result_path = logs_dir / "result.json"
        status = json.loads(result_path.read_text()).get("status") if result_path.exists() else None
        if status is not None and status != "success":
            message = f"{path}: {autopilot} result status is {status!r}"
            if not include_incomplete:
                raise ValueError(message + "; use --include-incomplete to inspect partial flights")
            warnings.warn(message + "; plotting available samples only", stacklevel=2)

        scenario_path = path / "scenario.yaml"
        if not scenario_path.is_file() and logs_dir == path:
            # Attempts use their archived session snapshot, not the live run.
            scenario_path = path.parent / "scenario.yaml"
        config = yaml.safe_load(scenario_path.read_text()) if scenario_path.exists() else {}
        config = config or {}
        common = dict(config.get("common", {}))
        mission = dict(common.get("scenario", {}))
        wind = mission.pop("wind_m_s", None)
        common["scenario"] = mission
        autopilot_settings = config.get("autopilots", {})
        scenario_key = {"ardu": "ardupilot", "cogni": "cognipilot"}.get(autopilot, autopilot)
        settings = (common, autopilot_settings.get(scenario_key, autopilot_settings.get(autopilot)))
        if not scenario_path.exists():
            warnings.warn(f"{path}: no scenario.yaml; wind and mission cannot be verified", stacklevel=2)
        elif baseline is None:
            baseline = settings
        elif settings != baseline:
            warnings.warn(f"{path}: settings other than wind differ from the first scenario", stacklevel=2)

        name = path.name if sum(p.name == path.name for p in run_dirs) == 1 else str(path)
        wind_text = ", ".join(f"{float(v):g}" for v in wind) if wind is not None else "unknown"
        label = f"{name} | wind ENU [{wind_text}] m/s"
        if status is not None and status != "success":
            label += f" | {status}"
        metadata.append((path, label))

    traces = []
    for path, label in metadata:
        response = load_run(path, [autopilot]).get(autopilot)
        if response is None or response.position is None:
            raise ValueError(f"{path}: no {autopilot} position log found")
        if segment == "mission":
            response = crop_to_mission_segment(response, resolve_log_directory(path, autopilot) / "wind_events.jsonl")
        elif align == "vertical":
            velocity = response.velocity
            if velocity is None or not np.any(
                np.isfinite(velocity.t) & np.isfinite(velocity.values[:, 2])
                & (np.abs(velocity.values[:, 2]) >= vertical_threshold)
            ):
                raise ValueError(f"{path}: no vertical-motion start found; use --align none")
            response = align_to_vertical_motion_start(response, vertical_threshold)
        traces.append(RunTrace(
            path, label, _to_enu(response.position),
            velocity=_to_enu(response.velocity) if response.velocity is not None else None,
            attitude=_clean_series(response.attitude) if response.attitude is not None else None,
            rate=_clean_series(response.rate) if response.rate is not None else None,
        ))
        print(f"[INFO] {label}: {response.source_file}")
    return traces


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_history(
    runs: Sequence[RunTrace],
    attr: str,
    title: str,
    unit: str,
    path: Path,
    time_label: str,
    *,
    angle_degrees: bool = False,
    include_norm: bool = False,
) -> None:
    fig, axes = plt.subplots(4 if include_norm else 3, 1, figsize=(13, 10), sharex=True)
    colors = plt.get_cmap("tab10")
    labels = {
        "position": ("East", "North", "Up"),
        "velocity": ("East", "North", "Up"),
        "attitude": ("roll", "pitch", "yaw"),
        "rate": ("p", "q", "r"),
    }[attr]
    for i, run in enumerate(runs):
        series = getattr(run, attr)
        if series is None:
            warnings.warn(f"{run.run_dir}: no {attr} data", stacklevel=2)
            continue
        values = np.rad2deg(series.values) if angle_degrees else series.values
        for axis in range(3):
            axes[axis].plot(series.t, values[:, axis], color=colors(i % 10),
                            linewidth=1.1, label=run.label)
        if include_norm:
            axes[3].plot(series.t, np.linalg.norm(values, axis=1), color=colors(i % 10),
                         linewidth=1.1, label=run.label)
    for axis, ax in enumerate(axes):
        ax.set_ylabel(f"{labels[axis] if axis < 3 else 'Speed'} [{unit}]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(time_label)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="lower center", fontsize=9)
    else:
        axes[0].text(0.5, 0.5, f"No {attr} data available", ha="center", transform=axes[0].transAxes)
    fig.suptitle(f"{title} | each run keeps its selected interval")
    _save(fig, path)


def plot_run_comparison(
    run_dirs: Sequence[Path],
    autopilot: str,
    output_dir: Path,
    *,
    prefix: str | None = None,
    align: str = "vertical",
    vertical_threshold: float = 0.2,
    include_incomplete: bool = False,
    segment: str = "full",
) -> list[Path]:
    """Save spatial overlays and position, velocity, attitude, and rate histories."""
    autopilot = canonical_autopilot(autopilot)
    runs = load_comparison_runs(
        run_dirs, autopilot, align=align, vertical_threshold=vertical_threshold,
        include_incomplete=include_incomplete, segment=segment,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or f"{autopilot}_" + "_vs_".join(run.run_dir.name for run in runs)
    paths = [output_dir / f"{prefix}_{suffix}.png" for suffix in (
        "trajectories", "position_history", "velocity_history", "attitude_history", "rate_history",
    )]
    colors = plt.get_cmap("tab10")
    time_label = "Time since mission segment start [s]" if segment == "mission" else (
        f"Time since first |vertical velocity| >= {vertical_threshold:g} m/s [s]"
        if align == "vertical" else "Time since first sample of each series [s]"
    )

    fig = plt.figure(figsize=(15, 7))
    xy = fig.add_subplot(121)
    xyz = fig.add_subplot(122, projection="3d")
    for i, run in enumerate(runs):
        points = run.position.values
        for ax, dims in ((xy, 2), (xyz, 3)):
            ax.plot(*points[:, :dims].T, color=colors(i % 10), linewidth=1.3, label=run.label)
            ax.scatter(*points[0, :dims], color=colors(i % 10), marker="o", s=30)
            ax.scatter(*points[-1, :dims], color=colors(i % 10), marker="x", s=40)
    xy.set_aspect("equal", adjustable="box")
    xy.set_title("Horizontal trajectory")
    xyz.set_title("3D trajectory")
    xyz.set_zlabel("Up [m]")
    extent = np.ptp(np.vstack([run.position.values for run in runs]), axis=0)
    xyz.set_box_aspect(np.maximum(extent, 1.0))
    for ax in (xy, xyz):
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.grid(True, alpha=0.3)
    handles, labels = xy.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=9)
    fig.suptitle(f"{autopilot.upper()} across runs | log local origins retained | o start, x end")
    _save(fig, paths[0])

    _plot_history(runs, "position", f"{autopilot.upper()} position histories", "m", paths[1], time_label)
    _plot_history(runs, "velocity", f"{autopilot.upper()} velocity histories", "m/s", paths[2], time_label,
                  include_norm=True)
    _plot_history(runs, "attitude", f"{autopilot.upper()} attitude histories", "deg", paths[3], time_label,
                  angle_degrees=True)
    _plot_history(runs, "rate", f"{autopilot.upper()} body rate histories", "deg/s", paths[4], time_label,
                  angle_degrees=True)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True,
                        help="Two or more run or archived attempt directories of the selected autopilot.")
    parser.add_argument("--autopilot", type=canonical_autopilot, default="px4")
    parser.add_argument("--output-dir", type=Path, default=Path("results/plotting/run_comparison"))
    parser.add_argument("--prefix")
    parser.add_argument("--segment", choices=("full", "mission"), default="full",
                        help="Mission uses wind_events.jsonl boundaries and overrides --align.")
    parser.add_argument("--align", choices=("none", "vertical"), default="vertical")
    parser.add_argument("--start-vertical-threshold", type=float, default=0.2)
    parser.add_argument("--include-incomplete", action="store_true",
                        help="Allow result.json states other than success; label partial flights.")
    args = parser.parse_args()
    try:
        paths = plot_run_comparison(
            args.run_dirs, args.autopilot, args.output_dir, prefix=args.prefix,
            align=args.align, vertical_threshold=args.start_vertical_threshold,
            include_incomplete=args.include_incomplete, segment=args.segment,
        )
    except ValueError as exc:
        parser.error(str(exc))
    for path in paths:
        print(f"[INFO] Saved {path}")


if __name__ == "__main__":
    main()
