#!/usr/bin/env python3
"""Plot SITL autopilot response for ArduPilot, PX4, and CogniPilot."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

# Support direct file execution as well as python -m tools.<package>.<module>.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.plotting.trajectory_loader import (
    AUTOPILOTS,
    AutopilotResponse,
    TimeSeries,
    align_to_vertical_motion_start,
    canonical_autopilot,
    load_response,
    load_run,
    crop_to_mission_segment, resolve_log_directory,
    find_source_file, quat_wxyz_to_euler,
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


@dataclass
class ComparisonTrace:
    response: AutopilotResponse
    parameters: dict[str, float]
    parameter_changes: list[dict]
    time_origin_boot_s: float
    source_topics: list[str]


def align_groundtruth(response: AutopilotResponse, threshold: float = 0.2) -> tuple[AutopilotResponse, float]:
    """Use one boot-clock origin for all signals, including asynchronous attitude.

    Require 0.3 s of ascent so a single startup velocity spike is not takeoff.
    Ground-truth positions retain their simulator origin; no spatial fitting.
    """
    velocity = response.velocity
    if velocity is None or velocity.boot_t is None or threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("Comparison requires boot timestamps and a positive ascent threshold")
    t = velocity.boot_t
    ascending = velocity.values[:, 2] < -threshold
    starts = np.flatnonzero(ascending & ~np.r_[False, ascending[:-1]])
    ends = np.flatnonzero(ascending & ~np.r_[ascending[1:], False])
    candidates = [s for s, e in zip(starts, ends) if t[e] - t[s] >= 0.3]
    if not candidates:
        raise ValueError(f"No sustained takeoff in {response.source_file}")
    origin = float(t[candidates[0]])

    def shift(series):
        if series is None:
            return None
        if series.boot_t is None:
            raise ValueError("All comparison signals must retain boot timestamps")
        valid = np.isfinite(series.boot_t) & np.all(np.isfinite(series.values), axis=1)
        times, indices = np.unique(series.boot_t[valid], return_index=True)
        values = series.values[valid][indices]
        keep = times >= origin - 1.0
        return TimeSeries(times[keep] - origin, values[keep], series.labels, times[keep])

    return AutopilotResponse(response.autopilot, response.source_file,
                             *(shift(getattr(response, key)) for key in
                               ("position", "velocity", "attitude", "rate"))), origin


def load_comparison_trace(path: Path, autopilot: str, threshold: float = 0.2) -> ComparisonTrace:
    """Load simulator truth on both stacks; never mix truth with EKF estimates."""
    parameters, changes = {}, []
    if autopilot == "px4" and path.suffix.lower() == ".ulg":
        from pyulog import ULog
        topics = ["vehicle_local_position_groundtruth", "vehicle_attitude_groundtruth"]
        ulog = ULog(str(path), message_name_filter_list=topics)
        data = {d.name: d.data for d in ulog.data_list if d.multi_id == 0}
        if any(topic not in data for topic in topics):
            raise ValueError(f"Missing PX4 ground-truth topics in {path}")
        loc, att = (data[topic] for topic in topics)
        t = loc["timestamp"] / 1e6
        position = TimeSeries.from_boot_time(t, np.column_stack([loc[k] for k in ("x", "y", "z")]), ("N", "E", "D"))
        velocity = TimeSeries.from_boot_time(t, np.column_stack([loc[k] for k in ("vx", "vy", "vz")]), ("VN", "VE", "VD"))
        attitude = TimeSeries.from_boot_time(att["timestamp"] / 1e6,
            quat_wxyz_to_euler(np.column_stack([att[f"q[{i}]"] for i in range(4)])), ("roll", "pitch", "yaw"))
        parameters = {k: float(v) for k, v in ulog.initial_parameters.items()}
        changes = [dict(boot_s=float(t) / 1e6, name=k, value=float(v)) for t, k, v in ulog.changed_parameters]
    elif autopilot == "ardu" and path.suffix.lower() == ".bin":
        from pymavlink import mavutil
        topics = ["SIM2", "SIM"]
        state, angles = [], []
        log = mavutil.mavlink_connection(str(path))
        try:
            while True:
                msg = log.recv_match(type=["SIM2", "SIM", "PARM"])
                if msg is None:
                    break
                if msg.get_type() == "SIM2":
                    state.append([msg.TimeUS / 1e6, msg.PN, msg.PE, msg.PD, msg.VN, msg.VE, msg.VD])
                elif msg.get_type() == "SIM":
                    angles.append([msg.TimeUS / 1e6, msg.Roll, msg.Pitch, msg.Yaw])
                else:
                    name = msg.Name.decode().rstrip("\x00") if isinstance(msg.Name, bytes) else str(msg.Name)
                    value = float(msg.Value)
                    if name not in parameters:
                        parameters[name] = value
                    elif parameters[name] != value:
                        changes.append(dict(boot_s=msg.TimeUS / 1e6, name=name, value=value))
        finally:
            log.close()
        if len(state) < 2 or len(angles) < 2:
            raise ValueError(f"Missing ArduPilot SIM2/SIM ground truth in {path}")
        state, angles = np.asarray(state), np.asarray(angles)
        position = TimeSeries.from_boot_time(state[:, 0], state[:, 1:4], ("N", "E", "D"))
        velocity = TimeSeries.from_boot_time(state[:, 0], state[:, 4:7], ("VN", "VE", "VD"))
        attitude = TimeSeries.from_boot_time(angles[:, 0], np.deg2rad(angles[:, 1:4]), ("roll", "pitch", "yaw"))
    else:
        raise ValueError("The comparison layout requires native PX4 ULog or ArduPilot BIN logs")
    response, origin = align_groundtruth(AutopilotResponse(autopilot, path, position, velocity, attitude), threshold)
    return ComparisonTrace(response, parameters, changes, origin, topics)


def acceptance_radius(autopilot: str, parameters: dict, mission: dict) -> tuple[str, float]:
    override = float(mission.get("waypoint_acceptance_radius_m", 1.0))
    # PX4 uses mission param2 when nonzero. ArduCopter uses its WP parameter.
    if autopilot == "px4" and override > 0:
        return "mission param2", override
    keys = [("NAV_ACC_RAD", 1.0)] if autopilot == "px4" else [("WP_RADIUS_M", 1.0), ("WPNAV_RADIUS", 0.01)]
    for key, scale in keys:
        if key in parameters:
            return key, float(parameters[key]) * scale
    raise ValueError(f"No logged acceptance-radius parameter for {autopilot}")


def waypoint_passages(position: TimeSeries, waypoints: np.ndarray) -> list[dict]:
    """Ordered nearest 3D passages, explicitly distinct from mission acceptance."""
    passages, start = [], int(np.searchsorted(position.t, 0.0))
    for waypoint in waypoints:
        if start >= len(position.t):
            raise ValueError("Insufficient trajectory after the preceding waypoint")
        delta = position.values[start:] - waypoint
        index = start + int(np.argmin(np.linalg.norm(delta, axis=1)))
        error = position.values[index] - waypoint
        passages.append(dict(time_s=float(position.t[index]), index=index,
                             horizontal_error_m=float(np.linalg.norm(error[:2])),
                             vertical_error_m=float(-error[2]),
                             distance_3d_m=float(np.linalg.norm(error))))
        start = index + 1
    return passages


def navigation_subtitle(scenario: dict, traces: dict[str, ComparisonTrace]) -> str:
    """Label speed-controlled experiments from their declared and logged values."""
    speed_labels = []
    has_overrides = False
    only_speed = True
    for ap, trace in traces.items():
        settings = scenario["autopilots"]["ardupilot" if ap == "ardu" else ap]
        flight = settings.get("flight_parameters") or {}
        gains = settings.get("controller_gains") or {}
        values = flight.get("values", {})
        has_overrides |= bool(values or gains)
        choices = [("MPC_XY_CRUISE", 1.)] if ap == "px4" else [("WP_SPD", 1.), ("WPNAV_SPEED", .01)]
        allowed = {name for name, _ in choices}
        gain_changes = gains and gains.get("values") != gains.get("baseline")
        only_speed &= not gain_changes and set(values) <= allowed
        for name, scale in choices:
            if name in trace.parameters:
                speed_labels.append(f"{LABELS[ap]} {trace.parameters[name] * scale:g} m/s")
                break
    if has_overrides and only_speed and len(speed_labels) == len(traces):
        return "Cruise speed: " + " | ".join(speed_labels)
    native = scenario["common"]["scenario"].get("navigation_defaults") == "native"
    return "Native navigation defaults" if native and not has_overrides else "Scenario navigation settings"


def plot_comparison(run_dir: Path, autopilots: list[str], scenario_path: Path,
                    output_dir: Path, prefix: str, *, zoom_waypoint: int = 2,
                    zoom_radius_m: float = 10., threshold: float = .2) -> list[Path]:
    """Publication overview: geometry, corner detail, speeds, roll/pitch, yaw."""
    scenario = yaml.safe_load(scenario_path.read_text())
    mission = scenario["common"]["scenario"]
    waypoints = np.array([p for c, p in zip(mission["command"], mission["waypoints_ned"]) if c == 16], dtype=float)
    if not 1 <= zoom_waypoint <= len(waypoints):
        raise ValueError("--zoom-waypoint must select a NAV_WAYPOINT (one based)")
    if not np.isfinite(zoom_radius_m) or zoom_radius_m <= 0:
        raise ValueError("--zoom-radius-m must be positive and finite")
    traces = {}
    if not autopilots:
        raise ValueError("Select at least one of px4,ardu for the comparison layout")
    for ap in autopilots:
        logs = resolve_log_directory(run_dir, ap)
        result = json.loads((logs / "result.json").read_text())
        if result.get("status") != "success":
            raise ValueError(f"{ap}: comparison requires a successful run")
        snapshot = yaml.safe_load((logs / "run_inputs/scenario.yaml").read_text())
        key = "ardupilot" if ap == "ardu" else ap
        if (snapshot["common"] != scenario["common"] or
                snapshot["autopilots"][key] != scenario["autopilots"][key]):
            raise ValueError(f"{ap}: scenario geometry/settings differ from the executed snapshot")
        source = find_source_file(run_dir, ap)
        if source is None:
            raise ValueError(f"No native log for {ap}")
        traces[ap] = load_comparison_trace(source, ap, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    wp = waypoints[:, [1, 0, 2]] * [1, 1, -1]
    center = wp[zoom_waypoint - 1, :2]
    metrics = dict(scenario=str(scenario_path.resolve()), state_source="simulator_ground_truth",
                   time_alignment="first ascent below -threshold VD sustained for 0.3 s; common boot clock for all signals",
                   vertical_threshold_m_s=threshold, waypoint_ned_m=waypoints.tolist(),
                   passage_definition="ordered nearest 3D point; not mission acceptance", autopilots={})
    with plt.rc_context({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig = plt.figure(figsize=(18, 10.5), layout="constrained")
        grid = fig.add_gridspec(2, 3)
        ax3d = fig.add_subplot(grid[0, 0], projection="3d")
        zoom = fig.add_subplot(grid[0, 1])
        speed = fig.add_subplot(grid[0, 2])
        vertical = fig.add_subplot(grid[1, 0])
        tilt = fig.add_subplot(grid[1, 1])
        heading = fig.add_subplot(grid[1, 2])
        ax3d.plot(*wp.T, color="0.25", ls="--", lw=1.3, label="Waypoint path", zorder=1)
        ax3d.scatter(*wp.T, color="black", marker="X", s=45, depthshade=False)
        zoom.plot(wp[:, 0] - center[0], wp[:, 1] - center[1], color="0.3", ls="--", lw=1.3)
        zoom.scatter([0], [0], color="black", marker="X", s=65, zorder=6)
        zoom.annotate(f"WP{zoom_waypoint}", (0, 0), xytext=(6, 8), textcoords="offset points", weight="bold")
        for i, (east, north, up) in enumerate(wp, 1):
            ax3d.text(east, north, up + 2, f"WP{i}\n{up:g} m", fontsize=12, weight="bold")
        end_times, corner_notes = [], []
        for ap, trace in traces.items():
            response = trace.response
            position, velocity, attitude = response.position, response.velocity, response.attitude
            color, label = COLORS[ap], LABELS[ap]
            enu = position.values[:, [1, 0, 2]] * [1, 1, -1]
            passages = waypoint_passages(position, waypoints)
            key, radius = acceptance_radius(ap, trace.parameters, mission)
            end_times.append(float(velocity.t[-1]))
            ax3d.plot(*enu.T, color=color, lw=1.5, label=label)
            zoom.plot(enu[:, 0] - center[0], enu[:, 1] - center[1], color=color, lw=2., label=label)
            circle_style = "-" if ap == "px4" else "--"
            zoom.add_patch(Circle((0, 0), radius, fill=False, color=color, ls=circle_style,
                                 lw=1.1 if ap == "px4" else 1.8,
                                 label=f"{label}: {key}"))
            selected = passages[zoom_waypoint - 1]
            corner_notes.append(f"{label}: {selected['horizontal_error_m']:.2f} m")
            point = enu[selected["index"], :2] - center
            zoom.scatter(*point, s=55, facecolors="white", edgecolors=color, linewidths=1.7, zorder=5)
            horizontal = np.linalg.norm(velocity.values[:, :2], axis=1)
            speed.plot(velocity.t, horizontal, color=color, lw=1.5, label=label)
            vertical.plot(velocity.t, -velocity.values[:, 2], color=color, lw=1.5, label=label)
            angles = np.rad2deg(attitude.values)
            for i, name in enumerate(("roll", "pitch")):
                tilt.plot(attitude.t, angles[:, i], color=color, ls=LINESTYLES[i], lw=1.2, label=f"{label} {name}")
            yaw = np.rad2deg(np.unwrap(attitude.values[:, 2]))
            yaw -= 360 * np.round(yaw[0] / 360)
            heading.plot(attitude.t, yaw, color=color, lw=1.5, label=label)
            # Use the same geometric passage times on every motion panel.
            # Interpolate only marker values on each signal's native time axis.
            for i, passage in enumerate(passages, 1):
                time_s = passage["time_s"]
                y = float(np.interp(time_s, velocity.t, horizontal))
                speed.scatter([time_s], [y], s=24, facecolors="white", edgecolors=color, zorder=4)
                speed.annotate(f"WP{i}", (time_s, y), xytext=(0, 8 if ap == "px4" or y < .6 else -15),
                               textcoords="offset points", ha="center", color=color, fontsize=11)
                offset = 9 if ap == "px4" else -16
                for ax, times, values in (
                        (vertical, velocity.t, -velocity.values[:, 2]),
                        (heading, attitude.t, yaw)):
                    value = float(np.interp(time_s, times, values))
                    ax.scatter([time_s], [value], s=28, facecolors="white",
                               edgecolors=color, linewidths=1.2, zorder=4)
                    ax.annotate(f"WP{i}", (time_s, value), xytext=(0, offset),
                                textcoords="offset points", ha="center", color=color,
                                fontsize=10, bbox=dict(facecolor="white", alpha=.75,
                                                     edgecolor="none", pad=.5))
                tilt_values = [float(np.interp(time_s, attitude.t, angles[:, k]))
                               for k in (0, 1)]
                tilt.scatter([time_s, time_s], tilt_values, s=28, facecolors="white",
                             edgecolors=color, linewidths=1.2, zorder=4)
                # One label per roll/pitch pair avoids duplicate overlapping text.
                anchor = max(tilt_values) if ap == "px4" else min(tilt_values)
                tilt.annotate(f"WP{i}", (time_s, anchor), xytext=(0, offset),
                              textcoords="offset points", ha="center", color=color,
                              fontsize=10, bbox=dict(facecolor="white", alpha=.75,
                                                   edgecolor="none", pad=.5))
            metrics["autopilots"][ap] = dict(source_file=str(response.source_file.resolve()),
                source_topics=trace.source_topics, time_origin_boot_s=trace.time_origin_boot_s,
                radius_parameter=key, radius_m=radius, waypoint_passages=passages,
                max_horizontal_speed_m_s=float(horizontal.max()),
                max_up_speed_m_s=float(-velocity.values[:, 2].min()),
                max_down_speed_m_s=float(velocity.values[:, 2].max()),
                max_abs_roll_deg=float(np.abs(angles[:, 0]).max()),
                max_abs_pitch_deg=float(np.abs(angles[:, 1]).max()),
                parameters=trace.parameters, parameter_changes=trace.parameter_changes)
            # Keep independent native timestamps; export without resampling.
            for attr in ("position", "velocity", "attitude"):
                series = getattr(response, attr)
                np.savetxt(output_dir / f"{prefix}_{ap}_{attr}.csv",
                           np.column_stack([series.t, series.boot_t, series.values]), delimiter=",",
                           header=",".join(["time_since_takeoff_s", "boot_s", *series.labels]), comments="")
        ax3d.set(title="(a) 3D trajectory", xlabel="East [m]", ylabel="North [m]", zlabel="Up [m]")
        ax3d.view_init(elev=65, azim=-125)
        ax3d.set_box_aspect((1.45, 1., .8))
        for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
            axis.set_major_locator(MaxNLocator(4))
        ax3d.legend(loc="upper left", fontsize=11)
        zoom.set(title=f"(b) Horizontal detail at WP{zoom_waypoint}", xlabel=f"East offset from WP{zoom_waypoint} [m]",
                 ylabel=f"North offset from WP{zoom_waypoint} [m]", xlim=(-zoom_radius_m, zoom_radius_m),
                 ylim=(-zoom_radius_m, zoom_radius_m), aspect="equal")
        zoom.legend(loc="lower left", fontsize=10)
        zoom.text(.98, .97, "Horizontal error at nearest 3D passage\n" + "  |  ".join(corner_notes),
                  transform=zoom.transAxes, ha="right", va="top", fontsize=10,
                  bbox=dict(facecolor="white", alpha=.85, edgecolor="none", pad=3))
        speed.set(title="(c) Horizontal speed", ylabel="Horizontal speed [m/s]")
        vertical.set(title="(d) Vertical velocity (up positive)", ylabel="Vertical velocity [m/s]")
        tilt.set(title="(e) Roll and pitch", ylabel="Angle [deg]")
        heading.set(title="(f) Yaw", ylabel="Unwrapped yaw [deg]")
        for ax in (zoom, speed, vertical, tilt, heading):
            ax.grid(True, color="0.88", lw=.6)
            ax.set_axisbelow(True)
        for ax in (speed, vertical, tilt, heading):
            ax.set_xlim(-1, max(end_times))
            ax.set_xlabel("Time since takeoff [s]")
            ax.legend(loc="best", ncol=2 if ax is tilt else 1)
        speed.set_ylim(bottom=-.3)
        vertical.axhline(0, color="0.5", lw=.7)
        title = " vs ".join(LABELS[ap] for ap in traces) + " | PX4Vision SITL"
        subtitle = navigation_subtitle(scenario, traces)
        metrics["navigation_label"] = subtitle
        fig.suptitle(title, fontsize=19, weight="bold")
        saved = [output_dir / f"{prefix}_comparison.{extension}" for extension in ("png", "pdf", "svg")]
        for path in saved:
            fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    report = output_dir / f"{prefix}_metrics.json"
    report.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return [*saved, report]


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
    parser.add_argument("--layout", choices=("separate", "comparison"), default="separate",
                        help="comparison: 2x3 ground-truth overview with native parameter radii")
    parser.add_argument("--scenario", type=Path, help="Defaults to RUN_DIR/scenario.yaml")
    parser.add_argument("--zoom-waypoint", type=int, default=2, help="One-based NAV_WAYPOINT number")
    parser.add_argument("--zoom-radius-m", type=float, default=10., help="Half-width of the waypoint detail")
    parser.add_argument("--segment", choices=("full", "mission"), default="full",
                        help="Use wind_events.jsonl mission boundaries; overrides --align.")
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
    if args.layout == "comparison" and (args.log_file is not None or args.segment != "full" or args.align != "vertical"):
        parser.error("--layout comparison requires --run-dir, --segment full and --align vertical")
    if args.log_file is not None:
        if len(args.autopilots) != 1:
            parser.error("--log-file requires exactly one autopilot, e.g. --autopilots ardu")
        if not args.log_file.is_file():
            parser.error(f"Log file does not exist: {args.log_file}")
    return args


def main() -> None:
    args = parse_args()
    if args.layout == "comparison":
        autopilots = [ap for ap in args.autopilots if ap in ("px4", "ardu")]
        saved = plot_comparison(args.run_dir, autopilots, args.scenario or args.run_dir / "scenario.yaml",
                                args.output_dir, args.prefix or args.run_dir.name,
                                zoom_waypoint=args.zoom_waypoint, zoom_radius_m=args.zoom_radius_m,
                                threshold=args.start_vertical_threshold)
        for path in saved:
            print(f"[INFO] Saved: {path}")
        return
    if args.log_file is not None:
        autopilot = args.autopilots[0]
        responses = {autopilot: load_response(args.log_file, autopilot)}
    else:
        responses = load_run(args.run_dir, args.autopilots)

    missing = [ap for ap in args.autopilots if ap not in responses]
    for ap in missing:
        print(f"[WARN] No {LABELS.get(ap, ap)} log found under {args.run_dir}")

    if args.segment == "mission":
        cropped = {}
        for ap, response in responses.items():
            if args.log_file is not None:
                events = next((parent / "wind_events.jsonl" for parent in response.source_file.resolve().parents
                               if (parent / "wind_events.jsonl").is_file()), None)
                if events is None:
                    raise ValueError("No wind_events.jsonl found beside the selected log or its parents")
            else:
                events = resolve_log_directory(args.run_dir, ap) / "wind_events.jsonl"
            cropped[ap] = crop_to_mission_segment(response, events)
        responses = cropped
    elif args.align == "vertical":
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
