#!/usr/bin/env python3
"""Identify Gazebo MulticopterMotorModel coefficients from a LiftDrag rotor.

The script launches a small, headless Gazebo test rig, commands a sweep of
rotor speeds and uniform winds, measures the rotor-joint wrench, and fits the
four coefficients used by gz-sim's MulticopterMotorModel:

  thrust               = motorConstant * omega**2
  yaw torque magnitude = momentConstant * thrust
  lateral force        = rotorDragCoefficient * abs(omega) * wind_perp
  lateral moment       = rollingMomentCoefficient * abs(omega) * wind_perp

The rotor is driven through a force/PID velocity controller. Step-up and
step-down joint-velocity measurements are fitted separately to delayed
first-order curves to estimate timeConstantUp and timeConstantDown. When an
ArduPilot source model is supplied, the matching rotor control's PID and force
limits are reused.

It deliberately identifies a lumped approximation over a user-selected test
envelope. Axial-inflow thrust changes in LiftDrag cannot be represented by the
stock MulticopterMotorModel and are therefore not included in the default
wind grid.

The fitted time constants represent the combined rotor inertia, aerodynamic
load, and selected PID / force limits. LiftDrag parameters alone do not encode
the electrical motor or ESC response, so these values are simulation-plant
matching parameters rather than physical motor constants.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import uuid
import xml.etree.ElementTree as ET

# Ubuntu's Gazebo Harmonic Python messages may be older than a user-installed
# protobuf runtime. This compatibility mode must be selected before importing
# any generated *_pb2 modules.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np


WORLD_NAME = "default_fire_rotor_test"
MODEL_NAME = "rotor_test"
WRENCH_TOPIC = "/rotor_test/wrench"
VELOCITY_TOPIC = f"/model/{MODEL_NAME}/joint/rotor_joint/cmd_vel"
WIND_TOPIC = f"/world/{WORLD_NAME}/wind/"
JOINT_STATE_TOPIC = f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state"


def controller_parameters_from_args(args) -> dict[str, float]:
    return {
        "p_gain": args.joint_p_gain,
        "i_gain": args.joint_i_gain,
        "d_gain": args.joint_d_gain,
        "i_max": args.joint_i_max,
        "i_min": args.joint_i_min,
        "cmd_max": args.joint_cmd_max,
        "cmd_min": args.joint_cmd_min,
    }


def extract_ardupilot_controller(
    source_model: ET.Element,
    source_joint_name: str,
    fallback: dict[str, float],
) -> tuple[dict[str, float], bool]:
    """Extract the ArduPilot velocity PID for the selected rotor joint."""

    for plugin in source_model.findall("plugin"):
        identity = " ".join(
            (plugin.get("name", ""), plugin.get("filename", ""))
        ).lower()
        if "ardupilot" not in identity:
            continue
        for control in plugin.findall("control"):
            controlled_joint = (control.findtext("jointName") or "").strip()
            if not (
                controlled_joint == source_joint_name
                or controlled_joint.endswith(f"::{source_joint_name}")
            ):
                continue
            if (control.findtext("type") or "VELOCITY").strip().upper() != "VELOCITY":
                continue
            result = dict(fallback)
            for name in result:
                value = control.findtext(name)
                if value is not None:
                    result[name] = float(value)
            return result, True
    return dict(fallback), False


def parse_number_grid(text: str) -> list[float]:
    """Parse comma values or inclusive start:stop:step notation."""

    text = text.strip()
    if ":" not in text:
        values = [float(item) for item in text.split(",") if item.strip()]
    else:
        parts = [float(item) for item in text.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "range must be start:stop:step, for example 200:1000:100"
            )
        start, stop, step = parts
        if step <= 0 or stop < start:
            raise argparse.ArgumentTypeError(
                "range requires stop >= start and step > 0"
            )
        count = int(math.floor((stop - start) / step + 1e-12)) + 1
        values = [start + index * step for index in range(count)]
        if values[-1] < stop - 1e-9:
            values.append(stop)

    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("all rotor speeds must be equal to or greater than zero")
    return values


def parse_wind_grid(text: str) -> list[tuple[float, float, float]]:
    """Parse semicolon-separated x,y[,z] wind vectors."""

    winds: list[tuple[float, float, float]] = []
    for vector_text in text.split(";"):
        if not vector_text.strip():
            continue
        values = [float(item) for item in vector_text.split(",")]
        if len(values) == 2:
            values.append(0.0)
        if len(values) != 3:
            raise argparse.ArgumentTypeError(
                "each wind must be x,y or x,y,z; separate winds with ';'"
            )
        winds.append((values[0], values[1], values[2]))

    if not winds:
        raise argparse.ArgumentTypeError("at least one wind vector is required")
    if not any(math.hypot(wind[0], wind[1]) < 1e-12 for wind in winds):
        winds.insert(0, (0.0, 0.0, 0.0))
    return winds


def lift_drag_parameter_blocks(
    plugins: list[ET.Element],
) -> list[dict[str, object]]:
    """Convert LiftDrag XML elements into JSON-serializable metadata."""

    numeric_names = (
        "a0",
        "alpha_stall",
        "cla",
        "cda",
        "cma",
        "cla_stall",
        "cda_stall",
        "cma_stall",
        "area",
        "air_density",
    )
    vector_names = ("cp", "forward", "upward")
    result: list[dict[str, object]] = []
    for plugin in plugins:
        values: dict[str, object] = {
            "link_name": plugin.findtext("link_name", ""),
        }
        for name in numeric_names:
            text = plugin.findtext(name)
            if text is not None:
                values[name] = float(text)
        for name in vector_names:
            text = plugin.findtext(name)
            if text is not None:
                values[name] = [float(value) for value in text.split()]
        result.append(values)
    return result


def find_source_model(root: ET.Element, source: Path) -> ET.Element:
    model = root.find("model")
    if model is None:
        raise ValueError(f"no top-level <model> found in {source}")
    return model


def build_model_element(
    inertial: ET.Element,
    joint_axis: ET.Element,
    lift_drag_plugins: list[ET.Element],
    controller_parameters: dict[str, float],
) -> ET.Element:
    """Build the fixed, single-rotor identification fixture."""

    model = ET.Element("model", {"name": MODEL_NAME})
    ET.SubElement(model, "static").text = "false"
    ET.SubElement(model, "self_collide").text = "false"

    base_link = ET.SubElement(model, "link", {"name": "base_link"})
    base_inertial = ET.SubElement(base_link, "inertial")
    ET.SubElement(base_inertial, "mass").text = "1.0"
    base_inertia = ET.SubElement(base_inertial, "inertia")
    for name, value in (("ixx", "0.01"), ("iyy", "0.01"), ("izz", "0.01")):
        ET.SubElement(base_inertia, name).text = value
    ET.SubElement(base_link, "gravity").text = "false"

    world_joint = ET.SubElement(
        model, "joint", {"name": "world_fixed", "type": "fixed"}
    )
    ET.SubElement(world_joint, "parent").text = "world"
    ET.SubElement(world_joint, "child").text = "base_link"

    rotor_link = ET.SubElement(model, "link", {"name": "rotor"})
    rotor_link.append(copy.deepcopy(inertial))
    ET.SubElement(rotor_link, "gravity").text = "false"

    rotor_joint = ET.SubElement(
        model, "joint", {"name": "rotor_joint", "type": "revolute"}
    )
    ET.SubElement(rotor_joint, "parent").text = "base_link"
    ET.SubElement(rotor_joint, "child").text = "rotor"
    rotor_joint.append(copy.deepcopy(joint_axis))
    sensor = ET.SubElement(
        rotor_joint, "sensor", {"name": "rotor_wrench", "type": "force_torque"}
    )
    ET.SubElement(sensor, "topic").text = WRENCH_TOPIC
    ET.SubElement(sensor, "always_on").text = "true"
    ET.SubElement(sensor, "update_rate").text = "500"
    force_torque = ET.SubElement(sensor, "force_torque")
    ET.SubElement(force_torque, "frame").text = "parent"
    ET.SubElement(force_torque, "measure_direction").text = "child_to_parent"

    controller = ET.SubElement(
        model,
        "plugin",
        {
            "filename": "gz-sim-joint-controller-system",
            "name": "gz::sim::systems::JointController",
        },
    )
    ET.SubElement(controller, "joint_name").text = "rotor_joint"
    ET.SubElement(controller, "initial_velocity").text = "0"
    ET.SubElement(controller, "use_force_commands").text = "true"
    for name, value in controller_parameters.items():
        ET.SubElement(controller, name).text = f"{value:.12g}"

    ET.SubElement(
        model,
        "plugin",
        {
            "filename": "gz-sim-joint-state-publisher-system",
            "name": "gz::sim::systems::JointStatePublisher",
        },
    )

    for source_plugin in lift_drag_plugins:
        plugin = copy.deepcopy(source_plugin)
        link_name = plugin.find("link_name")
        if link_name is None:
            link_name = ET.SubElement(plugin, "link_name")
        link_name.text = "rotor"
        model.append(plugin)
    return model


def make_rotor_model_from_source(
    source: Path, rotor_link_name: str, args
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    """Extract one rotor's inertia, joint axis and LiftDrag plugins."""

    if not source.is_file():
        raise FileNotFoundError(source)
    source_model = find_source_model(ET.parse(source).getroot(), source)
    source_link = next(
        (
            link
            for link in source_model.findall("link")
            if link.get("name") == rotor_link_name
        ),
        None,
    )
    if source_link is None:
        raise ValueError(f"link {rotor_link_name!r} not found in {source}")
    inertial = source_link.find("inertial")
    if inertial is None:
        raise ValueError(f"link {rotor_link_name!r} has no <inertial>")

    source_joint = next(
        (
            joint
            for joint in source_model.findall("joint")
            if joint.findtext("child") == rotor_link_name
        ),
        None,
    )
    if source_joint is None or source_joint.find("axis") is None:
        raise ValueError(f"no revolute joint axis found for {rotor_link_name!r}")

    plugins = [
        plugin
        for plugin in source_model.findall("plugin")
        if "lift-drag" in plugin.get("filename", "")
        and plugin.findtext("link_name") == rotor_link_name
    ]
    if not plugins:
        raise ValueError(
            f"no LiftDrag plugins targeting {rotor_link_name!r} in {source}"
        )

    source_joint_name = source_joint.get("name", "")
    controller_parameters, extracted = extract_ardupilot_controller(
        source_model,
        source_joint_name,
        controller_parameters_from_args(args),
    )
    model = build_model_element(
        inertial, source_joint.find("axis"), plugins, controller_parameters
    )
    ET.indent(model, space="  ")
    controller_metadata: dict[str, object] = {
        **controller_parameters,
        "source": "ArduPilotPlugin" if extracted else "command-line/default PID",
        "source_joint": source_joint_name,
    }
    return (
        ET.tostring(model, encoding="unicode"),
        lift_drag_parameter_blocks(plugins),
        controller_metadata,
    )


def vector_text(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def make_rotor_model_from_parameters(
    args,
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    """Create a symmetric blade-element rotor from command-line parameters."""

    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "mass").text = f"{args.rotor_mass:.12g}"
    inertia = ET.SubElement(inertial, "inertia")
    for name, value in (
        ("ixx", args.rotor_ixx),
        ("iyy", args.rotor_iyy),
        ("izz", args.rotor_izz),
    ):
        ET.SubElement(inertia, name).text = f"{value:.12g}"

    axis = ET.Element("axis")
    ET.SubElement(axis, "xyz").text = "0 0 1"
    limit = ET.SubElement(axis, "limit")
    ET.SubElement(limit, "lower").text = "-1e16"
    ET.SubElement(limit, "upper").text = "1e16"

    tangent_sign = 1.0 if args.turning_direction == "ccw" else -1.0
    plugins: list[ET.Element] = []
    for blade in range(args.blade_count):
        azimuth = 2.0 * math.pi * blade / args.blade_count
        radial = (math.cos(azimuth), math.sin(azimuth), 0.0)
        tangent = (
            tangent_sign * -math.sin(azimuth),
            tangent_sign * math.cos(azimuth),
            0.0,
        )
        plugin = ET.Element(
            "plugin",
            {
                "filename": "gz-sim-lift-drag-system",
                "name": "gz::sim::systems::LiftDrag",
            },
        )
        scalar_values = (
            ("a0", args.a0),
            ("alpha_stall", args.alpha_stall),
            ("cla", args.cla),
            ("cda", args.cda),
            ("cma", args.cma),
            ("cla_stall", args.cla_stall),
            ("cda_stall", args.cda_stall),
            ("cma_stall", args.cma_stall),
            ("area", args.area),
            ("air_density", args.air_density),
        )
        for name, value in scalar_values:
            ET.SubElement(plugin, name).text = f"{value:.12g}"
        ET.SubElement(plugin, "cp").text = vector_text(
            tuple(args.radius * component for component in radial)
        )
        ET.SubElement(plugin, "forward").text = vector_text(tangent)
        ET.SubElement(plugin, "upward").text = "0 0 1"
        ET.SubElement(plugin, "link_name").text = "rotor"
        plugins.append(plugin)

    controller_parameters = controller_parameters_from_args(args)
    model = build_model_element(inertial, axis, plugins, controller_parameters)
    ET.indent(model, space="  ")
    return (
        ET.tostring(model, encoding="unicode"),
        lift_drag_parameter_blocks(plugins),
        {**controller_parameters, "source": "command-line/default PID"},
    )


def rotor_model_document(model_xml: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sdf version="1.9">
{textwrap.indent(model_xml, "  ")}
</sdf>
"""


def make_world(model_xml: str) -> str:
    """Return a self-contained test world with the rotor model inline."""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sdf version="1.9">
  <world name="{WORLD_NAME}">
    <physics name="identification" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <gravity>0 0 0</gravity>
    <wind>
      <linear_velocity>0 0 0</linear_velocity>
    </wind>

    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-forcetorque-system"
            name="gz::sim::systems::ForceTorque"/>
    <!-- WindEffects owns /world/default_fire_rotor_test/wind/. The test links do not set
         enable_wind, so it updates the Wind component used by LiftDrag
         without adding a second, area-based wind force to the rig. -->
    <plugin filename="gz-sim-wind-effects-system"
            name="gz::sim::systems::WindEffects">
      <force_approximation_scaling_factor>0</force_approximation_scaling_factor>
      <!-- Keep command transitions short relative to the settling window. -->
      <horizontal>
        <magnitude><time_for_rise>0.01</time_for_rise></magnitude>
        <direction><time_for_rise>0.01</time_for_rise></direction>
      </horizontal>
      <vertical><time_for_rise>0.01</time_for_rise></vertical>
    </plugin>

{textwrap.indent(model_xml, "    ")}
  </world>
</sdf>
"""


class WrenchCollector:
    """Thread-safe storage for gz.msgs.Wrench callbacks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: list[tuple[float, np.ndarray]] = []

    def callback(self, message) -> None:  # Gazebo supplies the protobuf type.
        wrench = np.array(
            [
                message.force.x,
                message.force.y,
                message.force.z,
                message.torque.x,
                message.torque.y,
                message.torque.z,
            ],
            dtype=float,
        )
        with self._lock:
            self._samples.append((time.monotonic(), wrench))

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def snapshot(self) -> list[tuple[float, np.ndarray]]:
        with self._lock:
            return [(stamp, wrench.copy()) for stamp, wrench in self._samples]

    def count(self) -> int:
        with self._lock:
            return len(self._samples)


class JointVelocityCollector:
    """Thread-safe storage for the rotor velocity in joint-state messages."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: list[tuple[float, float]] = []

    def callback(self, message) -> None:  # Gazebo supplies gz.msgs.Model.
        for joint in message.joint:
            if joint.name == "rotor_joint" or joint.name.endswith("::rotor_joint"):
                with self._lock:
                    self._samples.append(
                        (time.monotonic(), float(joint.axis1.velocity))
                    )
                return

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def snapshot(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._samples)

    def count(self) -> int:
        with self._lock:
            return len(self._samples)


def wait_until(predicate, timeout: float, description: str, process=None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Gazebo exited with status {process.returncode} while waiting for "
                f"{description}"
            )
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {description}")


def publish_velocity(publisher, message_type, omega: float) -> None:
    message = message_type()
    message.data = float(omega)
    publisher.publish(message)


def publish_wind(publisher, message_type, wind: tuple[float, float, float]) -> None:
    message = message_type()
    message.linear_velocity.x = wind[0]
    message.linear_velocity.y = wind[1]
    message.linear_velocity.z = wind[2]
    message.enable_wind = True
    publisher.publish(message)


def average_wrench(
    samples: list[tuple[float, np.ndarray]], minimum_samples: int
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(samples) < minimum_samples:
        raise RuntimeError(
            f"received only {len(samples)} wrench samples; expected at least "
            f"{minimum_samples}"
        )
    values = np.stack([sample[1] for sample in samples])
    return values.mean(axis=0), values.std(axis=0, ddof=1), len(values)


def fit_nonnegative_scalar(design: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.dot(design, design))
    if denominator <= 1e-18:
        return 0.0
    return max(0.0, float(np.dot(design, target)) / denominator)


def vector_fit_nonnegative(design: np.ndarray, target: np.ndarray) -> float:
    """Fit y = coefficient*x for vector observations."""

    denominator = float(np.sum(design * design))
    if denominator <= 1e-18:
        return 0.0
    return max(0.0, float(np.sum(design * target)) / denominator)


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def fit_first_order_step(
    samples: list[tuple[float, float]],
    command_time: float,
    from_omega: float,
    to_omega: float,
) -> dict[str, object]:
    """Fit a delayed first-order response using a dependency-free grid search."""

    if len(samples) < 10:
        raise RuntimeError("too few joint-velocity samples for dynamic fitting")

    times = np.asarray([stamp - command_time for stamp, _ in samples], dtype=float)
    velocities = np.abs(np.asarray([velocity for _, velocity in samples], dtype=float))
    order = np.argsort(times)
    times = times[order]
    velocities = velocities[order]

    before = velocities[times < 0]
    after_mask = times >= 0
    after_times = times[after_mask]
    after_velocities = velocities[after_mask]
    if len(after_times) < 8:
        raise RuntimeError("too few post-command samples for dynamic fitting")

    initial = float(np.median(before)) if len(before) else float(from_omega)
    tail_count = max(5, len(after_velocities) // 5)
    final = float(np.median(after_velocities[-tail_count:]))
    amplitude = final - initial
    if abs(amplitude) < max(1.0, 0.01 * max(abs(from_omega), abs(to_omega))):
        raise RuntimeError(
            "joint velocity did not change enough during the dynamic step"
        )

    positive_deltas = np.diff(after_times)
    positive_deltas = positive_deltas[positive_deltas > 0]
    sample_period = (
        float(np.median(positive_deltas)) if len(positive_deltas) else 0.001
    )
    duration = max(float(after_times[-1]), sample_period)
    tau_candidates = np.geomspace(
        max(sample_period * 0.1, 1e-5), max(duration * 2.0, sample_period), 500
    )
    delay_candidates = np.linspace(0.0, min(0.05, duration * 0.25), 80)

    best_error = math.inf
    best_tau = 0.0
    best_delay = 0.0
    best_prediction = np.empty_like(after_velocities)
    for delay in delay_candidates:
        elapsed = np.maximum(after_times - delay, 0.0)
        for tau in tau_candidates:
            prediction = final + (initial - final) * np.exp(-elapsed / tau)
            error = float(np.mean(np.square(after_velocities - prediction)))
            if error < best_error:
                best_error = error
                best_tau = float(tau)
                best_delay = float(delay)
                best_prediction = prediction

    total_variance = float(
        np.sum(np.square(after_velocities - after_velocities.mean()))
    )
    residual_sum = float(np.sum(np.square(after_velocities - best_prediction)))
    r_squared = 1.0 - residual_sum / total_variance if total_variance > 1e-18 else 0.0
    sample_rows = [
        {
            "time_s": float(t),
            "omega_rad_s": float(measured),
            "fit_omega_rad_s": float(fitted),
        }
        for t, measured, fitted in zip(after_times, after_velocities, best_prediction)
    ]
    return {
        "from_omega_rad_s": float(from_omega),
        "to_omega_rad_s": float(to_omega),
        "initial_measured_omega_rad_s": initial,
        "final_measured_omega_rad_s": final,
        "time_constant_s": best_tau,
        "fit_delay_s": best_delay,
        "rmse_rad_s": math.sqrt(best_error),
        "r_squared": r_squared,
        "sample_period_s": sample_period,
        "sample_count": len(sample_rows),
        "samples": sample_rows,
    }


def identify_coefficients(
    cases: list[dict[str, object]], turning_direction: str
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    omega = np.asarray([case["omega_rad_s"] for case in cases], dtype=float)
    wind = np.asarray([case["wind_m_s"] for case in cases], dtype=float)
    raw_force = np.asarray([case["force_N"] for case in cases], dtype=float)
    raw_torque = np.asarray([case["torque_Nm"] for case in cases], dtype=float)

    hover_mask = np.linalg.norm(wind[:, :2], axis=1) < 1e-9
    if not np.any(hover_mask):
        raise RuntimeError("the sweep needs at least one zero-crosswind case")

    hover_force_z = raw_force[hover_mask, 2]
    force_sign = 1 if float(np.median(hover_force_z)) >= 0 else -1
    force = force_sign * raw_force

    # Aerodynamic reaction torque opposes rotation: -Z for CCW and +Z for CW.
    desired_yaw_sign = -1 if turning_direction == "ccw" else 1
    hover_torque_z = raw_torque[hover_mask, 2]
    raw_yaw_sign = 1 if float(np.median(hover_torque_z)) >= 0 else -1
    torque_sign = desired_yaw_sign * raw_yaw_sign
    torque = torque_sign * raw_torque

    omega2_hover = np.square(omega[hover_mask])
    motor_constant = fit_nonnegative_scalar(
        omega2_hover, force[hover_mask, 2]
    )
    yaw_magnitude = desired_yaw_sign * torque[hover_mask, 2]
    torque_constant = fit_nonnegative_scalar(omega2_hover, yaw_magnitude)
    moment_constant = (
        torque_constant / motor_constant if motor_constant > 1e-18 else 0.0
    )

    crosswind_mask = np.linalg.norm(wind[:, :2], axis=1) > 1e-9
    lateral_design = (
        np.abs(omega[crosswind_mask, None]) * wind[crosswind_mask, :2]
    )
    rotor_drag = vector_fit_nonnegative(
        lateral_design, force[crosswind_mask, :2]
    )
    rolling_moment = vector_fit_nonnegative(
        lateral_design, torque[crosswind_mask, :2]
    )

    predicted_thrust = motor_constant * np.square(omega[hover_mask])
    predicted_yaw = desired_yaw_sign * torque_constant * np.square(
        omega[hover_mask]
    )
    predicted_lateral_force = rotor_drag * lateral_design
    predicted_lateral_moment = rolling_moment * lateral_design

    errors = {
        "thrust_rmse_N": rmse(force[hover_mask, 2] - predicted_thrust),
        "yaw_torque_rmse_Nm": rmse(
            torque[hover_mask, 2] - predicted_yaw
        ),
        "lateral_force_rmse_N": rmse(
            force[crosswind_mask, :2] - predicted_lateral_force
        ),
        "lateral_moment_rmse_Nm": rmse(
            torque[crosswind_mask, :2] - predicted_lateral_moment
        ),
    }
    coefficients = {
        "motorConstant": motor_constant,
        "momentConstant": moment_constant,
        "rotorDragCoefficient": rotor_drag,
        "rollingMomentCoefficient": rolling_moment,
    }
    signs = {
        "force_sensor_to_aero": force_sign,
        "torque_sensor_to_aero": torque_sign,
    }
    return coefficients, errors, signs


def plugin_xml(
    coefficients: dict[str, float],
    max_rot_velocity: float,
    turning_direction: str,
    time_constant_up: float,
    time_constant_down: float,
) -> str:
    return f"""<plugin filename="gz-sim-multicopter-motor-model-system"
        name="gz::sim::systems::MulticopterMotorModel">
  <jointName>rotor_joint</jointName>
  <linkName>rotor</linkName>
  <turningDirection>{turning_direction}</turningDirection>
  <timeConstantUp>{time_constant_up:.9g}</timeConstantUp>
  <timeConstantDown>{time_constant_down:.9g}</timeConstantDown>
  <maxRotVelocity>{max_rot_velocity:.9g}</maxRotVelocity>
  <motorConstant>{coefficients['motorConstant']:.9g}</motorConstant>
  <momentConstant>{coefficients['momentConstant']:.9g}</momentConstant>
  <commandSubTopic>command/motor_speed</commandSubTopic>
  <actuator_number>0</actuator_number>
  <rotorDragCoefficient>{coefficients['rotorDragCoefficient']:.9g}</rotorDragCoefficient>
  <rollingMomentCoefficient>{coefficients['rollingMomentCoefficient']:.9g}</rollingMomentCoefficient>
  <rotorVelocitySlowdownSim>10</rotorVelocitySlowdownSim>
  <motorType>velocity</motorType>
</plugin>"""


def write_case_csv(path: Path, cases: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "omega_rad_s",
        "wind_x_m_s",
        "wind_y_m_s",
        "wind_z_m_s",
        "force_x_N",
        "force_y_N",
        "force_z_N",
        "torque_x_Nm",
        "torque_y_Nm",
        "torque_z_Nm",
        "sample_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            wind = case["wind_m_s"]
            force = case["force_N"]
            torque = case["torque_Nm"]
            writer.writerow(
                {
                    "omega_rad_s": case["omega_rad_s"],
                    "wind_x_m_s": wind[0],
                    "wind_y_m_s": wind[1],
                    "wind_z_m_s": wind[2],
                    "force_x_N": force[0],
                    "force_y_N": force[1],
                    "force_z_N": force[2],
                    "torque_x_Nm": torque[0],
                    "torque_y_Nm": torque[1],
                    "torque_z_Nm": torque[2],
                    "sample_count": case["sample_count"],
                }
            )


def write_dynamic_csv(path: Path, dynamic_response: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["step", "time_s", "omega_rad_s", "fit_omega_rad_s"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step_name in ("up", "down"):
            step = dynamic_response[step_name]
            for sample in step["samples"]:
                writer.writerow({"step": step_name, **sample})


def write_fit_plot(
    path: Path,
    cases: list[dict[str, object]],
    coefficients: dict[str, float],
    signs: dict[str, int],
    turning_direction: str,
    dynamic_response: dict[str, object],
) -> None:
    """Plot Gazebo measurements as points and the fitted model as lines."""

    # Use a writable cache in managed / headless environments.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fire-matplotlib")
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plots; install python3-matplotlib or "
            "run with --no-plot"
        ) from exc

    omega = np.asarray([case["omega_rad_s"] for case in cases], dtype=float)
    wind = np.asarray([case["wind_m_s"] for case in cases], dtype=float)
    force = signs["force_sensor_to_aero"] * np.asarray(
        [case["force_N"] for case in cases], dtype=float
    )
    torque = signs["torque_sensor_to_aero"] * np.asarray(
        [case["torque_Nm"] for case in cases], dtype=float
    )
    hover_mask = np.linalg.norm(wind[:, :2], axis=1) < 1e-9
    crosswind_mask = ~hover_mask
    desired_yaw_sign = -1 if turning_direction == "ccw" else 1

    figure, axes = plt.subplots(3, 2, figsize=(12, 12))
    thrust_axis, yaw_axis, force_axis, moment_axis, up_axis, down_axis = axes.flat

    hover_order = np.argsort(omega[hover_mask])
    hover_omega = omega[hover_mask][hover_order]
    hover_thrust = force[hover_mask, 2][hover_order]
    hover_yaw_magnitude = (
        desired_yaw_sign * torque[hover_mask, 2][hover_order]
    )
    omega_line = np.linspace(hover_omega.min(), hover_omega.max(), 300)

    thrust_axis.scatter(
        hover_omega,
        hover_thrust,
        color="tab:blue",
        marker="o",
        label="Gazebo LiftDrag",
        zorder=3,
    )
    thrust_axis.plot(
        omega_line,
        coefficients["motorConstant"] * np.square(omega_line),
        color="tab:orange",
        linewidth=2,
        label="MulticopterMotorModel fit",
    )
    thrust_axis.set_title("Axial thrust")
    thrust_axis.set_ylabel("Thrust [N]")

    yaw_axis.scatter(
        hover_omega,
        hover_yaw_magnitude,
        color="tab:blue",
        marker="o",
        label="Gazebo LiftDrag",
        zorder=3,
    )
    yaw_axis.plot(
        omega_line,
        coefficients["motorConstant"]
        * coefficients["momentConstant"]
        * np.square(omega_line),
        color="tab:orange",
        linewidth=2,
        label="MulticopterMotorModel fit",
    )
    yaw_axis.set_title("Yaw reaction torque magnitude")
    yaw_axis.set_ylabel("Torque [N m]")

    wind_groups = sorted(
        {
            (float(vector[0]), float(vector[1]))
            for vector in wind[crosswind_mask]
        }
    )
    if wind_groups:
        colors = plt.cm.tab10(np.linspace(0, 1, len(wind_groups)))
        for color, wind_xy in zip(colors, wind_groups):
            wind_vector = np.asarray(wind_xy)
            wind_speed = float(np.linalg.norm(wind_vector))
            group_mask = np.all(
                np.isclose(wind[:, :2], wind_vector, atol=1e-12), axis=1
            )
            group_order = np.argsort(omega[group_mask])
            group_omega = omega[group_mask][group_order]
            direction = wind_vector / wind_speed
            aligned_force = (
                force[group_mask, :2][group_order] @ direction
            )
            aligned_moment = (
                torque[group_mask, :2][group_order] @ direction
            )
            group_line = np.linspace(group_omega.min(), group_omega.max(), 300)
            label = f"wind=({wind_xy[0]:g}, {wind_xy[1]:g}) m/s"

            force_axis.scatter(
                group_omega, aligned_force, color=color, marker="o", zorder=3
            )
            force_axis.plot(
                group_line,
                coefficients["rotorDragCoefficient"] * wind_speed * group_line,
                color=color,
                linewidth=2,
                label=label,
            )
            moment_axis.scatter(
                group_omega, aligned_moment, color=color, marker="o", zorder=3
            )
            moment_axis.plot(
                group_line,
                coefficients["rollingMomentCoefficient"]
                * wind_speed
                * group_line,
                color=color,
                linewidth=2,
                label=label,
            )
    else:
        for axis in (force_axis, moment_axis):
            axis.text(
                0.5,
                0.5,
                "No crosswind cases",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    force_axis.set_title("Lateral force along wind\n(points: Gazebo, lines: fit)")
    force_axis.set_ylabel("Force [N]")
    moment_axis.set_title(
        "Lateral moment along wind\n(points: Gazebo, lines: fit)"
    )
    moment_axis.set_ylabel("Moment [N m]")

    for axis in (thrust_axis, yaw_axis, force_axis, moment_axis):
        axis.set_xlabel("Rotor angular velocity ω [rad/s]")

    for step_name, axis, title in (
        ("up", up_axis, "Step-up rotor response"),
        ("down", down_axis, "Step-down rotor response"),
    ):
        step = dynamic_response[step_name]
        sample_times = np.asarray(
            [sample["time_s"] for sample in step["samples"]], dtype=float
        )
        measured = np.asarray(
            [sample["omega_rad_s"] for sample in step["samples"]], dtype=float
        )
        fitted = np.asarray(
            [sample["fit_omega_rad_s"] for sample in step["samples"]], dtype=float
        )
        axis.scatter(
            sample_times,
            measured,
            s=8,
            alpha=0.55,
            color="tab:blue",
            label="Gazebo joint velocity",
            zorder=3,
        )
        axis.plot(
            sample_times,
            fitted,
            color="tab:orange",
            linewidth=2,
            label=(
                "first-order fit "
                f"(τ={step['time_constant_s'] * 1000:.2f} ms)"
            ),
        )
        axis.set_title(title)
        axis.set_xlabel("Time after command [s]")
        axis.set_ylabel("Rotor angular velocity |ω| [rad/s]")

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize="small")

    figure.suptitle("LiftDrag experiment data vs MulticopterMotorModel fit")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def run_sweep(
    args, world_path: Path
) -> tuple[list[dict[str, object]], list[float], dict[str, object]]:
    try:
        from gz.msgs10.double_pb2 import Double
        from gz.msgs10.model_pb2 import Model
        from gz.msgs10.wind_pb2 import Wind
        from gz.msgs10.wrench_pb2 import Wrench
        from gz.transport13 import Node
    except ImportError as exc:
        raise RuntimeError(
            "Gazebo Harmonic Python bindings are required: gz.transport13 and "
            "gz.msgs10"
        ) from exc

    partition = f"rotor-fit-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    os.environ["GZ_PARTITION"] = partition
    existing_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH")

    environment = os.environ.copy()
    if existing_resource_path:
        environment["GZ_SIM_RESOURCE_PATH"] = existing_resource_path

    node = Node()
    collector = WrenchCollector()
    joint_collector = JointVelocityCollector()
    if not node.subscribe(Wrench, WRENCH_TOPIC, collector.callback):
        raise RuntimeError(f"failed to subscribe to {WRENCH_TOPIC}")
    if not node.subscribe(Model, JOINT_STATE_TOPIC, joint_collector.callback):
        raise RuntimeError(f"failed to subscribe to {JOINT_STATE_TOPIC}")
    velocity_publisher = node.advertise(VELOCITY_TOPIC, Double)
    wind_publisher = node.advertise(WIND_TOPIC, Wind)

    log_path = world_path.with_suffix(".log")
    log_stream = log_path.open("w", encoding="utf-8")
    command = ["gz", "sim", "-s", "-r", "-v", str(args.gz_verbose), str(world_path)]
    process = subprocess.Popen(
        command,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )

    cases: list[dict[str, object]] = []
    dynamic_response: dict[str, object] = {}
    try:
        wait_until(
            velocity_publisher.has_connections,
            args.startup_timeout,
            "the rotor velocity controller",
            process,
        )
        wait_until(
            wind_publisher.has_connections,
            args.startup_timeout,
            "the world wind controller",
            process,
        )
        wait_until(
            lambda: collector.count() >= args.minimum_samples,
            args.startup_timeout,
            "the rotor wrench sensor",
            process,
        )
        wait_until(
            lambda: joint_collector.count() >= args.minimum_samples,
            args.startup_timeout,
            "the rotor joint-state publisher",
            process,
        )

        def run_case(
            omega: float, wind: tuple[float, float, float]
        ) -> tuple[np.ndarray, np.ndarray, int]:
            signed_omega = omega if args.turning_direction == "ccw" else -omega
            # Re-publish to make command changes robust to transport discovery.
            for _ in range(3):
                publish_wind(wind_publisher, Wind, wind)
                publish_velocity(velocity_publisher, Double, signed_omega)
                time.sleep(0.03)
            time.sleep(args.settle_seconds)
            collector.clear()
            time.sleep(args.sample_seconds)
            if process.poll() is not None:
                raise RuntimeError(
                    f"Gazebo exited with status {process.returncode}; see {log_path}"
                )
            return average_wrench(collector.snapshot(), args.minimum_samples)

        print("Collecting zero-speed tare...")
        tare_mean, _, tare_count = run_case(0.0, (0.0, 0.0, 0.0))
        print(f"  tare samples={tare_count}, wrench={tare_mean.tolist()}")

        total = len(args.omega) * len(args.winds)
        index = 0
        for wind in args.winds:
            for omega in args.omega:
                index += 1
                mean, standard_deviation, count = run_case(omega, wind)
                corrected = mean - tare_mean
                case = {
                    "omega_rad_s": float(omega),
                    "wind_m_s": [float(value) for value in wind],
                    "force_N": corrected[:3].tolist(),
                    "torque_Nm": corrected[3:].tolist(),
                    "wrench_stddev": standard_deviation.tolist(),
                    "sample_count": count,
                }
                cases.append(case)
                print(
                    f"[{index:3d}/{total}] omega={omega:7.1f} rad/s "
                    f"wind={wind} F={corrected[:3]} M={corrected[3:]}"
                )

        def capture_dynamic_step(
            step_name: str, from_omega: float, to_omega: float
        ) -> dict[str, object]:
            direction_sign = 1.0 if args.turning_direction == "ccw" else -1.0
            for _ in range(3):
                publish_wind(wind_publisher, Wind, (0.0, 0.0, 0.0))
                publish_velocity(
                    velocity_publisher, Double, direction_sign * from_omega
                )
                time.sleep(0.03)
            time.sleep(args.dynamic_settle_seconds)
            joint_collector.clear()
            time.sleep(args.dynamic_pre_step_seconds)
            command_time = time.monotonic()
            publish_velocity(velocity_publisher, Double, direction_sign * to_omega)
            time.sleep(args.dynamic_duration)
            samples = joint_collector.snapshot()
            if process.poll() is not None:
                raise RuntimeError(
                    f"Gazebo exited with status {process.returncode}; see {log_path}"
                )
            fitted = fit_first_order_step(
                samples, command_time, from_omega, to_omega
            )
            print(
                f"  {step_name}: {from_omega:g} -> {to_omega:g} rad/s, "
                f"tau={fitted['time_constant_s'] * 1000:.3f} ms, "
                f"R^2={fitted['r_squared']:.5f}"
            )
            return fitted

        print("Collecting dynamic step responses...")
        dynamic_omega = args.dynamic_omega or max(args.omega)
        dynamic_response["up"] = capture_dynamic_step(
            "up", args.dynamic_low_omega, dynamic_omega
        )
        dynamic_response["down"] = capture_dynamic_step(
            "down", dynamic_omega, args.dynamic_low_omega
        )
    finally:
        stop_process(process)
        log_stream.close()

    return cases, tare_mean.tolist(), dynamic_response


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an inline LiftDrag rotor test model and fit Gazebo "
            "MulticopterMotorModel parameters with a headless sweep."
        )
    )
    source = parser.add_argument_group("source SDF extraction")
    source.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "optional ArduPilot model.sdf; if omitted, construct the rotor "
            "from the LiftDrag parameters below"
        ),
    )
    source.add_argument(
        "--rotor-link",
        default="rotor_0",
        help="source link whose inertia, joint axis and LiftDrag blocks are extracted",
    )
    lift_drag = parser.add_argument_group(
        "direct LiftDrag parameters (used when source is omitted)"
    )
    lift_drag.add_argument("--blade-count", type=int, default=2)
    lift_drag.add_argument(
        "--radius", type=float, default=0.084, help="blade cp radius [m]"
    )
    lift_drag.add_argument(
        "--area", type=float, default=0.002, help="area per blade element [m^2]"
    )
    lift_drag.add_argument("--air-density", type=float, default=1.2041)
    lift_drag.add_argument("--a0", type=float, default=0.3)
    lift_drag.add_argument("--alpha-stall", type=float, default=1.4)
    lift_drag.add_argument("--cla", type=float, default=4.25)
    lift_drag.add_argument("--cda", type=float, default=0.10)
    lift_drag.add_argument("--cma", type=float, default=0.0)
    lift_drag.add_argument("--cla-stall", type=float, default=-0.025)
    lift_drag.add_argument("--cda-stall", type=float, default=0.0)
    lift_drag.add_argument("--cma-stall", type=float, default=0.0)
    lift_drag.add_argument("--rotor-mass", type=float, default=0.005)
    lift_drag.add_argument("--rotor-ixx", type=float, default=9.75e-7)
    lift_drag.add_argument("--rotor-iyy", type=float, default=2.73104e-4)
    lift_drag.add_argument("--rotor-izz", type=float, default=2.74004e-4)
    parser.add_argument(
        "--omega",
        type=parse_number_grid,
        default=parse_number_grid("0:600:100"),
        help="rad/s grid as comma values or start:stop:step",
    )
    parser.add_argument(
        "--winds",
        type=parse_wind_grid,
        default=parse_wind_grid("0,0;2.5,0;5,0;0,2.5;0,5"),
        help="semicolon-separated x,y[,z] wind vectors in m/s",
    )
    parser.add_argument("--settle-seconds", type=float, default=0.4)
    parser.add_argument("--sample-seconds", type=float, default=0.6)
    parser.add_argument("--minimum-samples", type=int, default=30)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--turning-direction", choices=("ccw", "cw"), default="ccw")
    controller = parser.add_argument_group(
        "force/PID rotor drive "
        "(fallback if source SDF has no matching ArduPilot control)"
    )
    controller.add_argument("--joint-p-gain", type=float, default=0.20)
    controller.add_argument("--joint-i-gain", type=float, default=0.0)
    controller.add_argument("--joint-d-gain", type=float, default=0.0)
    controller.add_argument("--joint-i-max", type=float, default=0.0)
    controller.add_argument("--joint-i-min", type=float, default=0.0)
    controller.add_argument("--joint-cmd-max", type=float, default=2.5)
    controller.add_argument("--joint-cmd-min", type=float, default=-2.5)
    dynamics = parser.add_argument_group("dynamic-response identification")
    dynamics.add_argument(
        "--dynamic-omega",
        type=float,
        default=None,
        help="step-test high speed [rad/s] (default: maximum --omega value)",
    )
    dynamics.add_argument(
        "--dynamic-low-omega",
        type=float,
        default=0.0,
        help="step-test low speed [rad/s]",
    )
    dynamics.add_argument("--dynamic-duration", type=float, default=0.5)
    dynamics.add_argument("--dynamic-settle-seconds", type=float, default=0.5)
    dynamics.add_argument("--dynamic-pre-step-seconds", type=float, default=0.05)
    parser.add_argument("--gz-verbose", type=int, choices=range(0, 5), default=1)
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=None,
        help=(
            "Analysis result path (default: beside source SDF or in the current directory)"
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="do not generate the experiment-data / fitting-line PNG",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="also save the generated rotor_test/model.sdf at this path",
    )
    parser.add_argument(
        "--world-output",
        type=Path,
        help="also save the generated test world at this path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and generate the world without starting Gazebo",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.source:
        args.source = args.source.resolve()
    if args.settle_seconds < 0 or args.sample_seconds <= 0:
        raise ValueError("settle-seconds must be >= 0 and sample-seconds must be > 0")
    if args.minimum_samples < 2:
        raise ValueError("minimum-samples must be at least 2")
    if args.blade_count < 1:
        raise ValueError("blade-count must be at least 1")
    if args.radius <= 0 or args.area <= 0 or args.air_density <= 0:
        raise ValueError("radius, area and air-density must be positive")
    dynamic_omega = args.dynamic_omega or max(args.omega)
    if args.dynamic_low_omega < 0 or dynamic_omega <= args.dynamic_low_omega:
        raise ValueError("dynamic-omega must be greater than dynamic-low-omega >= 0")
    if args.dynamic_duration <= 0 or args.dynamic_pre_step_seconds <= 0:
        raise ValueError(
            "dynamic-duration and dynamic-pre-step-seconds must be positive"
        )
    if args.dynamic_settle_seconds < 0:
        raise ValueError("dynamic-settle-seconds must be nonnegative")
    if args.joint_cmd_min >= args.joint_cmd_max:
        raise ValueError("joint-cmd-min must be less than joint-cmd-max")

    if args.source:
        model_xml, lift_drag_parameters, controller_parameters = (
            make_rotor_model_from_source(args.source, args.rotor_link, args)
        )
        source_description = str(args.source)
        print(
            f"Extracted {len(lift_drag_parameters)} LiftDrag elements for "
            f"{args.rotor_link!r} from {args.source}"
        )
    else:
        model_xml, lift_drag_parameters, controller_parameters = (
            make_rotor_model_from_parameters(args)
        )
        source_description = "command-line LiftDrag parameters"
        print(
            f"Generated {len(lift_drag_parameters)} symmetric LiftDrag blade "
            "elements from command-line parameters"
        )

    rotor_document = rotor_model_document(model_xml)
    world_text = make_world(model_xml)
    if args.model_output:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        args.model_output.write_text(rotor_document, encoding="utf-8")
        print(f"Wrote rotor test model: {args.model_output}")
    if args.world_output:
        args.world_output.parent.mkdir(parents=True, exist_ok=True)
        args.world_output.write_text(world_text, encoding="utf-8")
        print(f"Wrote test world: {args.world_output}")
    if args.dry_run:
        print("Dry run complete; Gazebo was not started.")
        return 0

    if args.source:
        output_file_name = f"{args.rotor_link}_motor_model_fit.json"
    else:
        output_file_name = "rotor_test_motor_model_fit.json"

    if args.analysis_output:
        output_dir = args.analysis_output.resolve()
        if output_dir.exists() and not output_dir.is_dir():
            raise ValueError(f"output path exists but is not a directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    elif args.source and args.analysis_output is None:
        output_dir = args.source.parent
    else:
        output_dir = Path.cwd()

    output_path = output_dir / output_file_name
    csv_path = output_path.with_suffix(".csv")
    dynamic_csv_path = output_path.with_name(f"{output_path.stem}_dynamic.csv")
    plot_path = output_path.with_suffix(".png")

    with tempfile.TemporaryDirectory(prefix="rotor-fit-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        rotor_model_path = temporary_path / MODEL_NAME / "model.sdf"
        rotor_model_path.parent.mkdir(parents=True)
        rotor_model_path.write_text(rotor_document, encoding="utf-8")
        world_path = temporary_path / f"{WORLD_NAME}.sdf"
        world_path.write_text(world_text, encoding="utf-8")
        cases, tare, dynamic_response = run_sweep(args, world_path)

    coefficients, errors, signs = identify_coefficients(
        cases, args.turning_direction
    )
    time_constant_up = float(dynamic_response["up"]["time_constant_s"])
    time_constant_down = float(dynamic_response["down"]["time_constant_s"])
    max_rot_velocity = max(max(args.omega), dynamic_omega)
    snippet = plugin_xml(
        coefficients=coefficients,
        max_rot_velocity=max_rot_velocity,
        turning_direction=args.turning_direction,
        time_constant_up=time_constant_up,
        time_constant_down=time_constant_down,
    )
    result = {
        "source_model": source_description,
        "source_rotor_link": args.rotor_link if args.source else None,
        "lift_drag_elements": lift_drag_parameters,
        "test_envelope": {
            "omega_rad_s": args.omega,
            "wind_m_s": [list(wind) for wind in args.winds],
            "settle_seconds": args.settle_seconds,
            "sample_seconds": args.sample_seconds,
            "dynamic_low_omega_rad_s": args.dynamic_low_omega,
            "dynamic_high_omega_rad_s": dynamic_omega,
            "dynamic_duration_seconds": args.dynamic_duration,
        },
        "rotor_drive_controller": controller_parameters,
        "tare_wrench": tare,
        "sensor_signs": signs,
        "multicopter_motor_model": {
            **coefficients,
            "maxRotVelocity": max_rot_velocity,
            "timeConstantUp": time_constant_up,
            "timeConstantDown": time_constant_down,
            "turningDirection": args.turning_direction,
        },
        "fit_error": errors,
        "dynamic_response": dynamic_response,
        "plot": None if args.no_plot else str(plot_path),
        "plugin_xml": snippet,
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_case_csv(csv_path, cases)
    write_dynamic_csv(dynamic_csv_path, dynamic_response)
    if not args.no_plot:
        write_fit_plot(
            path=plot_path,
            cases=cases,
            coefficients=coefficients,
            signs=signs,
            turning_direction=args.turning_direction,
            dynamic_response=dynamic_response,
        )

    print("\nIdentified MulticopterMotorModel coefficients:")
    for name, value in coefficients.items():
        print(f"  {name:28s} {value:.9g}")
    print(f"  {'timeConstantUp':28s} {time_constant_up:.9g}")
    print(f"  {'timeConstantDown':28s} {time_constant_down:.9g}")
    print("Fit errors:")
    for name, value in errors.items():
        print(f"  {name:28s} {value:.9g}")
    print(f"\nJSON: {output_path}")
    print(f"CSV:  {csv_path}")
    print(f"Dynamic CSV: {dynamic_csv_path}")
    if not args.no_plot:
        print(f"Plot: {plot_path}")
    print("\nSuggested plugin:\n")
    print(snippet)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
