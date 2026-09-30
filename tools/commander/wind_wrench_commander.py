#!/usr/bin/env python3
"""Apply body-axis aerodynamic drag to a Gazebo link from a ROS 2 node.

Gazebo's OdometryPublisher is bridged to ``nav_msgs/msg/Odometry``.  Its
linear twist is expressed in the body / child frame.  Wind commands arrive as
``geometry_msgs/msg/Vector3Stamped`` in the Gazebo world ENU frame.  This node
computes independent body-axis quadratic drag and updates a persistent wrench
through Gazebo Transport's stock ApplyLinkWrench system.

The node also forwards every ROS wind command to WindEffects, so the same wind
field is seen by rotor LiftDrag systems and by this body-drag calculation.

Required odometry bridge (run in another shell):

  ros2 run ros_gz_bridge parameter_bridge \
    '/model/px4vision/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'

Example:

  python3 tools/commander/wind_wrench_commander.py --initial-wind 10 0 0
"""

from __future__ import annotations

import os

# Ubuntu's Gazebo Python messages were generated for protobuf 3.x, while this
# workspace may expose a newer Python protobuf package.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import math
import signal
import sys
from dataclasses import dataclass
from typing import Sequence

try:
    import rclpy
    from geometry_msgs.msg import Vector3Stamped, WrenchStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node as RosNode
    from rclpy.qos import qos_profile_sensor_data
except ImportError as exc:  # pragma: no cover - depends on sourced ROS setup
    raise SystemExit(
        "ROS 2 Python packages are unavailable. Source /opt/ros/humble/setup.bash "
        f"before running this command ({exc})."
    ) from exc

try:
    from gz.msgs10.entity_pb2 import Entity
    from gz.msgs10.entity_wrench_pb2 import EntityWrench
    from gz.msgs10.wind_pb2 import Wind
    from gz.transport13 import Node as GzNode
except (ImportError, TypeError) as exc:  # pragma: no cover - environment-specific
    raise SystemExit(f"Gazebo Transport Python bindings are unavailable: {exc}") from exc


Vector = tuple[float, float, float]


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float


def _vector(values: Sequence[float]) -> Vector:
    return float(values[0]), float(values[1]), float(values[2])


def _add(a: Vector, b: Vector) -> Vector:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _sub(a: Vector, b: Vector) -> Vector:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _normalize_quaternion(q: Quaternion) -> Quaternion:
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm < 1e-12:
        raise ValueError("odometry contains a zero-length quaternion")
    return Quaternion(q.x / norm, q.y / norm, q.z / norm, q.w / norm)


def rotate_body_to_world(q_in: Quaternion, value: Vector) -> Vector:
    """Rotate a vector with a body-to-world quaternion."""

    q = _normalize_quaternion(q_in)
    x, y, z = value

    # R(q) v, expanded to avoid an extra numerical dependency.
    return (
        (1 - 2 * (q.y * q.y + q.z * q.z)) * x
        + 2 * (q.x * q.y - q.z * q.w) * y
        + 2 * (q.x * q.z + q.y * q.w) * z,
        2 * (q.x * q.y + q.z * q.w) * x
        + (1 - 2 * (q.x * q.x + q.z * q.z)) * y
        + 2 * (q.y * q.z - q.x * q.w) * z,
        2 * (q.x * q.z - q.y * q.w) * x
        + 2 * (q.y * q.z + q.x * q.w) * y
        + (1 - 2 * (q.x * q.x + q.y * q.y)) * z,
    )


def rotate_world_to_body(q: Quaternion, value: Vector) -> Vector:
    return rotate_body_to_world(Quaternion(-q.x, -q.y, -q.z, q.w), value)


def quadratic_drag(relative_air_velocity_body: Vector, rho: float, cda: Vector) -> Vector:
    """Return body-frame drag opposing the vehicle velocity relative to air."""

    return tuple(
        -0.5 * rho * cda_i * velocity * abs(velocity)
        for velocity, cda_i in zip(relative_air_velocity_body, cda)
    )  # type: ignore[return-value]


class WindWrenchCommander(RosNode):
    def __init__(self, args: argparse.Namespace):
        super().__init__("wind_wrench_commander")
        self.args = args
        self.rho = args.air_density
        self.cda = tuple(cd * area for cd, area in zip(args.cd, args.area))
        self.wind_world = _vector(args.initial_wind)
        self.odometry: Odometry | None = None
        self.additional_force_world: Vector = (0.0, 0.0, 0.0)
        self.additional_torque_world: Vector = (0.0, 0.0, 0.0)
        self.update_count = 0

        self.gz_node = GzNode()
        self.wrench_topic = f"/world/{args.world}/wrench/persistent"
        self.wrench_clear_topic = f"/world/{args.world}/wrench/clear"
        self.gz_wind_topic = f"/world/{args.world}/wind"
        self.wrench_pub = self.gz_node.advertise(self.wrench_topic, EntityWrench)
        self.wrench_clear_pub = self.gz_node.advertise(self.wrench_clear_topic, Entity)
        self.wind_pub = self.gz_node.advertise(self.gz_wind_topic, Wind)

        self.create_subscription(
            Odometry,
            args.odometry_topic,
            self._on_odometry,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Vector3Stamped,
            args.wind_topic,
            self._on_wind,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            WrenchStamped,
            args.additional_wrench_topic,
            self._on_additional_wrench,
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(1.0 / args.rate_hz, self._update)
        self._publish_gazebo_wind()

        self.get_logger().info(
            f"target={args.entity} world={args.world} rate={args.rate_hz:g} Hz "
            f"rho={self.rho:g} CdA={self.cda} initial_wind_ENU={self.wind_world}"
        )

    def _on_odometry(self, msg: Odometry) -> None:
        self.odometry = msg

    def _on_wind(self, msg: Vector3Stamped) -> None:
        if msg.header.frame_id and msg.header.frame_id not in ("world", "map", "enu"):
            self.get_logger().warning(
                f"wind frame_id={msg.header.frame_id!r}; interpreting it as world ENU"
            )
        self.wind_world = (msg.vector.x, msg.vector.y, msg.vector.z)
        self._publish_gazebo_wind()

    def _on_additional_wrench(self, msg: WrenchStamped) -> None:
        if msg.header.frame_id and msg.header.frame_id not in ("world", "map", "enu"):
            self.get_logger().warning(
                f"additional wrench frame_id={msg.header.frame_id!r}; expected world ENU"
            )
        self.additional_force_world = (
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z,
        )
        self.additional_torque_world = (
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        )

    def _publish_gazebo_wind(self) -> None:
        msg = Wind()
        msg.linear_velocity.x, msg.linear_velocity.y, msg.linear_velocity.z = self.wind_world
        msg.enable_wind = True
        self.wind_pub.publish(msg)

    def _update(self) -> None:
        # Re-advertise the commanded wind so startup order does not matter and
        # rotor LiftDrag always receives the same field used below.
        self._publish_gazebo_wind()

        if self.odometry is None:
            if self.update_count % max(1, round(self.args.rate_hz * 2)) == 0:
                self.get_logger().warning(
                    f"waiting for ROS odometry on {self.args.odometry_topic}"
                )
            self.update_count += 1
            return

        pose_q = self.odometry.pose.pose.orientation
        q_body_to_world = Quaternion(pose_q.x, pose_q.y, pose_q.z, pose_q.w)

        # Gazebo's 3D OdometryPublisher expresses twist.linear in child/body frame.
        velocity = self.odometry.twist.twist.linear
        vehicle_velocity_body = (velocity.x, velocity.y, velocity.z)
        wind_body = rotate_world_to_body(q_body_to_world, self.wind_world)
        relative_air_velocity_body = _sub(vehicle_velocity_body, wind_body)
        drag_body = quadratic_drag(relative_air_velocity_body, self.rho, self.cda)
        drag_world = rotate_body_to_world(q_body_to_world, drag_body)
        force_world = _add(drag_world, self.additional_force_world)

        msg = EntityWrench()
        msg.entity.name = self.args.entity
        msg.entity.type = Entity.LINK
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = force_world
        (
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z,
        ) = self.additional_torque_world
        self.wrench_pub.publish(msg)

        self.update_count += 1
        if self.args.log_rate_hz > 0:
            divisor = max(1, round(self.args.rate_hz / self.args.log_rate_hz))
            if self.update_count % divisor == 0:
                self.get_logger().info(
                    f"Vrel_body={tuple(round(x, 3) for x in relative_air_velocity_body)} "
                    f"Fdrag_body={tuple(round(x, 3) for x in drag_body)} "
                    f"Fworld={tuple(round(x, 3) for x in force_world)}"
                )

    def clear_wrench(self) -> None:
        msg = Entity()
        msg.name = self.args.entity
        msg.type = Entity.LINK
        self.wrench_clear_pub.publish(msg)


def positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("expected a finite value greater than zero")
    return value


def finite_triplet(values: Sequence[str]) -> Vector:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise argparse.ArgumentTypeError("vector values must be finite")
    return result  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="px4vision_windy")
    parser.add_argument("--entity", default="px4vision::base_link")
    parser.add_argument("--odometry-topic", default="/model/px4vision/odometry")
    parser.add_argument("--wind-topic", default="/wind_wrench/wind")
    parser.add_argument(
        "--additional-wrench-topic",
        default="/wind_wrench/additional_wrench",
        help="optional world-frame lift / force / torque input",
    )
    parser.add_argument("--rate-hz", type=positive_float, default=100.0)
    parser.add_argument("--log-rate-hz", type=float, default=1.0)
    parser.add_argument("--air-density", type=positive_float, default=1.2041)
    parser.add_argument(
        "--cd",
        nargs=3,
        type=float,
        metavar=("CD_X", "CD_Y", "CD_Z"),
        default=(1.28, 1.28, 0.0),
    )
    parser.add_argument(
        "--area",
        nargs=3,
        type=float,
        metavar=("AREA_X", "AREA_Y", "AREA_Z"),
        default=(0.0268, 0.0256, 0.0),
        help="body-axis reference areas in m^2",
    )
    parser.add_argument(
        "--initial-wind",
        nargs=3,
        type=float,
        metavar=("EAST", "NORTH", "UP"),
        default=(0.0, 0.0, 0.0),
        help="initial world ENU wind in m/s",
    )
    args = parser.parse_args()

    for label, values in (("--cd", args.cd), ("--area", args.area)):
        if not all(math.isfinite(value) and value >= 0 for value in values):
            parser.error(f"{label} values must be finite and non-negative")
    args.initial_wind = finite_triplet([str(value) for value in args.initial_wind])
    return args


def main() -> int:
    args = parse_args()
    rclpy.init()
    node: WindWrenchCommander | None = None
    try:
        node = WindWrenchCommander(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.clear_wrench()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda _signum, _frame: rclpy.shutdown())
    raise SystemExit(main())
