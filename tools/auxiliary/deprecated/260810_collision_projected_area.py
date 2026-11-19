#!/usr/bin/env python3
"""Calculate axis-aligned projected areas of box collisions in an SDF file.

The reported areas are expressed in the containing model frame.  Link and
collision poses are applied.  The total is a sum of individual silhouettes;
overlap between different collision boxes is therefore counted more than once.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class Pose:
    translation: np.ndarray
    rotation: np.ndarray


def rotation_matrix_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the SDF fixed-axis roll-pitch-yaw rotation matrix."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


def parse_vector(text: str | None, count: int, description: str) -> np.ndarray:
    if text is None:
        raise ValueError(f"missing {description}")

    try:
        values = np.asarray([float(value) for value in text.split()], dtype=float)
    except ValueError as exc:
        raise ValueError(f"invalid {description}: {text!r}") from exc

    if len(values) != count:
        raise ValueError(
            f"{description} must contain {count} numbers, got {len(values)}: {text!r}"
        )
    return values


def parse_pose(element: ET.Element) -> Pose:
    pose_element = element.find("pose")
    if pose_element is None or not pose_element.text or not pose_element.text.strip():
        return Pose(np.zeros(3), np.eye(3))

    if pose_element.get("relative_to"):
        raise ValueError(
            f"pose relative_to={pose_element.get('relative_to')!r} is not supported"
        )

    values = parse_vector(pose_element.text, 6, "pose")
    return Pose(values[:3], rotation_matrix_rpy(*values[3:]))


def compose(parent: Pose, child: Pose) -> Pose:
    return Pose(
        parent.translation + parent.rotation @ child.translation,
        parent.rotation @ child.rotation,
    )


def projected_areas(size: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Return YZ, XZ and XY silhouette areas for flow along model X, Y, Z.

    For a rectangular box, the orthographic projection along unit vector n is
    the sum of each distinct face area multiplied by the absolute dot product
    between n and that face's normal.
    """

    sx, sy, sz = size
    local_face_areas = np.array((sy * sz, sx * sz, sx * sy))

    # abs(rotation)[i, j] is the contribution of local face-normal j to a
    # projection whose viewing / flow direction is model axis i.
    return np.abs(rotation) @ local_face_areas


def box_collisions(model: ET.Element):
    for link in model.findall("link"):
        link_name = link.get("name", "<unnamed-link>")
        link_pose = parse_pose(link)

        for collision in link.findall("collision"):
            box = collision.find("geometry/box")
            if box is None:
                continue

            size_element = box.find("size")
            size = parse_vector(
                size_element.text if size_element is not None else None,
                3,
                "box size",
            )
            if np.any(size <= 0):
                raise ValueError(f"box size must be positive, got {size}")

            collision_pose = compose(link_pose, parse_pose(collision))
            collision_name = collision.get("name", "<unnamed-collision>")
            yield link_name, collision_name, size, collision_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdf", type=Path, help="SDF model file to inspect")
    parser.add_argument(
        "--model",
        help="Model name when the SDF contains more than one model",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        root = ET.parse(args.sdf).getroot()
    except (OSError, ET.ParseError) as exc:
        print(f"error: unable to read {args.sdf}: {exc}", file=sys.stderr)
        return 2

    models = root.findall("model") + root.findall("world/model")
    if root.tag == "model":
        models = [root]

    if args.model:
        models = [model for model in models if model.get("name") == args.model]
        if not models:
            print(f"error: model {args.model!r} was not found", file=sys.stderr)
            return 2
    elif len(models) > 1:
        names = ", ".join(repr(model.get("name")) for model in models)
        print(
            f"error: SDF contains multiple models ({names}); select one with --model",
            file=sys.stderr,
        )
        return 2
    elif not models:
        print("error: no inline <model> found in the SDF", file=sys.stderr)
        return 2

    model = models[0]
    total = np.zeros(3)
    count = 0

    print(f"SDF:   {args.sdf}")
    print(f"Model: {model.get('name', '<unnamed-model>')}")
    print("Frame: model")
    print()

    try:
        for link_name, collision_name, size, pose in box_collisions(model):
            areas = projected_areas(size, pose.rotation)
            total += areas
            count += 1

            print(f"{link_name}/{collision_name}")
            print(f"  size: {size[0]:.6g} {size[1]:.6g} {size[2]:.6g} m")
            print(f"  flow X -> YZ: {areas[0]:.6f} m^2")
            print(f"  flow Y -> XZ: {areas[1]:.6f} m^2")
            print(f"  flow Z -> XY: {areas[2]:.6f} m^2")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if count == 0:
        print("No box collision geometries found.")
        return 1

    if count > 1:
        print()
        print("Sum of individual box projections (overlap is not removed)")
        print(f"  flow X -> YZ: {total[0]:.6f} m^2")
        print(f"  flow Y -> XZ: {total[1]:.6f} m^2")
        print(f"  flow Z -> XY: {total[2]:.6f} m^2")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
