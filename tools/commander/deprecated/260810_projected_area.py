#!/usr/bin/env python3

from pathlib import Path
import argparse
import math

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def rotation_matrix_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """SDF roll-pitch-yaw rotation: R = Rz(yaw) Ry(pitch) Rx(roll)."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr],
    ])

    ry = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp],
    ])

    rz = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1],
    ])

    return rz @ ry @ rx


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")

    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry.copy()
            for geometry in loaded.dump()
            if isinstance(geometry, trimesh.Trimesh)
        ]

        if not meshes:
            raise RuntimeError(f"No triangle mesh found in {path}")

        return trimesh.util.concatenate(meshes)

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    raise RuntimeError(f"Unsupported mesh type: {type(loaded)}")


def transform_vertices(
    vertices: np.ndarray,
    scale: np.ndarray,
    translation: np.ndarray,
    rpy: np.ndarray,
) -> np.ndarray:
    # SDF visual transform: p_link = translation + R * (scale * p_mesh)
    scaled = vertices * scale
    rotation = rotation_matrix_rpy(*rpy)

    return (rotation @ scaled.T).T + translation


def projection_indices(flow_axis: str) -> tuple[int, int]:
    # 유동 방향에 수직인 평면
    if flow_axis == "x":
        return 1, 2  # YZ
    if flow_axis == "y":
        return 0, 2  # XZ
    if flow_axis == "z":
        return 0, 1  # XY

    raise ValueError(flow_axis)


def rasterized_projected_area(
    vertices: np.ndarray,
    faces: np.ndarray,
    flow_axis: str,
    pixels_per_meter: float,
    padding: int = 4,
) -> tuple[float, Image.Image]:
    axis_u, axis_v = projection_indices(flow_axis)

    projected = vertices[:, [axis_u, axis_v]]

    min_uv = projected.min(axis=0)
    max_uv = projected.max(axis=0)
    extent = max_uv - min_uv

    width = int(math.ceil(extent[0] * pixels_per_meter)) + 2 * padding
    height = int(math.ceil(extent[1] * pixels_per_meter)) + 2 * padding

    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)

    def to_pixel(points: np.ndarray) -> list[tuple[float, float]]:
        pixel = (points - min_uv) * pixels_per_meter
        pixel[:, 0] += padding

        # 이미지 Y축은 아래 방향이므로 뒤집음
        pixel[:, 1] = (
            height - padding - pixel[:, 1]
        )

        return [tuple(point) for point in pixel]

    # 겹치는 삼각형은 같은 bitmap에 그리므로 중복 합산되지 않는다.
    for face in faces:
        triangle = projected[face]

        # 투영 후 면적이 거의 0인 삼각형 제외
        twice_area = abs(
            np.cross(
                triangle[1] - triangle[0],
                triangle[2] - triangle[0],
            )
        )

        if twice_area < 1e-14:
            continue

        draw.polygon(to_pixel(triangle), fill=1)

    occupied_pixels = np.count_nonzero(np.asarray(image))
    pixel_area = 1.0 / pixels_per_meter**2
    area = occupied_pixels * pixel_area

    return area, image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=10000,
        help="Raster resolution; 10000 means 0.1 mm per pixel",
    )
    parser.add_argument(
        "--output-prefix",
        default="px4vision_projection",
    )
    args = parser.parse_args()

    mesh = load_mesh(args.mesh)

    # px4vision/model.sdf의 visual 설정
    scale = np.array([0.001, 0.001, 0.001])
    translation = np.array([0.0, 0.0, 0.0])
    rpy = np.array([1.5707, 0.0, 0.01])

    vertices = transform_vertices(
        vertices=np.asarray(mesh.vertices),
        scale=scale,
        translation=translation,
        rpy=rpy,
    )

    print(f"Triangles: {len(mesh.faces)}")
    print(f"Vertices:  {len(mesh.vertices)}")
    print(f"Bounds after SDF transform:\n{np.vstack((vertices.min(0), vertices.max(0)))}")

    plane_names = {
        "x": "YZ",
        "y": "XZ",
        "z": "XY",
    }

    for flow_axis in ("x", "y", "z"):
        area, image = rasterized_projected_area(
            vertices=vertices,
            faces=np.asarray(mesh.faces),
            flow_axis=flow_axis,
            pixels_per_meter=args.pixels_per_meter,
        )

        output = Path(
            f"{args.output_prefix}_flow_{flow_axis}.png"
        )
        image.save(output)

        print(
            f"Flow along {flow_axis.upper()}: "
            f"{plane_names[flow_axis]} area = {area:.6f} m² "
            f"({output})"
        )


if __name__ == "__main__":
    main()