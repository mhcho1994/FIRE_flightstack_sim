#!/usr/bin/env python3
r"""Projected areas of a Gazebo model along model-frame X, Y and Z.

Modes
-----
scalecheck: Read four rotor link poses from model.sdf, compute both opposite
            motor-to-motor distances in the model XY plane and compare them
            with the real wheelbase (default 286 mm). Also check whether
            CLI --scale/--rpy/--translation match SDF visual declarations.
collision: Parse box-shaped SDF <collision> geometries and compute their
           analytical projected areas after composing link/collision poses.
           Areas of overlapping boxes are summed (not unioned).
raster:    Load the triangle body mesh, apply scale and SDF-style RPY,
           rasterize its YZ/XZ/XY silhouettes and count occupied pixels.
           Compute each projected silhouette's *geometric area centroid* and
           save a three-panel figure with centroid markers (no mask PNGs).

All reported areas are geometric areas in m², not aerodynamic CdA.
Projections: flow X -> YZ, flow Y -> XZ, flow Z -> XY.

Default px4vision/body.dae transform: scale=(0.001, 0.001, 0.001),
translation=(0, 0, 0) metres, RPY=(1.5707, 0, 0.01) radians.
Default 10,000 pixels/m (0.1 mm/pixel) is a numerical choice, not DAE metadata.
The silhouette centroid is NOT a measured aerodynamic center of pressure.

Examples
--------
python compute_projected_area.py --mode scalecheck model.sdf
python compute_projected_area.py --mode raster body.dae --sdf model.sdf --show-wheelbase-on-plot
python compute_projected_area.py --mode raster body.dae --sdf model.sdf --centroid-csv projected_centroids.csv
python compute_projected_area.py --mode collision model.sdf

Dependencies: numpy, Pillow, Matplotlib; trimesh+pycollada for general DAE meshes.
A fallback COLLADA parser handles the provided simple body.dae if pycollada
is unavailable. Collision mode needs only NumPy and the Python stdlib.

Limitations: collision supports boxes only; SDF relative_to, nested models and
include expansion are unsupported. Raster uses the CLI visual transform and
checks it against SDF; the SDF check does not override that transform.
The SDF rotor positions are independent of the body.dae scale, so a correct
SDF wheelbase does NOT by itself establish that the body mesh scale is correct.
For an independent mesh-scale check, optionally supply --mesh-motor-points
with two rotor-hub centers measured directly in unscaled DAE coordinates.
Aerodynamic interactions are not modeled.
When SDF link inertial masses/poses are available, mass-weighted model-frame
center of mass and planar centroid-to-CG moment arms are printed as context.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
import math
import re
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np


@dataclass(frozen=True)
class Pose:
    """Rigid transform from a local frame into its parent frame.

    ``translation`` is a three-element vector in metres and ``rotation`` is a

    3-by-3 direction cosine matrix.

    """
    translation: np.ndarray
    rotation: np.ndarray


def rotation_matrix_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the SDF fixed-axis rotation Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


def parse_vector(text: str | None, count: int, description: str) -> np.ndarray:
    """Parse exactly ``count`` whitespace-separated floating-point values."""
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
    """Parse an element's SDF ``<pose>`` or return the identity transform.

    SDF pose values have the order ``x y z roll pitch yaw``.  A ``relative_to``

    attribute is deliberately rejected because this script does not construct

    or resolve a complete SDF semantic pose graph.

    """
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
    """Compose parent-to-model and child-to-parent rigid transforms."""
    return Pose(
        parent.translation + parent.rotation @ child.translation,
        parent.rotation @ child.rotation,
    )


def box_projected_areas(size: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Return a rotated box's YZ, XZ, and XY projection areas.

    For a rectangular box, its orthographic projection along unit vector ``n``

    is the sum of each distinct face area multiplied by the absolute dot

    product between ``n`` and that face's normal.  With ``rotation`` mapping

    local vectors into the model frame, this becomes::

        abs(rotation) @ [sy*sz, sx*sz, sx*sy]

    The returned indices correspond to flow along model X, Y, and Z.

    """
    sx, sy, sz = size
    face_areas = np.array((sy * sz, sx * sz, sx * sy))
    return np.abs(rotation) @ face_areas


def box_collisions(model: ET.Element):
    """Yield box collision geometry and model-frame pose from an SDF model.

    Non-box collision geometries are ignored.  The yielded tuple contains link

    name, collision name, box size, and the composed link/collision pose.

    """
    for link in model.findall("link"):
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
            yield (
                link.get("name", "<unnamed-link>"),
                collision.get("name", "<unnamed-collision>"),
                size,
                compose(link_pose, parse_pose(collision)),
            )


def select_sdf_model(root: ET.Element, requested_name: str | None) -> ET.Element:
    """Select one inline SDF model, requiring a name if selection is ambiguous."""
    models = [root] if root.tag == "model" else root.findall("model") + root.findall("world/model")
    if requested_name:
        models = [model for model in models if model.get("name") == requested_name]
        if not models:
            raise ValueError(f"model {requested_name!r} was not found")
    elif len(models) > 1:
        names = ", ".join(repr(model.get("name")) for model in models)
        raise ValueError(f"SDF contains multiple models ({names}); select one with --model")
    elif not models:
        raise ValueError("no inline <model> found in the SDF")
    return models[0]



@dataclass(frozen=True)
class WheelbaseResult:
    """Computed SDF rotor geometry and optional consistency-check result."""
    centers: dict[str, np.ndarray]
    pairs: tuple[tuple[str, str], tuple[str, str]]
    distances_m: tuple[float, float]
    wheelbase_m: float
    real_wheelbase_m: float
    error_pct: float
    within_tolerance: bool
    visual_transform_matches: bool | None
    mesh_points_match: bool | None
    center_of_mass_m: np.ndarray | None


@dataclass(frozen=True)
class ProjectionResult:
    """Area and projected geometric centroid from one raster silhouette.

    The two centroid components belong to the plane perpendicular to `axis`;
    the third spatial coordinate is fundamentally undetermined by projection.
    """
    axis: str
    plane: str
    minimum: np.ndarray
    maximum: np.ndarray
    width: int
    height: int
    area_m2: float
    mask: np.ndarray
    centroid_uv_m: np.ndarray


def model_center_of_mass(model: ET.Element) -> np.ndarray | None:
    """Mass-weighted center of mass in the model frame, if inertials exist.

    Missing inertials are ignored. For a complete model CG, every component
    carrying significant mass must provide a <link><inertial> element.
    """
    total_mass = 0.0
    weighted_center = np.zeros(3, dtype=float)
    for link in model.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_text = inertial.findtext("mass")
        if mass_text is None:
            continue
        mass = float(mass_text)
        if not math.isfinite(mass) or mass < 0:
            raise ValueError(f"invalid link inertial mass for {link.get('name')!r}")
        if mass == 0:
            continue
        center = compose(parse_pose(link), parse_pose(inertial)).translation
        weighted_center += mass * center
        total_mass += mass
    return weighted_center / total_mass if total_mass > 0 else None


def opposite_rotor_pairs(centers: dict[str, np.ndarray]):
    """Choose the pairing with maximum sum of horizontal motor separations.

    A quadrotor has three ways to partition four motors into two disjoint
    pairs.  Its two long, diagonal pairs maximize the sum of XY distances;
    no assumptions about rotor index ordering are needed.
    """
    names = sorted(centers, key=lambda name: int(name.rsplit("_", 1)[1]))
    if len(names) != 4:
        raise ValueError(f"expected four rotor_N links, found {names}")
    a, b, c, d = names
    candidates = [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]

    def horizontal_distance(pair):
        first, second = pair
        return float(np.linalg.norm((centers[first] - centers[second])[:2]))

    selected = max(candidates, key=lambda pairs: sum(horizontal_distance(pair)
                                                      for pair in pairs))
    lengths = (horizontal_distance(selected[0]), horizontal_distance(selected[1]))
    return selected, lengths


def find_body_visual(model: ET.Element, mesh_filename: str | None):
    """Find body visual mesh and its model-frame pose, including the link pose.

    If `mesh_filename` is provided, match that filename against SDF URI.
    In scalecheck-only mode, prefer a unique `body.dae` entry.
    """
    meshes = []
    for link in model.findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            uri = (mesh.findtext("uri") or "").strip()
            if not uri:
                continue
            scale_element = mesh.find("scale")
            sdf_scale = (parse_vector(scale_element.text, 3, "visual mesh scale")
                         if scale_element is not None and scale_element.text
                         else np.ones(3))
            visual_pose = compose(parse_pose(link), parse_pose(visual))
            meshes.append((Path(uri).name, uri, sdf_scale, visual_pose))
    target = mesh_filename or "body.dae"
    matches = [entry for entry in meshes if entry[0] == target]
    if len(matches) != 1:
        return None
    return matches[0]


def check_wheelbase(
    sdf_path: Path,
    model_name: str | None,
    real_wheelbase_mm: float,
    tolerance_pct: float,
    mesh_path: Path | None,
    cli_scale: np.ndarray,
    cli_translation: np.ndarray,
    cli_rpy: np.ndarray,
    mesh_motor_points: np.ndarray | None,
) -> WheelbaseResult:
    """Compare SDF rotor spacing with physical wheelbase and CLI visual scale.

    SDF link poses use metres independently of any visual <mesh><scale>.
    Therefore only the *configuration consistency* of the mesh scale is
    checked unless independent DAE rotor-center coordinates are supplied.
    """
    if real_wheelbase_mm <= 0:
        raise ValueError("--real-wheelbase-mm must be positive")
    if tolerance_pct < 0:
        raise ValueError("--wheelbase-tolerance-pct must be nonnegative")
    root = ET.parse(sdf_path).getroot()
    model = select_sdf_model(root, model_name)
    centers: dict[str, np.ndarray] = {}
    for link in model.findall("link"):
        name = link.get("name", "")
        if re.fullmatch(r"rotor_\d+", name):
            centers[name] = parse_pose(link).translation
    if len(centers) != 4:
        raise ValueError(
            f"Expected four direct rotor_N link poses in {sdf_path}; "
            f"found {sorted(centers)}. Nested/include/relative_to frames are unsupported."
        )
    pairs, distances = opposite_rotor_pairs(centers)
    mean_wheelbase = float(np.mean(distances))
    real_m = real_wheelbase_mm / 1000
    error_pct = 100 * (mean_wheelbase / real_m - 1)
    within_tolerance = abs(error_pct) <= tolerance_pct
    center_of_mass = model_center_of_mass(model)

    print("\n" + "=" * 68)
    print(f"PX4 VISION SCALE CHECK  |  {sdf_path}")
    print("=" * 68)
    for name in sorted(centers):
        v = centers[name]
        print(f"{name:<10}: ({v[0]:+.6f}, {v[1]:+.6f}, {v[2]:+.6f}) m")
    for pair, distance in zip(pairs, distances):
        print(f"Opposite motors {pair[0]} <-> {pair[1]}: {distance*1000:.3f} mm")
    print(f"SDF wheelbase (mean):         {mean_wheelbase*1000:.3f} mm")
    print(f"Real wheelbase (reference):   {real_wheelbase_mm:.3f} mm")
    print(f"Relative difference (SDF-real)/real: {error_pct:+.3f}%")
    print(f"Tolerance:                   +/- {tolerance_pct:g}%")
    print(f"[{'PASS' if within_tolerance else 'WARN'}] SDF rotor spacing vs real wheelbase")
    diagonal_mismatch = abs(distances[0] - distances[1]) / mean_wheelbase * 100
    if diagonal_mismatch > tolerance_pct:
        print(f"[WARN] Opposite diagonal lengths differ by {diagonal_mismatch:.3f}%")

    visual = find_body_visual(model, mesh_path.name if mesh_path else None)
    match: bool | None = None
    if visual is None:
        selected_name = mesh_path.name if mesh_path else "body.dae"
        print(f"[INFO] Could not uniquely identify SDF visual for {selected_name!r}; "
              "mesh-scale comparison skipped")
    else:
        _, uri, sdf_scale, visual_pose = visual
        mesh_scale_match = bool(np.allclose(sdf_scale, cli_scale, rtol=1e-8, atol=1e-12))
        pose_match = (bool(np.allclose(visual_pose.translation, cli_translation,
                                       rtol=1e-8, atol=1e-9)) and
                      bool(np.allclose(visual_pose.rotation, rotation_matrix_rpy(*cli_rpy),
                                       rtol=1e-8, atol=1e-8)))
        match = mesh_scale_match and pose_match
        print(f"SDF body visual:             {uri}")
        print(f"SDF mesh scale:              {sdf_scale.tolist()}")
        print(f"CLI mesh scale:              {cli_scale.tolist()}")
        print(f"[{'PASS' if mesh_scale_match else 'WARN'}] CLI --scale matches SDF visual scale")
        print(f"[{'PASS' if pose_match else 'WARN'}] CLI pose matches SDF link+visual pose")
        if not mesh_scale_match:
            print("[WARN] Raster areas use the CLI scale, not the SDF scale.")
        if not pose_match:
            print("[WARN] Raster orientation/location uses CLI --rpy and --translation.")

    mesh_points_match: bool | None = None
    if mesh_motor_points is not None:
        points = (rotation_matrix_rpy(*cli_rpy) @
                  (mesh_motor_points.reshape(2, 3) * cli_scale).T).T + cli_translation
        distance_xy = float(np.linalg.norm(points[1, :2] - points[0, :2]))
        mesh_error = 100 * (distance_xy / real_m - 1)
        vs_sdf = 100 * (distance_xy / mean_wheelbase - 1)
        mesh_points_match = abs(mesh_error) <= tolerance_pct
        print(f"Independent DAE motor-center spacing after CLI transform: "
              f"{distance_xy*1000:.3f} mm")
        print(f"DAE vs SDF rotor spacing: {vs_sdf:+.3f}%")
        print(f"DAE vs real rotor spacing: {mesh_error:+.3f}%")
        print(f"[{'PASS' if mesh_points_match else 'WARN'}] "
              "independently supplied mesh geometry vs real wheelbase")
    else:
        print("[INFO] Rotor link poses do not depend on visual mesh scale.")
        print("[INFO] SDF spacing + SDF/CLI scale consistency do not prove "
              "physical body.dae dimensions.")
        print("[INFO] To check them independently, supply --mesh-motor-points "
              "measured from the raw DAE.")
    if center_of_mass is not None:
        print("SDF inertial center of mass (model frame): "
              f"({center_of_mass[0]:+.6f}, {center_of_mass[1]:+.6f}, "
              f"{center_of_mass[2]:+.6f}) m")
        print("[INFO] Model CG assumes masses and inertial poses are specified "
              "for all significant components.")
    else:
        print("[INFO] No positive-mass SDF inertials found; model CG unavailable.")
    print("=" * 68 + "\n")
    return WheelbaseResult(centers, pairs, distances, mean_wheelbase, real_m,
                           error_pct, within_tolerance, match, mesh_points_match,
                           center_of_mass)


def run_collision(path: Path, model_name: str | None) -> int:
    """Run analytical SDF box projection and print individual and total areas."""
    try:
        root = ET.parse(path).getroot()
        model = select_sdf_model(root, model_name)
        boxes = list(box_collisions(model))
    except (OSError, ET.ParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not boxes:
        print("No box collision geometries found.")
        return 1
    total = np.zeros(3)
    print(f"SDF:   {path}\nModel: {model.get('name', '<unnamed-model>')}\nFrame: model\n")
    for link_name, collision_name, size, pose in boxes:
        areas = box_projected_areas(size, pose.rotation)
        total += areas
        print(f"{link_name}/{collision_name}")
        print(f"  size: {size[0]:.6g} {size[1]:.6g} {size[2]:.6g} m")
        print(f"  flow X -> YZ: {areas[0]:.6f} m^2")
        print(f"  flow Y -> XZ: {areas[1]:.6f} m^2")
        print(f"  flow Z -> XY: {areas[2]:.6f} m^2")
    if len(boxes) > 1:
        print("\nSum of individual box projections (overlap is not removed)")
        print(f"  flow X -> YZ: {total[0]:.6f} m^2")
        print(f"  flow Y -> XZ: {total[1]:.6f} m^2")
        print(f"  flow Z -> XY: {total[2]:.6f} m^2")
    return 0


def projection_indices(flow_axis: str) -> tuple[int, int]:
    """Return vertex-coordinate indices perpendicular to a flow axis."""
    return {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[flow_axis]


def raster_geometric_centroid(
    mask: np.ndarray,
    minimum: np.ndarray,
    height: int,
    pixels_per_meter: float,
    padding: int = 4,
) -> np.ndarray:
    """Calculate the physical coordinates of the occupied pixel centers.

    `mask` is a Pillow raster with row 0 at the *top*, while geometric plane
    V increases upward. The returned point is the area-weighted centroid of
    the raster union, in the projection plane's (U, V) model coordinates.
    Summing by rows/columns avoids allocating millions of pixel-index pairs.
    """
    n_occupied = int(np.count_nonzero(mask))
    if n_occupied == 0:
        raise ValueError("empty projected mask; cannot determine its centroid")
    count_by_col = np.count_nonzero(mask, axis=0).astype(np.float64)
    count_by_row = np.count_nonzero(mask, axis=1).astype(np.float64)
    mean_column_center = float(np.dot(
        np.arange(mask.shape[1], dtype=float) + 0.5, count_by_col
    )) / n_occupied
    mean_row_center = float(np.dot(
        np.arange(mask.shape[0], dtype=float) + 0.5, count_by_row
    )) / n_occupied
    return np.array([
        minimum[0] + (mean_column_center - padding) / pixels_per_meter,
        minimum[1] + (height - padding - mean_row_center) / pixels_per_meter,
    ], dtype=float)


def projected_moment_arm_per_newton(
    axis: str,
    centroid_uv: np.ndarray,
    center_of_mass: np.ndarray,
) -> np.ndarray:
    """Return tau/F_axis = (r_centroid-r_CG) cross e_axis (unit: metres).

    Only transverse offsets are used. Unknown depth along the flow direction
    does not contribute to moment from a force parallel to that axis.
    """
    u, v = projection_indices(axis)
    offset = np.zeros(3)
    offset[u] = centroid_uv[0] - center_of_mass[u]
    offset[v] = centroid_uv[1] - center_of_mass[v]
    direction = np.zeros(3)
    direction[{"x": 0, "y": 1, "z": 2}[axis]] = 1.0
    return np.cross(offset, direction)


def load_mesh_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load DAE as in the original script, with a small fallback for body.dae.

    Trimesh's Collada loader relies on the optional `pycollada` package. If it
    is absent, parse a *simple, untransformed, single-geometry* DAE directly.
    Complex DAE files should be read via trimesh/pycollada as before.
    """
    try:
        import trimesh
        loaded = trimesh.load(path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            meshes = [g.copy() for g in loaded.dump()
                      if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                raise RuntimeError(f"No triangle mesh found in {path}")
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise RuntimeError(f"Unsupported mesh type: {type(loaded)}")
        return np.asarray(mesh.vertices), np.asarray(mesh.faces)
    except ImportError as exc:
        print(f"Trimesh loader unavailable ({exc}); using basic COLLADA reader.",
              file=sys.stderr)
        return _read_simple_collada(path)


def _read_simple_collada(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one untransformed COLLADA triangles mesh without pycollada.

    This supports the provided PX4 Vision body.dae, not arbitrary COLLADA
    hierarchies, non-triangle primitives, or transformed visual scenes.
    """
    root = ET.parse(path).getroot()
    geometry = root.findall(".//{*}library_geometries/{*}geometry")
    if len(geometry) != 1:
        raise ValueError("basic COLLADA reader requires one geometry")
    scene_nodes = root.findall(".//{*}visual_scene/{*}node")
    transform_tags = {"matrix", "rotate", "translate", "scale", "skew", "lookat"}
    for node in scene_nodes:
        if any(child.tag.split("}")[-1] in transform_tags for child in node):
            raise ValueError("scene contains node transforms; install pycollada")
    mesh = geometry[0].find("{*}mesh")
    if mesh is None:
        raise ValueError("COLLADA geometry has no mesh")
    vtx = mesh.find("{*}vertices")
    if vtx is None:
        raise ValueError("COLLADA geometry has no vertices")
    inp = next((a for a in vtx.findall("{*}input")
                if a.get("semantic") == "POSITION"), None)
    if inp is None:
        raise ValueError("COLLADA vertices have no POSITION input")
    source_id = inp.get("source", "").lstrip("#")
    source = next((a for a in mesh.findall("{*}source")
                   if a.get("id") == source_id), None)
    if source is None or source.find("{*}float_array") is None:
        raise ValueError("COLLADA POSITION source was not found")
    vertices = np.fromstring(source.find("{*}float_array").text,
                             sep=" ", dtype=np.float64).reshape(-1, 3)
    all_faces = []
    for triangles in mesh.findall("{*}triangles"):
        inputs = triangles.findall("{*}input")
        if not inputs:
            continue
        stride = max(int(a.get("offset", "0")) for a in inputs) + 1
        position_offset = next((int(a.get("offset", "0")) for a in inputs
                                if a.get("semantic") == "VERTEX"), None)
        if position_offset is None:
            raise ValueError("triangle primitive lacks VERTEX input")
        for p in triangles.findall("{*}p"):
            indices = np.fromstring(p.text, sep=" ", dtype=np.int64)
            all_faces.append(indices.reshape(-1, 3, stride)[:, :, position_offset])
    if not all_faces:
        raise ValueError("basic COLLADA reader only supports triangle primitives")
    return vertices, np.concatenate(all_faces, axis=0)


def plot_projected_views(
    projections: list[ProjectionResult],
    pixels_per_meter: float,
    output: Path,
    scale: np.ndarray,
    rpy: np.ndarray,
    wheelbase: WheelbaseResult | None = None,
    show_wheelbase: bool = False,
    show_cg: bool = False,
) -> None:
    """Draw a three-view silhouette figure with geometric centroid markers.

    Plotting consumes the *same raster masks* used for area and centroid.
    It does not change the original numerical raster calculation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1, 3, figsize=(14.7, 5.8))
    fig.suptitle("PX4 Vision body mesh: projected silhouettes (no propellers)",
                 fontsize=16, fontweight="semibold", y=0.965)
    cmap = ListedColormap(["#ffffff", "#c9d5dc"])
    axes_names = {
        "x": ("Front view / YZ plane", "Body Y (m)", "Body Z (m)"),
        "y": ("Side view / XZ plane", "Body X (m)", "Body Z (m)"),
        "z": ("Top view / XY plane", "Body X (m)", "Body Y (m)"),
    }
    for ax, result in zip(axes, projections):
        axis = result.axis
        minimum, maximum = result.minimum, result.maximum
        width, height = result.width, result.height
        area, mask = result.area_m2, result.mask
        title, xlabel, ylabel = axes_names[axis]
        # Pixel-center range in metres. Image rows are top-to-bottom by design.
        pad = 4
        left = float(minimum[0] - pad / pixels_per_meter)
        right = left + width / pixels_per_meter
        bottom = float(minimum[1] - pad / pixels_per_meter)
        top = bottom + height / pixels_per_meter
        ax.imshow(mask, cmap=cmap, vmin=0, vmax=1,
                  extent=(left, right, bottom, top), origin="upper",
                  interpolation="nearest", aspect="equal", zorder=2)
        # Outline, as in the earlier polygon-union plot.
        xs = np.linspace(left + 0.5 / pixels_per_meter,
                         right - 0.5 / pixels_per_meter, width)
        ys = np.linspace(bottom + 0.5 / pixels_per_meter,
                         top - 0.5 / pixels_per_meter, height)
        ax.contour(xs, ys, mask[::-1, :], levels=[0.5],
                   colors=["#465d6b"], linewidths=0.35, zorder=3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}\nA$_{{{axis}}}$ = {area:.6f} m$^2$",
                     fontsize=13, pad=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.12, linewidth=0.7)
        cu, cv = result.centroid_uv_m
        ax.scatter([cu], [cv], marker="o", s=67, facecolor="#cf4e47",
                   edgecolor="white", linewidth=1.0, zorder=8,
                   label="Geometric centroid")
        ax.annotate("C", xy=(cu, cv), xytext=(6, 5),
                    textcoords="offset points", color="#9d302c",
                    fontsize=10, fontweight="bold", zorder=9)
        ax.text(0.025, 0.025, f"Centroid = ({cu:+.4f}, {cv:+.4f}) m",
                transform=ax.transAxes, fontsize=8.6, zorder=10,
                bbox=dict(boxstyle="round,pad=0.33", facecolor="white",
                          edgecolor="#d2d8dd", alpha=0.95))
        if show_wheelbase and wheelbase is not None and axis == "z":
            # SDF rotor centers already lie in the model XY frame (metres).
            for (a, b) in wheelbase.pairs:
                p, q = wheelbase.centers[a], wheelbase.centers[b]
                ax.plot([p[0], q[0]], [p[1], q[1]], linestyle="--",
                        color="#c77929", linewidth=1.15, alpha=0.8, zorder=4)
            rotor_xy = np.vstack([wheelbase.centers[k][:2]
                                  for k in sorted(wheelbase.centers)])
            ax.scatter(rotor_xy[:, 0], rotor_xy[:, 1], marker="o", s=32,
                       facecolor="#c77929", edgecolor="white", linewidth=0.6,
                       zorder=5, label="SDF rotor centers")
        if show_cg and wheelbase is not None and wheelbase.center_of_mass_m is not None:
            u, v = projection_indices(axis)
            center = wheelbase.center_of_mass_m
            ax.scatter([center[u]], [center[v]], marker="*", s=110,
                       facecolor="#217b82", edgecolor="white", linewidth=0.65,
                       zorder=7, label="SDF center of mass")
        ax.legend(loc="upper right", fontsize=7.6, framealpha=0.92,
                  facecolor="white")
        ax.set_axisbelow(False)
        # imshow has sticky edges, so add margins explicitly.
        min_x, max_x, min_y, max_y = left, right, bottom, top
        if show_wheelbase and wheelbase is not None and axis == "z":
            # Include model-frame SDF motor centers even if the mesh is wrongly
            # scaled (e.g. 0.0001 instead of 0.001), exposing misalignment.
            motor_xy = np.vstack([c[:2] for c in wheelbase.centers.values()])
            min_x = min(left, float(motor_xy[:, 0].min()))
            max_x = max(right, float(motor_xy[:, 0].max()))
            min_y = min(bottom, float(motor_xy[:, 1].min()))
            max_y = max(top, float(motor_xy[:, 1].max()))
        if show_cg and wheelbase is not None and wheelbase.center_of_mass_m is not None:
            u, v = projection_indices(axis)
            min_x = min(min_x, float(wheelbase.center_of_mass_m[u]))
            max_x = max(max_x, float(wheelbase.center_of_mass_m[u]))
            min_y = min(min_y, float(wheelbase.center_of_mass_m[v]))
            max_y = max(max_y, float(wheelbase.center_of_mass_m[v]))
        du, dv = max_x - min_x, max_y - min_y
        ax.set_xlim(min_x - 0.06 * du, max_x + 0.06 * du)
        ax.set_ylim(min_y - 0.06 * dv, max_y + 0.06 * dv)
    fig.text(0.5, 0.045,
             f"Mesh scale: {scale[0]:g} m/unit; "
             f"SDF RPY: ({rpy[0]:g}, {rpy[1]:g}, {rpy[2]:g}) rad; "
             f"Raster: {pixels_per_meter:g} pixels/m",
             ha="center", va="center", fontsize=10)
    fig.text(0.5, 0.012,
             "Red marker = projected geometric centroid (CoP approximation only, not measured CoP)",
             ha="center", va="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=[0, 0.082, 1, 0.93], w_pad=1.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Plot written: {output}")


def run_raster(
    path: Path,
    pixels_per_meter: float,
    output_directory: Path,
    output_prefix: str,
    scale: np.ndarray,
    translation: np.ndarray,
    rpy: np.ndarray,
    plot_file: Path | None = None,
    wheelbase: WheelbaseResult | None = None,
    show_wheelbase_on_plot: bool = False,
    show_cg_on_plot: bool = False,
    centroid_csv: Path | None = None,
) -> int:
    """Compute raster areas and centroids; write one three-panel PNG plot."""
    try:
        from PIL import Image, ImageDraw
        import matplotlib
    except ImportError as exc:
        print(f"error: raster mode requires Pillow and Matplotlib: {exc}",
              file=sys.stderr)
        return 2
    if pixels_per_meter <= 0:
        print("error: --pixels-per-meter must be positive", file=sys.stderr)
        return 2
    try:
        original_vertices, faces = load_mesh_vertices_faces(path)
    except (OSError, ET.ParseError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: unable to load mesh: {exc}", file=sys.stderr)
        return 2
    vertices = np.asarray(original_vertices, dtype=np.float64) * scale
    vertices = (rotation_matrix_rpy(*rpy) @ vertices.T).T + translation
    faces = np.asarray(faces, dtype=np.int64)
    print(f"Triangles: {len(faces)}")
    print(f"Vertices:  {len(vertices)}")
    print(f"Bounds after SDF transform:\n{np.vstack((vertices.min(0), vertices.max(0)))}")
    projections: list[ProjectionResult] = []
    for axis, plane in (("x", "YZ"), ("y", "XZ"), ("z", "XY")):
        u, v = projection_indices(axis)
        projected = vertices[:, [u, v]]
        minimum, maximum = projected.min(0), projected.max(0)
        extent = maximum - minimum
        padding = 4
        width = int(math.ceil(extent[0] * pixels_per_meter)) + 2 * padding
        height = int(math.ceil(extent[1] * pixels_per_meter)) + 2 * padding
        image = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(image)
        for face in faces:
            triangle = projected[face]
            edge1 = triangle[1] - triangle[0]
            edge2 = triangle[2] - triangle[0]
            cross2 = edge1[0] * edge2[1] - edge1[1] * edge2[0]
            if abs(cross2) < 1e-14:
                continue
            pixel = (triangle - minimum) * pixels_per_meter
            pixel[:, 0] += padding
            pixel[:, 1] = height - padding - pixel[:, 1]
            draw.polygon([tuple(point) for point in pixel], fill=1)
        # Keep the exact same raster area formula as the supplied code.
        mask = np.asarray(image, dtype=np.uint8)
        area = np.count_nonzero(mask) / pixels_per_meter**2
        centroid = raster_geometric_centroid(mask, minimum, height,
                                            pixels_per_meter, padding)
        u_name, v_name = plane
        print(f"Flow along {axis.upper()}: {plane} area = {area:.8f} m^2")
        print(f"  Projected geometric centroid: "
              f"{u_name} = {centroid[0]:+.6f} m, "
              f"{v_name} = {centroid[1]:+.6f} m")
        if wheelbase is not None and wheelbase.center_of_mass_m is not None:
            cg = wheelbase.center_of_mass_m
            delta = centroid - cg[[u, v]]
            arm = projected_moment_arm_per_newton(axis, centroid, cg)
            print(f"  Centroid - SDF CG in {plane}: "
                  f"({delta[0]:+.6f}, {delta[1]:+.6f}) m")
            print(f"  Drag moment per +1 N along {axis.upper()} "
                  f"(tau/F_{axis}): "
                  f"({arm[0]:+.6f}, {arm[1]:+.6f}, {arm[2]:+.6f}) m")
        projections.append(ProjectionResult(axis, plane, minimum, maximum,
                                            width, height, area, mask, centroid))
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    plot_path = (plot_file if plot_file is not None else
                 output_directory / f"{output_prefix}_views.png")
    plot_projected_views(projections, pixels_per_meter, plot_path, scale, rpy,
                         wheelbase=wheelbase,
                         show_wheelbase=show_wheelbase_on_plot,
                         show_cg=show_cg_on_plot)
    if centroid_csv is not None:
        write_centroid_csv(projections, centroid_csv,
                           wheelbase.center_of_mass_m if wheelbase else None)
    return 0


def write_centroid_csv(
    projections: list[ProjectionResult],
    path: Path,
    cg: np.ndarray | None,
) -> None:
    """Optionally export numerical areas, centroids, and moment arms as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "flow_axis", "projection_plane", "area_m2", "coord_u", "coord_v",
            "centroid_u_m", "centroid_v_m", "cg_u_m", "cg_v_m",
            "offset_u_m", "offset_v_m", "tau_over_F_x_m", "tau_over_F_y_m",
            "tau_over_F_z_m",
        ])
        for p in projections:
            u, v = projection_indices(p.axis)
            if cg is None:
                tail = [""] * 7
            else:
                moment_arm = projected_moment_arm_per_newton(
                    p.axis, p.centroid_uv_m, cg
                )
                tail = [
                    cg[u], cg[v], p.centroid_uv_m[0] - cg[u],
                    p.centroid_uv_m[1] - cg[v], *moment_arm,
                ]
            writer.writerow([
                p.axis.upper(), p.plane, p.area_m2,
                *p.plane, *p.centroid_uv_m, *tail,
            ])
    print(f"Centroid CSV written: {path}")


def mesh_points_argument(value: str) -> np.ndarray:
    """Parse six raw DAE coordinates x0,y0,z0,x1,y1,z1."""
    try:
        values = np.asarray([float(item) for item in value.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected six comma-separated coordinates") from exc
    if len(values) != 6 or not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError("expected six finite numbers x0,y0,z0,x1,y1,z1")
    return values.reshape(2, 3)


def vector_argument(value: str) -> np.ndarray:
    """Parse a CLI vector written as three comma-separated numbers."""
    try:
        result = np.asarray([float(item) for item in value.split(",")], dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers") from exc
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return result


def parse_args() -> argparse.Namespace:
    """Define and parse the collision/raster/scalecheck CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("collision", "raster", "scalecheck"),
        required=True,
        help="SDF scale/rotor-spacing check, box collisions, or raster silhouette",
    )
    parser.add_argument("path", nargs="?", type=Path,
                        help="SDF file (collision) or DAE mesh file (raster)")
    parser.add_argument("--input", dest="input_opt", type=Path,
                        help="alternative to positional path")
    parser.add_argument("--model", help="model name for an SDF containing multiple models")
    parser.add_argument(
        "--sdf", type=Path,
        help="in raster mode, path to model.sdf for rotor wheelbase and mesh-scale checks",
    )
    parser.add_argument(
        "--real-wheelbase-mm", type=float, default=286.0,
        help="reference real diagonal rotor-to-rotor wheelbase in mm (default 286)",
    )
    parser.add_argument(
        "--wheelbase-tolerance-pct", type=float, default=1.0,
        help="PASS threshold for SDF vs real wheelbase in percent (default 1)",
    )
    parser.add_argument(
        "--strict-scalecheck", "--strict-wheelbase", dest="strict_scalecheck",
        action="store_true",
        help="return nonzero exit status if SDF wheelbase/CLI transform fails checks "
             "(--strict-wheelbase is a compatibility alias)",
    )
    parser.add_argument(
        "--mesh-motor-points", type=mesh_points_argument,
        help="optional two independently measured rotor centers in *unscaled* DAE "
             "coordinates: x0,y0,z0,x1,y1,z1 (not inferred from SDF)",
    )
    parser.add_argument(
        "--show-wheelbase-on-plot", action="store_true",
        help="overlay SDF rotor centers and the two wheelbase diagonals on top view",
    )
    parser.add_argument(
        "--show-cg-on-plot", action="store_true",
        help="overlay mass-weighted SDF inertial CG on the three projected views "
             "(requires --sdf and link inertial data)",
    )
    parser.add_argument(
        "--centroid-csv", type=Path, default=None,
        help="optional CSV file for areas, projected centroids, and CG offsets",
    )
    parser.add_argument(
        "--pixels-per-meter",
        type=float,
        default=10000.0,
        help="raster resolution (default: 10000, or 0.1 mm/pixel)",
    )
    parser.add_argument(
        "--output-prefix",
        default="px4vision_projection",
        help="prefix for the combined PNG (default: px4vision_projection)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("."),
        help="directory for the combined PNG (default: current directory)",
    )
    parser.add_argument(
        "--scale",
        type=vector_argument,
        default=np.array((0.001, 0.001, 0.001)),
        help="raster mesh XYZ scale as x,y,z (default: 0.001,0.001,0.001)",
    )
    parser.add_argument(
        "--translation",
        type=vector_argument,
        default=np.zeros(3),
        help="raster mesh XYZ translation in metres (default: 0,0,0)",
    )
    parser.add_argument(
        "--rpy",
        type=vector_argument,
        default=np.array((1.5707, 0.0, 0.01)),
        help="raster mesh SDF roll,pitch,yaw in radians (default: 1.5707,0,0.01)",
    )
    parser.add_argument("--plot-file", type=Path, default=None,
                        help="optional custom filename for the combined silhouette plot")
    args = parser.parse_args()
    args.input = args.input_opt if args.input_opt is not None else args.path
    if args.input is None:
        parser.error("provide an input path or --input")
    if args.mode in ("collision", "scalecheck") and args.input.suffix.lower() != ".sdf":
        parser.error("--mode collision / scalecheck requires an .sdf input file")
    if args.mode == "raster" and args.input.suffix.lower() != ".dae":
        parser.error("--mode raster requires a .dae input file")
    if args.mode != "raster" and args.sdf:
        parser.error("--sdf is for raster mode; for SDF-only use --mode scalecheck model.sdf")
    if args.mode == "raster" and args.show_wheelbase_on_plot and not args.sdf:
        parser.error("--show-wheelbase-on-plot requires --sdf")
    if args.show_cg_on_plot and (args.mode != "raster" or not args.sdf):
        parser.error("--show-cg-on-plot requires --mode raster and --sdf")
    if args.mode == "raster" and args.mesh_motor_points is not None and not args.sdf:
        parser.error("--mesh-motor-points requires --sdf")
    if args.centroid_csv is not None and args.mode != "raster":
        parser.error("--centroid-csv is only supported in --mode raster")
    return args


def main() -> int:
    """Dispatch to projection mode, preserving the original area computation."""
    args = parse_args()
    if args.mode == "collision":
        return run_collision(args.input, args.model)

    sdf_file = args.input if args.mode == "scalecheck" else args.sdf
    wheelbase = None
    if sdf_file is not None:
        try:
            wheelbase = check_wheelbase(
                sdf_file, args.model,
                real_wheelbase_mm=args.real_wheelbase_mm,
                tolerance_pct=args.wheelbase_tolerance_pct,
                mesh_path=(args.input if args.mode == "raster" else None),
                cli_scale=args.scale,
                cli_translation=args.translation,
                cli_rpy=args.rpy,
                mesh_motor_points=args.mesh_motor_points,
            )
        except (OSError, ET.ParseError, ValueError) as exc:
            print(f"error: could not validate SDF wheelbase: {exc}", file=sys.stderr)
            return 2
        if args.strict_scalecheck and (
            not wheelbase.within_tolerance or
            wheelbase.visual_transform_matches is False or
            wheelbase.mesh_points_match is False
        ):
            print("error: strict scalecheck validation failed", file=sys.stderr)
            return 1
    if args.mode == "scalecheck":
        return 0

    if args.show_cg_on_plot and wheelbase is not None and wheelbase.center_of_mass_m is None:
        print("warning: --show-cg-on-plot requested but SDF inertial CG is "
              "unavailable; the plot will show centroid markers only.",
              file=sys.stderr)

    return run_raster(
        args.input,
        args.pixels_per_meter,
        args.output_directory,
        args.output_prefix,
        args.scale,
        args.translation,
        args.rpy,
        args.plot_file,
        wheelbase=wheelbase,
        show_wheelbase_on_plot=args.show_wheelbase_on_plot,
        show_cg_on_plot=args.show_cg_on_plot,
        centroid_csv=args.centroid_csv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
