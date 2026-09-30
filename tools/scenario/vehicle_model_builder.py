"""Materialize per-scenario Gazebo inertials without modifying source models.

Only the selected link is scaled. Meshes remain shared with the source model;
generated SDFs and resolved SI values are kept next to scenario.yaml.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIRE_ROOT = PROJECT_ROOT / "gz/FIRE_moonshot_gazebo"
INERTIA_TERMS = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")


def positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number greater than zero")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number greater than zero") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return number


@dataclass(frozen=True)
class VehicleConfig:
    model_name: str
    target_link: str = "base_link"
    mass_scale: float = 1.0
    inertia_scale: float = 1.0

    @classmethod
    def from_common(cls, common: dict) -> VehicleConfig | None:
        vehicle = common.get("vehicle")
        if vehicle is None:
            return None
        if not isinstance(vehicle, dict):
            raise ValueError("common.vehicle must be a mapping")
        inertial = vehicle.get("inertial", {})
        if not isinstance(inertial, dict) or inertial.get("mode", "scale") != "scale":
            raise ValueError("common.vehicle.inertial must use mode: scale")
        model_name = vehicle.get("model_name")
        target_link = inertial.get("target_link", "base_link")
        if not isinstance(model_name, str) or not model_name or Path(model_name).name != model_name:
            raise ValueError("common.vehicle.model_name must be a model name")
        if not isinstance(target_link, str) or not target_link:
            raise ValueError("common.vehicle.inertial.target_link must be a link name")
        return cls(model_name, target_link,
                   positive_finite(inertial.get("mass_scale", 1), "mass_scale"),
                   positive_finite(inertial.get("inertia_scale", 1), "inertia_scale"))


def resource_file(value: str, kind: str) -> Path:
    """Resolve a direct path or a FIRE / GZ_SIM_RESOURCE_PATH resource."""
    direct = Path(os.path.expandvars(value)).expanduser()
    if direct.is_file():
        return direct.resolve()
    relative = (Path(value) / "model.sdf" if kind == "models" else
                Path(value if value.endswith(".sdf") else f"{value}.sdf"))
    roots = [FIRE_ROOT / kind]
    roots.extend(Path(p) for p in os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(os.pathsep) if p)
    for root in roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Gazebo {kind} resource not found: {value}")


def _inertial_values(link: ET.Element) -> dict:
    name = link.get("name", "unnamed")
    inertial = link.find("inertial")
    if inertial is None or inertial.get("auto", "false").lower() in ("1", "true"):
        raise ValueError(f"Link {name} requires explicit inertial values (auto inertia is unsupported)")
    mass = positive_finite(inertial.findtext("mass"), f"{name}.mass")
    tensor = inertial.find("inertia")
    if tensor is None:
        raise ValueError(f"Link {name} has no inertia tensor")
    terms = {key: float(tensor.findtext(key, "0")) for key in INERTIA_TERMS}
    matrix = np.array([[terms["ixx"], terms["ixy"], terms["ixz"]],
                       [terms["ixy"], terms["iyy"], terms["iyz"]],
                       [terms["ixz"], terms["iyz"], terms["izz"]]])
    if not np.isfinite(matrix).all():
        raise ValueError(f"Link {name} inertia must be finite")
    moments = np.linalg.eigvalsh(matrix)
    tolerance = 1e-10 * float(np.max(np.abs(moments)))
    if moments[0] <= 0 or moments[0] + moments[1] + tolerance < moments[2]:
        raise ValueError(f"Link {name} inertia must be positive definite and satisfy the triangle inequality")
    return {"mass_kg": mass, "inertia_kg_m2": terms,
            "inertial_pose": inertial.findtext("pose", "0 0 0 0 0 0")}


def _relocate_uris(root: ET.Element, source: Path, model_name: str | None = None) -> None:
    """Keep local asset paths valid when copying an SDF to a run directory."""
    for uri in root.iter("uri"):
        value = (uri.text or "").strip()
        prefix = f"model://{model_name}/" if model_name else None
        if prefix and value.startswith(prefix):
            uri.text = str((source.parent / value[len(prefix):]).resolve())
        elif value and "://" not in value and not Path(value).is_absolute():
            uri.text = str((source.parent / value).resolve())


def _write_xml(tree: ET.ElementTree, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    # Readers (including the other autopilot launcher) must never see a
    # partially written SDF.
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            tree.write(temporary, encoding="utf-8", xml_declaration=True)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_vehicle_model(config: VehicleConfig, source: Path, run_dir: Path) -> Path:
    """Always scale the original model, never a previously generated model."""
    source = source.resolve()
    try:
        tree = ET.parse(source)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid model SDF {source}: {exc}") from exc
    model = tree.getroot().find("model")
    if model is None or model.get("name") != config.model_name:
        raise ValueError(f"Expected model {config.model_name!r} in {source}")
    if model.find("include") is not None or model.find("model") is not None:
        raise ValueError("Inertial scaling currently requires a flat model with explicit links")
    links = model.findall("link")
    targets = [link for link in links if link.get("name") == config.target_link]
    if len(targets) != 1:
        raise ValueError(f"Expected exactly one link {config.target_link!r} in {source}")
    before = {link.attrib["name"]: _inertial_values(link) for link in links}
    target = targets[0]
    inertial = target.find("inertial")
    mass_scale = positive_finite(config.mass_scale, "mass_scale")
    inertia_scale = positive_finite(config.inertia_scale, "inertia_scale")
    inertial.find("mass").text = format(before[config.target_link]["mass_kg"] * mass_scale, ".17g")
    for key in INERTIA_TERMS:
        element = inertial.find(f"inertia/{key}")
        if element is not None:
            element.text = format(before[config.target_link]["inertia_kg_m2"][key] * inertia_scale, ".17g")
    after = {link.attrib["name"]: _inertial_values(link) for link in links}
    _relocate_uris(model, source, config.model_name)
    destination = (run_dir / "generated/vehicle/model.sdf").resolve()
    if destination == source:
        raise ValueError("Source model cannot be the generated model")
    _write_xml(tree, destination)
    manifest = {
        "scaling": asdict(config),
        "source_model": {"path": str(source), "sha256": _sha256(source)},
        "generated_model": {"path": str(destination), "sha256": _sha256(destination)},
        "links": after,
        "total_mass_kg": sum(link["mass_kg"] for link in after.values()),
        "source_total_mass_kg": sum(link["mass_kg"] for link in before.values()),
    }
    (destination.parent.parent / "resolved_vehicle.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return destination


def prepare_ardupilot_world(config: VehicleConfig, world: str, run_dir: Path) -> Path:
    """Preserve the include's entity name, pose and ArduPilot adapter plugins."""
    source_world = resource_file(world, "worlds")
    try:
        tree = ET.parse(source_world)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid world SDF {source_world}: {exc}") from exc
    world_element = tree.getroot().find("world")
    if world_element is None:
        raise ValueError(f"Expected a world in {source_world}")
    includes = [item for item in world_element.findall("include")
                if item.findtext("uri", "").strip().rstrip("/") == f"model://{config.model_name}"]
    if len(includes) != 1:
        raise ValueError(f"World {source_world} must include model://{config.model_name} exactly once")
    source_model = resource_file(config.model_name, "models")
    model_path = generate_vehicle_model(config, source_model, run_dir)
    _relocate_uris(world_element, source_world)
    includes[0].find("uri").text = str(model_path)
    destination = (run_dir / "generated/world_ardupilot.sdf").resolve()
    if destination == source_world:
        raise ValueError("Source world cannot be the generated world")
    _write_xml(tree, destination)
    (destination.parent / "resolved_ardupilot_world.yaml").write_text(yaml.safe_dump({
        "source_world": {"path": str(source_world), "sha256": _sha256(source_world)},
        "generated_world": {"path": str(destination), "sha256": _sha256(destination)},
        "world_name": world_element.get("name"),
    }, sort_keys=False), encoding="utf-8")
    return destination
