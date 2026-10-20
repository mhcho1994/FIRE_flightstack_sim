"""Send scenario wind commands through Gazebo's CLI in the server's environment.

The wind_info service acknowledges the target velocity, not the filtered wind
velocity used by the aerodynamic plugins. No Gazebo Python bindings are needed.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
import re
import subprocess
import time
from typing import Callable
import xml.etree.ElementTree as ET


def parse_wind_m_s(value: object) -> tuple[float, float, float]:
    """Validate an ENU velocity without converting it to the aircraft frame."""
    message = "common.scenario.wind_m_s must contain three finite numbers [east, north, up] in m/s"
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(message)
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in value):
        raise ValueError(message)
    try:
        x, y, z = (float(v) for v in value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError(message)
    return x, y, z


def world_name_from_sdf(path: Path) -> str:
    """Generated world filenames need not match the transport namespace."""
    world = ET.parse(path).getroot().find("world")
    name = world.get("name", "") if world is not None else ""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Expected a world with a valid name in {path}")
    return name


def _parse_wind_info(text: str) -> tuple[tuple[float, float, float], bool] | None:
    """Read Wind's text-protobuf fields, including omitted zero components."""
    vector = re.search(r"\blinear_velocity\s*\{([^{}]*)\}", text)
    enabled = re.search(r"\benable_wind\s*:\s*(true|false)\b", text)
    if vector is None and enabled is None:
        return None
    body = vector.group(1) if vector else ""
    values = []
    for axis in "xyz":
        field = re.search(rf"\b{axis}\s*:\s*([^\s,]+)", body)
        try:
            value = float(field.group(1)) if field else 0.0
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return (values[0], values[1], values[2]), bool(enabled and enabled.group(1) == "true")


def set_world_wind(
    world_name: str,
    wind_m_s: tuple[float, float, float],
    *,
    env: dict[str, str],
    log_path: Path,
    process: subprocess.Popen | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    timeout: float = 30.0,
) -> None:
    """Wait for WindEffects, publish even zero wind, and require acknowledgement.

    Retry publication after transport discovery delays. A missing plugin or a
    mismatched acknowledgement fails the attempt instead of flying in still air.
    """
    wind = parse_wind_m_s(wind_m_s)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", world_name):
        raise ValueError(f"Invalid Gazebo world name: {world_name!r}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Wind startup timeout must be positive and finite")
    report = {
        "world": world_name, "partition": env.get("GZ_PARTITION"),
        "frame": "ENU", "requested_wind_m_s": list(wind), "status": "pending",
    }

    def save() -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def check_running() -> None:
        if cancelled():
            raise InterruptedError("Wind setup interrupted")
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Gazebo exited during wind setup (status {process.returncode})")

    save()
    print(f"[WIND] world={world_name} ENU={wind} m/s; waiting for WindEffects", flush=True)
    deadline = time.monotonic() + timeout
    next_publish = 0.0
    published = False
    x, y, z = wind
    command = f"linear_velocity: {{x: {x}, y: {y}, z: {z}}}, enable_wind: true"
    last_error = "wind_info service has not replied"
    try:
        while time.monotonic() < deadline:
            check_running()
            remaining = max(0.001, deadline - time.monotonic())
            try:
                result = subprocess.run(
                    ["gz", "service", "-s", f"/world/{world_name}/wind_info",
                     "--reqtype", "gz.msgs.Empty", "--reptype", "gz.msgs.Wind",
                     "--timeout", str(max(1, min(1000, int(remaining * 1000)))), "--req", ""],
                    env=env, text=True, capture_output=True, timeout=min(3.0, remaining),
                )
                check_running()
                info = _parse_wind_info(result.stdout) if result.returncode == 0 else None
                if info is not None:
                    velocity, enabled = info
                    if published and enabled and all(
                        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
                        for a, b in zip(velocity, wind)
                    ):
                        report.update(status="acknowledged", acknowledged_wind_m_s=list(velocity),
                                      enable_wind=enabled)
                        save()
                        print(f"[WIND] target acknowledged: ENU={velocity} m/s", flush=True)
                        return
                    last_error = f"last target={velocity}, enable_wind={enabled}"
                    if next_publish <= time.monotonic() < deadline:
                        subprocess.run(
                            ["gz", "topic", "-t", f"/world/{world_name}/wind",
                             "-m", "gz.msgs.Wind", "-p", command],
                            env=env, text=True, capture_output=True, check=True,
                            timeout=max(0.001, min(5.0, deadline - time.monotonic())),
                        )
                        published = True
                        next_publish = time.monotonic() + 1.0
                elif result.stdout.strip() or result.stderr.strip():
                    last_error = (result.stderr.strip() or result.stdout.strip())[-500:]
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                last_error = str(exc)
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        check_running()
        raise TimeoutError(
            f"Wind target {wind} was not acknowledged by /world/{world_name}/wind_info "
            f"within {timeout:g}s. Check WindEffects and GZ_PARTITION. {last_error}"
        )
    except Exception as exc:
        report.update(status="interrupted" if isinstance(exc, InterruptedError) else "failed",
                      error=str(exc))
        save()
        raise
