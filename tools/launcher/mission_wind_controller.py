"""Mission wind controller and private Gazebo Transport worker.

The controller starts this script with ``--worker WORLD`` in the server's
process environment. Gazebo bindings load only in that child process, keeping
its protobuf runtime and partition separate from the launcher.
"""

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
import os
import random
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

from gazebo_wind import set_world_wind


def _config(cls, value, name):
    if value is None:
        return cls()
    if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
        raise ValueError(f"{name} must be a mapping with known settings")
    return cls(**value)


def _number(name, value, *, minimum=0.0, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else f'>= {minimum}'}")


@dataclass(frozen=True)
class GustConfig:
    magnitude_m_s: float = 0.0
    start_s: float = 0.0
    duration_s: float = 5.0
    period_s: float = 0.0
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0

    def __post_init__(self):
        for name in ('magnitude_m_s', 'start_s', 'duration_s', 'period_s', 'azimuth_deg'):
            _number(f"gust.{name}", getattr(self, name), positive=name == 'duration_s')
        _number("gust.elevation_deg", self.elevation_deg, minimum=-90)
        if self.azimuth_deg > 360 or self.elevation_deg > 90:
            raise ValueError("gust angles require azimuth in [0, 360], elevation in [-90, 90]")
        if self.period_s and self.period_s < self.duration_s:
            raise ValueError("gust.period_s must be zero (one pulse) or >= duration_s")

    @classmethod
    def parse(cls, value=None):
        return _config(cls, value, 'gust')


@dataclass(frozen=True)
class TurbulenceConfig:
    magnitude_m_s: float = 0.0
    length_scale_m: tuple = (30.0, 30.0, 10.0)
    reference_speed_m_s: float = 5.0
    azimuth_deg: float = 0.0
    sample_period_s: float = 0.1
    seed: int = 0

    def __post_init__(self):
        for name in ('magnitude_m_s', 'reference_speed_m_s', 'azimuth_deg', 'sample_period_s'):
            _number(f"turbulence.{name}", getattr(self, name),
                    positive=name in ('reference_speed_m_s', 'sample_period_s'))
        if self.azimuth_deg > 360:
            raise ValueError("turbulence.azimuth_deg must be in [0, 360]")
        if not isinstance(self.length_scale_m, (list, tuple)) or len(self.length_scale_m) != 3:
            raise ValueError("turbulence.length_scale_m requires three lengths [u, v, w]")
        for length in self.length_scale_m:
            _number("turbulence.length_scale_m", length, positive=True)
        object.__setattr__(self, 'length_scale_m', tuple(self.length_scale_m))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("turbulence.seed must be a non-negative integer")

    @classmethod
    def parse(cls, value=None):
        return _config(cls, value, 'turbulence')


def one_minus_cosine_gust(elapsed_s, config):
    """Finite pulse with peak magnitude at duration/2; direction points toward ENU.

    U = A/2 * (1 - cos(2*pi*t/T)), equivalent to the spatial 1-cosine
    profile with T=2H/V. period_s=0 gives one pulse, otherwise repeat.
    """
    t = elapsed_s - config.start_s
    if t < 0 or config.magnitude_m_s == 0:
        return (0.0, 0.0, 0.0)
    if config.period_s:
        t %= config.period_s
    if t >= config.duration_s:
        return (0.0, 0.0, 0.0)
    amplitude = config.magnitude_m_s * 0.5 * (1 - math.cos(2 * math.pi * t / config.duration_s))
    azimuth, elevation = math.radians(config.azimuth_deg), math.radians(config.elevation_deg)
    return (amplitude * math.cos(elevation) * math.sin(azimuth),
            amplitude * math.cos(elevation) * math.cos(azimuth), amplitude * math.sin(elevation))


class DrydenTurbulence:
    """MIL-F-8785C translational Dryden shaping filters, fixed V and L.

    With unit spectral-density white noise, a=V/L:
      H_u(s) = sigma*sqrt(2*a)/(s+a)
      H_v,w(s) = sigma*sqrt(3*a)*(s+a/sqrt(3))/(s+a)**2.
    Each output has stationary standard deviation sigma=magnitude_m_s.
    Exact Gaussian state transitions preserve variance at any sample period.
    Fixed simulation ticks make a seed independent of worker polling cadence.
    """

    def __init__(self, config):
        self.config = config
        self.rng = random.Random(config.seed)
        self.tick = 0
        self.state = []
        self.transition = []
        for length in config.length_scale_m:
            # Stationary cascade: Var(x)=1, Var(y)=Cov(x,y)=1/2.
            x = self.rng.gauss(0, 1)
            self.state.append((x, 0.5 * (x + self.rng.gauss(0, 1))))
            h = config.reference_speed_m_s / length * config.sample_period_s
            if not math.isfinite(h) or h <= 0:
                raise ValueError("Dryden V * sample_period / L must be finite and positive")
            decay = math.exp(-h)
            qxx = -math.expm1(-2 * h)
            if h < 1e-3:  # Avoid cancellation in the covariance integrals.
                qxy = h*h * (1 - 4*h/3 + h*h - 8*h**3/15)
                qyy = h**3 * (2/3 - h + 4*h*h/5 - 4*h**3/9)
            elif h > 350:
                qxy = qyy = 0.5
            else:
                qxy = 0.5 * qxx - h * decay**2
                qyy = 0.5 * qxx - h * (1 + h) * decay**2
            l11 = math.sqrt(qxx)
            l21 = qxy / l11
            l22 = math.sqrt(max(0.0, qyy - l21*l21))
            self.transition.append((decay, h*decay, l11, l21, l22))

    def sample(self, elapsed_s):
        if self.config.magnitude_m_s == 0:
            return (0.0, 0.0, 0.0)
        tick = max(0, math.floor(elapsed_s / self.config.sample_period_s + 1e-9))
        if tick < self.tick:
            raise ValueError("Dryden simulation time must not go backwards")
        while self.tick < tick:
            for i, (decay, h_decay, l11, l21, l22) in enumerate(self.transition):
                x, y = self.state[i]
                n1, n2 = self.rng.gauss(0, 1), self.rng.gauss(0, 1)
                self.state[i] = (decay*x + l11*n1,
                                 decay*y + h_decay*x + l21*n1 + l22*n2)
            self.tick += 1
        u = self.state[0][0]
        v, w = (math.sqrt(1.5)*x + (1-math.sqrt(3))/math.sqrt(2)*y
                for x, y in self.state[1:])
        azimuth = math.radians(self.config.azimuth_deg)
        sigma = self.config.magnitude_m_s
        return (sigma * (u*math.sin(azimuth) + v*math.cos(azimuth)),
                sigma * (u*math.cos(azimuth) - v*math.sin(azimuth)), sigma*w)


class WindSignal:
    """Compose mean wind and disturbances on the Gazebo simulation clock."""

    def __init__(self, gust=GustConfig(), turbulence=TurbulenceConfig()):
        self.gust, self.turbulence = gust, turbulence
        self.base = (0.0, 0.0, 0.0)
        self.started = None
        self.dryden = None

    def activate(self, sim_s):
        self.started = sim_s
        self.dryden = DrydenTurbulence(self.turbulence)

    def deactivate(self):
        self.started = None
        self.dryden = None

    def sample(self, sim_s):
        gust = turbulence = (0.0, 0.0, 0.0)
        if self.started is not None:
            elapsed = sim_s - self.started
            if elapsed < 0:
                raise ValueError("Gazebo simulation clock moved backwards")
            gust = one_minus_cosine_gust(elapsed, self.gust)
            turbulence = self.dryden.sample(elapsed)
        return dict(target=[b + g + t for b, g, t in zip(self.base, gust, turbulence)],
                    base=list(self.base), gust=list(gust), turbulence=list(turbulence),
                    disturbances_active=self.started is not None)


class MissionWind:
    def __init__(self, world, target, policy, *, env, logs_dir, process=None, cancelled=lambda: False,
                 gust=GustConfig(), turbulence=TurbulenceConfig()):
        self.world, self.target, self.policy = world, tuple(target), policy
        self.gust, self.turbulence = gust, turbulence
        self.env, self.logs_dir = env, Path(logs_dir)
        self.process, self.cancelled = process, cancelled
        self.worker = None
        self._snapshot = None
        self._lock = threading.Lock()
        self._error = None
        self._stderr = None
        self._reader = None

    def start(self):
        set_world_wind(self.world, (0, 0, 0), env=self.env,
                       log_path=self.logs_dir / "wind.json", process=self.process,
                       cancelled=self.cancelled)
        report_path = self.logs_dir / "wind.json"
        report = json.loads(report_path.read_text())
        report.update(wind_policy=asdict(self.policy), mission_wind_m_s=list(self.target),
                      gust=asdict(self.gust), turbulence=asdict(self.turbulence))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "wind_events.jsonl").write_text("")
        (self.logs_dir / "wind_samples.jsonl").write_text("")
        env = dict(self.env, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
        self._stderr = (self.logs_dir / "wind_worker.log").open("w")
        self.worker = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", self.world],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, bufsize=1,
        )
        self._send(dict(config=dict(gust=asdict(self.gust), turbulence=asdict(self.turbulence))))
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            self.check()
            if self._snapshot is not None:
                sample = self.snapshot()
                if self.matches(sample, (0, 0, 0)):
                    self.event("calm_initialized")
                    if self.policy.mode == "continuous":
                        self.set_target(self.target)
                        self.set_disturbances(True, sample["sim_s"])
                    return
            time.sleep(0.05)
        raise TimeoutError(f"Actual calm wind not confirmed: {self._error}")

    def _read(self):
        with (self.logs_dir / "wind_samples.jsonl").open("a", buffering=1) as log:
            for line in self.worker.stdout:
                try:
                    sample = json.loads(line)
                    with self._lock:
                        if "error" in sample:
                            self._error = sample["error"]
                        else:
                            sample["received_monotonic_s"] = time.monotonic()
                            self._snapshot = sample
                    log.write(json.dumps(sample) + "\n")
                except (ValueError, KeyError) as exc:
                    self._error = str(exc)

    def check(self):
        if self.cancelled():
            raise InterruptedError("Mission wind interrupted")
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError("Gazebo exited during mission wind control")
        if self.worker is not None and self.worker.poll() is not None:
            raise RuntimeError("Gazebo wind worker exited; see wind_worker.log")

    def snapshot(self):
        self.check()
        with self._lock:
            sample = dict(self._snapshot) if self._snapshot else None
        if sample is None or time.monotonic() - sample["received_monotonic_s"] > 3:
            raise RuntimeError(f"Actual wind telemetry unavailable/stale: {self._error}")
        return sample

    def _send(self, message):
        self.check()
        self.worker.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.worker.stdin.flush()

    def set_target(self, vector):
        self._send({"target": list(vector)})

    def set_disturbances(self, active, sim_s):
        if self.gust.magnitude_m_s or self.turbulence.magnitude_m_s:
            self._send(dict(disturbances_active=active, sim_s=sim_s))

    def matches(self, sample, target):
        if sample.get("disturbances_active"):
            # The constant-wind settling criterion is inappropriate for a moving
            # stochastic target. Verify command delivery; actual wind stays logged.
            return (sample["enabled"] and math.dist(sample["base"], target) < 1e-6
                    and any(math.dist(sample["acknowledged"], candidate) < 1e-6
                            for candidate in (sample["target"], sample.get("previous_target", sample["target"]))))
        return (sample["enabled"]
                and math.dist(sample["acknowledged"], target) < 1e-6
                and math.dist(sample["actual"], target) <= self.policy.wind_tolerance_m_s)

    def event(self, name, *, boot_s=None, **details):
        sample = self.snapshot()
        if name in ("mission_segment_start", "mission_segment_end", "wind_mission_failed"):
            self.set_disturbances(name == "mission_segment_start", sample["sim_s"])
        event = dict(event=name, sim_s=sample["sim_s"], boot_s=boot_s,
                     monotonic_s=time.monotonic(), actual_wind_m_s=sample["actual"], **details)
        with (self.logs_dir / "wind_events.jsonl").open("a") as log:
            log.write(json.dumps(event, allow_nan=False) + "\n")
        print(f"[WIND PHASE] {name} sim={sample['sim_s']:.2f}s", flush=True)

    def save_plan(self, items):
        plan = [dict(seq=seq, scenario_index=it.get("scenario_index"),
                     command=it["command"], checkpoint=it.get("wind_checkpoint"))
                for seq, it in enumerate(items)]
        (self.logs_dir / "mission_sequence_map.json").write_text(json.dumps(plan, indent=2) + "\n")

    def close(self):
        if self.worker is not None:
            if self.worker.stdin:
                try:
                    self.worker.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                self.worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.worker.terminate()
                try:
                    self.worker.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.worker.kill()
                    self.worker.wait()
            if self._reader:
                self._reader.join(timeout=2)
            if self.worker.stdout:
                self.worker.stdout.close()
        if self._stderr:
            self._stderr.close()


def component_id(name):
    """gz::common::hash64: 64-bit FNV-1a of the registered component name."""
    value = 0xcbf29ce484222325
    for byte in name.encode():
        value = ((value ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    return value


def values(mapping):
    return mapping.values() if hasattr(mapping, "values") else (item.value for item in mapping)


def actual_wind(step):
    wind_id = component_id("gz_sim_components.Wind")
    velocity_id = component_id("gz_sim_components.WorldLinearVelocity")
    for entity in values(step.state.entities):
        components = {c.type: c for c in values(entity.components)}
        if wind_id in components and velocity_id in components:
            raw = components[velocity_id].component
            if isinstance(raw, bytes):
                raw = raw.decode()
            vector = [float(x) for x in raw.split()]
            if len(vector) == 3 and all(math.isfinite(x) for x in vector):
                return vector
    raise RuntimeError("Wind.WorldLinearVelocity missing from world state")


def run_worker(world):
    from gz.transport13 import Node
    from gz.msgs10.empty_pb2 import Empty
    from gz.msgs10.wind_pb2 import Wind
    from gz.msgs10.serialized_map_pb2 import SerializedStepMap

    node = Node()
    publisher = node.advertise(f"/world/{world}/wind", Wind)
    signal = WindSignal()
    sim_s = 0.0
    previous_target = [0.0, 0.0, 0.0]
    pending = ""
    while True:
        if select.select([sys.stdin], [], [], 0)[0]:
            chunk = os.read(sys.stdin.fileno(), 65536)
            if not chunk:
                return
            pending += chunk.decode()
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                message = json.loads(line)
                if "config" in message:
                    signal = WindSignal(GustConfig.parse(message["config"].get("gust")),
                                        TurbulenceConfig.parse(message["config"].get("turbulence")))
                if "target" in message:
                    signal.base = tuple(message["target"])
                if "disturbances_active" in message:
                    if message["disturbances_active"]:
                        signal.activate(message["sim_s"])
                    else:
                        signal.deactivate()
        # Use the last observed simulation time, never wall time. Pauses do not
        # advance either model; skipped Dryden ticks are evaluated in order.
        command_sim_s = max(sim_s, signal.started or 0.0)
        components = signal.sample(command_sim_s)
        target = components["target"]
        command = Wind()
        command.linear_velocity.x, command.linear_velocity.y, command.linear_velocity.z = target
        command.enable_wind = True
        publisher.publish(command)
        try:
            ok, info = node.request(f"/world/{world}/wind_info", Empty(), Empty, Wind, 500)
            if not ok:
                raise RuntimeError("wind_info request failed")
            acknowledged = [info.linear_velocity.x, info.linear_velocity.y, info.linear_velocity.z]
            ok, state = node.request(f"/world/{world}/state", Empty(), Empty, SerializedStepMap, 500)
            if not ok:
                raise RuntimeError("world state request failed (SceneBroadcaster required)")
            sim_s = state.stats.sim_time.sec + state.stats.sim_time.nsec * 1e-9
            output = dict(sim_s=sim_s, command_sim_s=command_sim_s,
                          actual=actual_wind(state), previous_target=previous_target, **components,
                          acknowledged=acknowledged, enabled=info.enable_wind)
        except Exception as exc:
            output = {"error": str(exc)}
        print(json.dumps(output), flush=True)
        previous_target = target
        time.sleep(0.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Private Gazebo mission wind worker")
    parser.add_argument("--worker", metavar="WORLD", required=True,
                        help="run the JSON-lines worker for this Gazebo world")
    args = parser.parse_args(argv)
    run_worker(args.worker)


if __name__ == "__main__":
    main()
