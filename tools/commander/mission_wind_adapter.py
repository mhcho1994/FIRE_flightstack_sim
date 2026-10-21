"""Wind policy, mission checkpoints, and single-reader MAVLink monitoring."""

from dataclasses import dataclass, fields
import math
import time


@dataclass(frozen=True)
class WindPolicy:
    mode: str = "continuous"
    ramp_s: float = 3.0
    min_settle_s: float = 5.0
    settle_timeout_s: float = 30.0
    position_tolerance_m: float = 1.5
    altitude_tolerance_m: float = 0.5
    speed_tolerance_m_s: float = 0.3
    wind_tolerance_m_s: float = 0.1
    calm_tilt_deg: float = 5.0

    @classmethod
    def parse(cls, value=None):
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("wind_policy must be a mapping")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown wind_policy settings: {sorted(unknown)}")
        policy = cls(**value)
        if policy.mode not in ("continuous", "mission_only"):
            raise ValueError("wind_policy.mode must be continuous or mission_only")
        for f in fields(cls):
            if f.name == "mode":
                continue
            number = getattr(policy, f.name)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError(f"wind_policy.{f.name} must be finite")
            if number < 0 or (number == 0 and f.name != "ramp_s"):
                raise ValueError(f"wind_policy.{f.name} must be positive (ramp_s may be zero)")
        if policy.settle_timeout_s <= policy.ramp_s + 2 * policy.min_settle_s:
            raise ValueError("settle_timeout_s must exceed ramp_s + 2 * min_settle_s (calm and windy takeoff holds)")
        return policy


def add_wind_checkpoints(items):
    """Keep original commands, inserting indefinite holds after takeoff/before LAND.

    Restrict this policy to the single-takeoff, single-landing missions for which
    the phase boundaries are unambiguous. Returned indices are uploaded indices.
    """
    commands = [it["command"] for it in items]
    if (commands.count(22) != 1 or commands[0] != 22
            or commands.count(21) != 1 or commands[-1] != 21
            or 16 not in commands):
        raise ValueError("mission_only requires TAKEOFF first, waypoints, and one final LAND")
    if any(it["p7"] <= 0 for it in items if it["command"] in (16, 22)):
        raise ValueError("mission_only requires positive airborne waypoint altitudes")
    expanded, gates = [], {}
    last_position = None
    for index, item in enumerate(items):
        if item["command"] == 21:
            gates[len(expanded)] = "pre_land"
            expanded.append(_hold(last_position, "pre_land"))
        expanded.append(dict(item, scenario_index=item.get("scenario_index", index)))
        if item["command"] in (16, 22):
            last_position = item
        if item["command"] == 22:
            gates[len(expanded)] = "post_takeoff"
            expanded.append(_hold(item, "post_takeoff"))
    return expanded, gates


def _hold(position, phase):
    return dict(frame=position["frame"], command=17, autocontinue=1,
                p1=0, p2=0, p3=0, p4=float("nan"),
                p5=position["p5"], p6=position["p6"], p7=position["p7"],
                scenario_index=None, wind_checkpoint=phase)


def distance_m(lat1, lon1, lat2, lon2):
    return math.hypot(math.radians(lat2 - lat1) * 6371000,
                      math.radians(lon2 - lon1) * 6371000 * math.cos(math.radians(lat1)))


def monitor_wind_mission(runner, m, items, wind, duration=300.0):
    # Scenario generation uses the policy without requiring pymavlink.
    from pymavlink import mavutil

    policy = wind.policy
    gates = {seq: it["wind_checkpoint"] for seq, it in enumerate(items) if "wind_checkpoint" in it}
    wind.save_plan(items)
    # One consumer owns this connection. Transport polling runs in the wind worker.
    for message_id, hz in ((33, 10), (30, 10), (245, 2), (42, 2)):
        m.mav.command_long_send(m.target_system, m.target_component,
                               mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                               message_id, 1e6 / hz, 0, 0, 0, 0, 0)
    position = attitude = None
    position_received = attitude_received = 0
    seq = None
    active_gate = None
    completed = set()
    phase = "takeoff"
    stable_since = None
    ramp_start = None
    gate_sim = gate_wall = 0
    pending = None
    started = time.monotonic()
    last_sim = None
    sim_progress = started
    last_sent = None
    wind_mismatch_since = None

    def event(name, **details):
        wind.event(name, boot_s=position.time_boot_ms / 1000 if position else None,
                   telemetry_age_s=time.monotonic() - position_received if position else None,
                   mission_seq=seq, **details)

    try:
        while time.monotonic() - started < duration:
            runner._check_stop()
            sample = wind.snapshot()
            sim = sample["sim_s"]
            now = time.monotonic()
            if last_sim != sim:
                last_sim, sim_progress = sim, now
            elif now - sim_progress > 10:
                raise TimeoutError("Gazebo simulation clock stopped during wind mission")

            msg = m.recv_match(blocking=True, timeout=0.1)
            if msg is not None and msg.get_srcSystem() == m.target_system and msg.get_srcComponent() == m.target_component:
                kind = msg.get_type()
                if kind == "GLOBAL_POSITION_INT":
                    position, position_received = msg, time.monotonic()
                elif kind == "ATTITUDE":
                    attitude, attitude_received = msg, time.monotonic()
                elif kind == "STATUSTEXT":
                    print(f"[MISSION] {msg.text}", flush=True)
                elif kind == "MISSION_CURRENT":
                    seq = msg.seq
                    if pending is not None and pending["next"] <= seq <= pending["limit"]:
                        completed.add(pending["gate"])
                        event("checkpoint_released", checkpoint=gates[pending["gate"]])
                        pending = None
                        active_gate = None
                        phase = "mission"

            if pending is not None:
                if now - pending["started"] > 10:
                    raise TimeoutError("Mission checkpoint release was not confirmed")
                if now >= pending["retry"]:
                    m.mav.mission_set_current_send(m.target_system, m.target_component, pending["next"])
                    pending["retry"] = now + 1
                continue

            outstanding = [g for g in gates if g not in completed]
            if seq is not None and outstanding and seq > min(outstanding):
                raise RuntimeError("Mission advanced past an unreleased wind checkpoint")
            if phase == "mission" and len(completed) == 1 and seq not in gates:
                if not wind.matches(sample, wind.target):
                    if wind_mismatch_since is None:
                        wind_mismatch_since = sim
                    elif sim - wind_mismatch_since > 2:
                        raise RuntimeError("Actual wind left the requested mission condition")
                else:
                    wind_mismatch_since = None

            if seq == len(items) - 1:
                if set(gates) != completed:
                    raise RuntimeError("LAND started before wind checkpoints completed")
                event("landing_start")
                return True
            if seq not in gates or seq in completed:
                continue
            if active_gate != seq:
                active_gate = seq
                gate_sim, gate_wall = sim, now
                stable_since = ramp_start = None
                last_sent = None
                phase = "arrival" if gates[seq] == "post_takeoff" else "ramp"
                event("takeoff_hold" if gates[seq] == "post_takeoff" else "mission_segment_end")
                if phase == "ramp":
                    ramp_start = sim
                    event("wind_ramp_down")

            if sim - gate_sim > policy.settle_timeout_s or now - gate_wall > max(60, 4 * policy.settle_timeout_s):
                raise TimeoutError(f"{gates[seq]} did not settle within {policy.settle_timeout_s:g} simulated seconds")

            post_takeoff = gates[seq] == "post_takeoff"
            target = wind.target if post_takeoff else (0.0, 0.0, 0.0)
            if phase == "ramp":
                fraction = 1 if policy.ramp_s == 0 else min(1, (sim - ramp_start) / policy.ramp_s)
                vector = tuple(fraction * v if post_takeoff else (1 - fraction) * v for v in wind.target)
                if vector != last_sent:
                    wind.set_target(vector)
                    last_sent = vector
                if fraction >= 1:
                    phase = "settle"
                    stable_since = None
                continue

            fresh = (position is not None and attitude is not None
                     and now - position_received < 1 and now - attitude_received < 1)
            stable = False
            if fresh:
                item = items[seq]
                at_position = distance_m(position.lat * 1e-7, position.lon * 1e-7,
                                         item["p5"], item["p6"]) <= policy.position_tolerance_m
                at_altitude = abs(position.relative_alt * 0.001 - item["p7"]) <= policy.altitude_tolerance_m
                slow = math.sqrt(position.vx**2 + position.vy**2 + position.vz**2) * 0.01 <= policy.speed_tolerance_m_s
                not_rotating = max(abs(attitude.rollspeed), abs(attitude.pitchspeed), abs(attitude.yawspeed)) < 0.2
                calm = phase == "arrival" or not post_takeoff
                level = (not calm or math.hypot(attitude.roll, attitude.pitch) <= math.radians(policy.calm_tilt_deg))
                expected = (0, 0, 0) if phase == "arrival" else target
                stable = at_position and at_altitude and slow and not_rotating and level and wind.matches(sample, expected)
            if not stable:
                stable_since = None
                continue
            if stable_since is None:
                stable_since = sim
            if sim - stable_since < policy.min_settle_s:
                continue
            if phase == "arrival":
                event("takeoff_complete")
                event("wind_ramp_up")
                phase, ramp_start, stable_since = "ramp", sim, None
                continue

            event("wind_ready" if post_takeoff else "calm_ready")
            if post_takeoff:
                event("mission_segment_start")
            following_gate = min((g for g in gates if g > seq), default=len(items) - 1)
            pending = dict(gate=seq, next=seq + 1, limit=following_gate, started=now, retry=now)
        raise TimeoutError("Mission monitor timeout")
    except Exception as exc:
        try:
            event("wind_mission_failed", error=str(exc), phase=phase)
        except Exception:
            pass
        raise
