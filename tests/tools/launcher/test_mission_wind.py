"""Checkpoint ordering, actual-wind gating, and failures without launching SITL."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tools/commander"), str(ROOT / "tools/launcher")]
from mission_wind_adapter import WindPolicy, add_wind_checkpoints, monitor_wind_mission
from mission_wind_controller import (component_id, actual_wind, MissionWind, WindSignal,
                                     GustConfig, TurbulenceConfig)
from pymavlink_ardupilot_commander import _build_items_from_scenario as ardu_items
from pymavlink_px4_commander import _build_items_from_scenario as px4_items


def scenario():
    return {"common": {"scenario": {
        "home_lla": [40, -86, 0], "takeoff_alt_m": 10,
        "command": [22, 178, 16, 16, 21],
        "speed_m_s": [None, 4, None, None, None],
        "waypoints_lla": [None, None, [40.0001, -86, 10], [40, -86, 5], None],
        "wind_policy": {"mode": "mission_only"},
    }}}


class Simulation:
    def __init__(self, *, acknowledge_only=False, release=False):
        self.t = 0
        self.policy = WindPolicy.parse(dict(mode="mission_only", ramp_s=.5,
                                           min_settle_s=.3, settle_timeout_s=5))
        self.target = (2, 1, .5)
        self.actual = [0, 0, 0]
        self.command = (0, 0, 0)
        self.command_time = 0
        self.events = []
        self.seq = 1
        self.target_system = self.target_component = 1
        self.mav = Mock()
        self.mav.mission_set_current_send.side_effect = self.advance
        self.i = 0
        self.acknowledge_only = acknowledge_only
        self.no_release = release
        self.body_started = None
        self.mission_duration_s = .5
        self.items, *_ = px4_items(scenario())
        self.land_seq = len(self.items) - 1
        self.preland_seq = self.land_seq - 1

    def advance(self, _sys, _comp, seq):
        if self.no_release:
            return
        self.seq = seq
        if seq == 2:
            self.body_started = self.t

    def recv_match(self, **kwargs):
        self.t += .05
        if self.body_started is not None and self.seq < self.preland_seq and self.t - self.body_started > self.mission_duration_s:
            self.seq = self.preland_seq
        altitude = 5 if self.seq >= self.preland_seq else 10
        kind = ("GLOBAL_POSITION_INT", "ATTITUDE", "MISSION_CURRENT")[self.i % 3]
        self.i += 1
        if kind == "GLOBAL_POSITION_INT":
            fields = dict(time_boot_ms=int((100 + self.t) * 1000), lat=400000000, lon=-860000000,
                          relative_alt=altitude * 1000, vx=0, vy=0, vz=0)
        elif kind == "ATTITUDE":
            fields = dict(roll=.25 if max(self.actual) > .2 else 0,
                          pitch=0, rollspeed=0, pitchspeed=0, yawspeed=0)
        else:
            fields = dict(seq=self.seq)
        return SimpleNamespace(get_type=lambda: kind, get_srcSystem=lambda: 1,
                               get_srcComponent=lambda: 1, **fields)

    def snapshot(self):
        if self.t - self.command_time > .2:
            self.actual = [0, 0, 0] if self.acknowledge_only else list(self.command)
        return dict(sim_s=self.t, actual=self.actual, acknowledged=list(self.command), enabled=True)

    def set_target(self, vector):
        if vector != self.command:
            self.command_time = self.t
        self.command = vector

    def matches(self, sample, target):
        import math
        return math.dist(sample["actual"], target) < .1 and list(target) == sample["acknowledged"]

    def event(self, name, **details):
        self.events.append(dict(event=name, sim_s=self.t, actual=list(self.actual), **details))

    def save_plan(self, items):
        pass

    def run(self):
        with patch("mission_wind_adapter.time.monotonic", side_effect=lambda: self.t):
            return monitor_wind_mission(Mock(), self, self.items, self, duration=60)


class DisturbedSimulation(Simulation):
    def __init__(self):
        super().__init__()
        self.mission_duration_s = 4
        self.signal = WindSignal(GustConfig(magnitude_m_s=2, duration_s=2),
                                 TurbulenceConfig(magnitude_m_s=.5, seed=12))
        self.samples = []

    def snapshot(self):
        sample = super().snapshot()
        self.signal.base = tuple(sample['actual'])
        components = self.signal.sample(self.t)
        sample.update(components, actual=components['target'], acknowledged=components['target'])
        self.samples.append(sample)
        return sample

    def event(self, name, **details):
        super().event(name, **details)
        if name == 'mission_segment_start':
            self.signal.activate(self.t)
        elif name == 'mission_segment_end':
            self.signal.deactivate()

    matches = MissionWind.matches


class MissionWindTests(unittest.TestCase):
    def test_both_builders_insert_holds_and_preserve_original_indices(self):
        for builder in (ardu_items, px4_items):
            items, *_ = builder(scenario())
            self.assertEqual([it["command"] for it in items], [22, 17, 178, 16, 16, 17, 21])
            self.assertEqual([it["scenario_index"] for it in items], [0, None, 1, 2, 3, None, 4])
            self.assertEqual(items[1]["p7"], 10)
            self.assertEqual(items[-2]["p7"], 5)
            scn = scenario()
            del scn["common"]["scenario"]["wind_policy"]
            self.assertEqual(len(builder(scn)[0]), 5)

    def test_policy_and_ambiguous_missions_are_rejected(self):
        for value in ({"mode": "typo"}, {"ramp_s": -1}, {"ramp_s": True},
                      {"min_settle_s": float("nan")}, {"settle_timeout_s": 1}, {"typo": 1}):
            with self.assertRaises(ValueError):
                WindPolicy.parse(value)
        items = px4_items(scenario())[0]
        with self.assertRaises(ValueError):
            add_wind_checkpoints(items[:-1])

    def test_wind_and_calm_must_settle_before_release_and_landing(self):
        sim = Simulation()
        self.assertTrue(sim.run())
        names = [e["event"] for e in sim.events]
        expected = ["takeoff_complete", "wind_ready", "mission_segment_start",
                    "mission_segment_end", "calm_ready", "landing_start"]
        self.assertEqual([n for n in names if n in expected], expected)
        ready = next(e for e in sim.events if e["event"] == "wind_ready")
        self.assertEqual(ready["actual"], list(sim.target))
        self.assertEqual(sim.events[-1]["actual"], [0, 0, 0])
        # A nonzero equilibrium roll in wind must not block mission start.
        self.assertGreater(ready["boot_s"], 100)

    def test_disturbances_run_only_in_mission_and_landing_still_requires_calm(self):
        sim = DisturbedSimulation()
        self.assertTrue(sim.run())
        start = next(e['sim_s'] for e in sim.events if e['event'] == 'mission_segment_start')
        end = next(e['sim_s'] for e in sim.events if e['event'] == 'mission_segment_end')
        active = [s for s in sim.samples if s['disturbances_active']]
        self.assertGreater(len(active), 20)
        self.assertTrue(all(start <= s['sim_s'] <= end for s in active))
        self.assertTrue(any(abs(s['actual'][1] - sim.target[1]) > .5 for s in active))
        self.assertEqual(sim.samples[-1]['actual'], [0, 0, 0])
        self.assertFalse(sim.samples[-1]['disturbances_active'])

    def test_target_ack_without_actual_wind_never_releases_mission(self):
        sim = Simulation(acknowledge_only=True)
        with self.assertRaisesRegex(TimeoutError, "post_takeoff"):
            sim.run()
        self.assertNotIn("mission_segment_start", [e["event"] for e in sim.events])
        sim.mav.mission_set_current_send.assert_not_called()

    def test_release_without_telemetry_confirmation_fails(self):
        sim = Simulation(release=True)
        with self.assertRaisesRegex(TimeoutError, "release"):
            sim.run()
        self.assertNotIn("landing_start", [e["event"] for e in sim.events])

    def test_world_state_identifies_wind_entity_not_vehicle_velocity(self):
        wind_id = component_id("gz_sim_components.Wind")
        velocity_id = component_id("gz_sim_components.WorldLinearVelocity")
        self.assertEqual(wind_id, 17419187034320976751)
        self.assertEqual(velocity_id, 6822503852486540549)
        def entity(components):
            return SimpleNamespace(components={i: SimpleNamespace(type=k, component=v)
                                               for i, (k, v) in enumerate(components)})
        state = SimpleNamespace(state=SimpleNamespace(entities={
            1: entity([(velocity_id, b"99 0 0")]),
            2: entity([(wind_id, b""), (velocity_id, b"2 1 0.5")]),
        }))
        self.assertEqual(actual_wind(state), [2, 1, .5])


if __name__ == "__main__":
    unittest.main()
