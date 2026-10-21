"""Wind waveform, Dryden statistics, and simulation-clock regression tests."""

from dataclasses import replace
import io
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tools/launcher'), str(ROOT / 'tools/commander')]
from mission_wind_controller import (
    DrydenTurbulence, GustConfig, MissionWind, TurbulenceConfig, WindSignal,
    one_minus_cosine_gust,
)
from mission_wind_adapter import WindPolicy


class WindModelTests(unittest.TestCase):
    def test_gust_boundaries_peak_direction_and_repeat(self):
        config = GustConfig(magnitude_m_s=4, start_s=2, duration_s=6, azimuth_deg=90)
        for t in (-1, 0, 2, 8, 12):
            self.assertEqual(one_minus_cosine_gust(t, config), (0, 0, 0))
        for t, expected in ((3.5, 2), (5, 4), (6.5, 2)):
            gust = one_minus_cosine_gust(t, config)
            self.assertAlmostEqual(gust[0], expected)
            self.assertAlmostEqual(gust[1], 0)
            self.assertEqual(gust[2], 0)
        vertical = one_minus_cosine_gust(5, replace(config, elevation_deg=-90))
        self.assertAlmostEqual(vertical[2], -4)
        periodic = replace(config, period_s=10)
        self.assertEqual(one_minus_cosine_gust(10, periodic), (0, 0, 0))
        self.assertEqual(one_minus_cosine_gust(15, periodic), one_minus_cosine_gust(5, config))

    def test_disabled_models_and_superposition(self):
        signal = WindSignal()
        signal.base = (2, -1, .5)
        signal.activate(10)
        for t in (10, 12, 30):
            self.assertEqual(signal.sample(t)['target'], list(signal.base))
        signal = WindSignal(GustConfig(magnitude_m_s=3, duration_s=4),
                            TurbulenceConfig(magnitude_m_s=.5))
        signal.base = (2, -1, .5)
        self.assertEqual(signal.sample(10)['target'], list(signal.base))
        signal.activate(10)
        sample = signal.sample(12)
        for i in range(3):
            self.assertAlmostEqual(sample['target'][i], signal.base[i] + sample['gust'][i] + sample['turbulence'][i])
        self.assertAlmostEqual(sample['gust'][1], 3)
        signal.deactivate()
        self.assertEqual(signal.sample(20)['target'], list(signal.base))

    def test_seed_pause_skipped_ticks_and_reactivation(self):
        config = TurbulenceConfig(magnitude_m_s=2, seed=42)
        regular, skipped = DrydenTurbulence(config), DrydenTurbulence(config)
        for tick in range(101):
            expected = regular.sample(tick / 10)
        self.assertEqual(expected, skipped.sample(10))
        self.assertEqual(expected, skipped.sample(10))  # paused clock consumes no noise
        self.assertNotEqual(expected, DrydenTurbulence(replace(config, seed=43)).sample(10))
        scaled = DrydenTurbulence(replace(config, magnitude_m_s=4)).sample(10)
        self.assertEqual(scaled, tuple(2 * v for v in expected))
        signal = WindSignal(turbulence=config)
        signal.activate(100)
        initial = signal.sample(102)
        signal.deactivate()
        signal.activate(200)
        self.assertEqual(signal.sample(202), initial)
        with self.assertRaises(ValueError):
            signal.sample(199)
        with self.assertRaises(ValueError):
            skipped.sample(9)

    def test_dryden_variance_and_distinct_longitudinal_transverse_correlations(self):
        # The longitudinal covariance is sigma^2*exp(-V*t/L).
        # The transverse covariance is sigma^2*(1-V*t/(2L))*exp(-V*t/L).
        # This catches replacing the second-order filters with three OU signals,
        # incorrect noise scaling, and sample-period-dependent variance.
        for dt in (.1, 1.0):
            config = TurbulenceConfig(magnitude_m_s=2, length_scale_m=(1, 1, 1),
                                      reference_speed_m_s=1, sample_period_s=dt, seed=1234)
            model = DrydenTurbulence(config)
            samples = [model.sample(i * dt) for i in range(40000)]
            lag = max(1, round(1 / dt))
            for axis in range(3):
                data = [s[axis] for s in samples]
                mean = statistics.fmean(data)
                variance = statistics.pvariance(data)
                self.assertAlmostEqual(mean, 0, delta=.12)
                self.assertAlmostEqual(variance, 4, delta=.3)
                covariance = statistics.fmean((a-mean)*(b-mean) for a, b in zip(data, data[lag:]))
                expected = math.exp(-lag*dt) * (1 if axis == 1 else 1-lag*dt/2)
                self.assertAlmostEqual(covariance / 4, expected, delta=.06)

    def test_turbulence_rotation_and_time_scales(self):
        base = TurbulenceConfig(magnitude_m_s=1, seed=4)
        north = DrydenTurbulence(base).sample(2)
        east = DrydenTurbulence(replace(base, azimuth_deg=90)).sample(2)
        self.assertAlmostEqual(east[0], north[1])
        self.assertAlmostEqual(east[1], -north[0])
        self.assertEqual(east[2], north[2])
        same_ratio = replace(base, reference_speed_m_s=10, length_scale_m=(60, 60, 20))
        self.assertEqual(north, DrydenTurbulence(same_ratio).sample(2))
        self.assertNotEqual(north, DrydenTurbulence(replace(base, reference_speed_m_s=10)).sample(2))
        for length in (1e-6, 1e9):
            model = DrydenTurbulence(replace(base, length_scale_m=(length,)*3))
            self.assertTrue(all(math.isfinite(v) for v in model.sample(2)))

    def test_invalid_config_rejected(self):
        for cls, cases in (
            (GustConfig, [False, [], {'unknown': 1}, {'magnitude_m_s': -1},
                          {'duration_s': 0}, {'start_s': -1}, {'period_s': 1},
                          {'azimuth_deg': 361}, {'elevation_deg': -91}]),
            (TurbulenceConfig, [False, [], {'unknown': 1}, {'length_scale_m': [1, 2]},
                                {'length_scale_m': [1, 0, 1]}, {'reference_speed_m_s': 0},
                                {'sample_period_s': 0}, {'seed': True}, {'seed': -1}, {'seed': 1.5}]),
        ):
            for value in cases + [{'magnitude_m_s': v} for v in (True, '1', float('nan'), float('inf'))]:
                with self.subTest(config=cls.__name__, value=value), self.assertRaises(ValueError):
                    cls.parse(value)

    def test_controller_phase_commands_and_dynamic_matching(self):
        with tempfile.TemporaryDirectory() as root:
            wind = MissionWind('world', (1, 0, 0), WindPolicy(mode='mission_only'),
                               env={}, logs_dir=root, gust=GustConfig(magnitude_m_s=2))
            wind.worker = Mock()
            wind.worker.poll.return_value = None
            wind.worker.stdin = io.StringIO()
            sample = dict(sim_s=10, actual=[1, 0, 0], target=[1, 0, 0],
                          base=[1, 0, 0], acknowledged=[1, 0, 0], enabled=True)
            with patch.object(wind, 'snapshot', return_value=sample):
                wind.event('takeoff_complete')
                self.assertEqual(wind.worker.stdin.getvalue(), '')
                wind.event('mission_segment_start')
                wind.event('mission_segment_end')
            commands = [json.loads(line) for line in wind.worker.stdin.getvalue().splitlines()]
            self.assertEqual([c['disturbances_active'] for c in commands], [True, False])
            sample.update(disturbances_active=True, target=[1, 2, 0], acknowledged=[1, 2, 0])
            self.assertTrue(wind.matches(sample, wind.target))
            self.assertFalse(wind.matches(sample, (0, 0, 0)))
            sample.update(previous_target=[1, 1.5, 0], acknowledged=[1, 1.5, 0])
            self.assertTrue(wind.matches(sample, wind.target))
            sample["acknowledged"] = [9, 9, 9]
            self.assertFalse(wind.matches(sample, wind.target))
            sample['disturbances_active'] = False
            self.assertFalse(wind.matches(sample, wind.target))


if __name__ == '__main__':
    unittest.main()
