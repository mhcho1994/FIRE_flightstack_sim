"""Mission cropping must retain the common boot clock across different signals."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/plotting"))
from trajectory_loader import (
    AutopilotResponse, TimeSeries, crop_to_mission_segment, read_generic_csv,
)


class MissionSegmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = self.root / "wind_events.jsonl"
        self.events.write_text("\n".join(json.dumps(e) for e in [
            dict(event="mission_segment_start", boot_s=12),
            dict(event="mission_segment_end", boot_s=15),
        ]))

    def test_signals_with_different_first_samples_use_one_origin(self):
        def series(t):
            return TimeSeries.from_boot_time(t, np.zeros((len(t), 3)), ("x", "y", "z"))
        response = AutopilotResponse("px4", Path("flight.ulg"),
                                     position=series(np.arange(10, 17)),
                                     attitude=series(np.arange(11.5, 17)))
        cropped = crop_to_mission_segment(response, self.events)
        np.testing.assert_array_equal(cropped.position.t, [0, 1, 2])
        np.testing.assert_array_equal(cropped.attitude.t, [.5, 1.5, 2.5])
        np.testing.assert_array_equal(cropped.attitude.boot_t, [12.5, 13.5, 14.5])

    def test_microsecond_csv_supports_crop_but_relative_time_csv_is_rejected(self):
        csv = self.root / "response.csv"
        csv.write_text("TimeUS,PN,PE,PD\n12000000,0,0,-10\n13000000,1,0,-10\n15000000,2,0,-5\n")
        cropped = crop_to_mission_segment(read_generic_csv(csv, "ardu"), self.events)
        np.testing.assert_array_equal(cropped.position.t, [0, 1])
        csv.write_text("time_s,x,y,z\n0,0,0,-10\n1,1,0,-10\n2,2,0,-5\n")
        with self.assertRaisesRegex(ValueError, "boot timestamps"):
            crop_to_mission_segment(read_generic_csv(csv, "ardu"), self.events)

    def test_incomplete_or_duplicate_boundaries_are_rejected(self):
        response = AutopilotResponse("ardu", Path("flight.BIN"))
        for events in ([dict(event="mission_segment_start", boot_s=12)],
                       [dict(event="mission_segment_start", boot_s=12)] * 2):
            self.events.write_text("\n".join(json.dumps(e) for e in events))
            with self.assertRaises(ValueError):
                crop_to_mission_segment(response, self.events)


if __name__ == "__main__":
    unittest.main()
