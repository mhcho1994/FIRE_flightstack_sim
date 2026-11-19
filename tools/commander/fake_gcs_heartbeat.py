#!/usr/bin/env python3
"""
Lightweight fake GCS heartbeat sender for PX4 SITL headless runs.

Purpose
-------
When QGroundControl is not running, PX4 may reject arming or mission mode
because it does not detect a connected ground control station (GCS).

This helper sends a periodic MAVLink heartbeat as MAV_TYPE_GCS so that
PX4 sees an active GCS during headless batch simulations.

Typical usage
-------------
python3 fake_gcs_heartbeat.py --connect udpout:127.0.0.1:14550

Notes
-----
- This script only emulates GCS presence.
- It does not upload missions or send commands.
- Mission control should still be handled by a separate commander script.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

from pymavlink import mavutil


def main() -> int:
    parser = argparse.ArgumentParser(description="Fake GCS heartbeat sender")
    parser.add_argument(
        "--connect",
        default="udp:127.0.0.1:14550",
        help="pymavlink connection string for sending GCS heartbeat "
             "(default: udp:127.0.0.1:14550)",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=1.0,
        help="Heartbeat rate in Hz (default: 1.0)",
    )
    parser.add_argument(
        "--source-system",
        type=int,
        default=255,
        help="MAVLink source system ID for the fake GCS (default: 255)",
    )
    parser.add_argument(
        "--source-component",
        type=int,
        default=190,
        help="MAVLink source component ID for the fake GCS "
             "(default: 190, MAV_COMP_ID_MISSIONPLANNER)",
    )
    args = parser.parse_args()

    stop = {"flag": False}

    def _sig_handler(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    period = 1.0 / max(args.rate_hz, 1e-6)

    print(
        f"[FAKE_GCS] starting: connect={args.connect} "
        f"rate_hz={args.rate_hz} "
        f"source_system={args.source_system} "
        f"source_component={args.source_component}"
    )

    m = mavutil.mavlink_connection(
        args.connect,
        source_system=args.source_system,
        source_component=args.source_component,
    )

    def _wait_heartbeat(m) -> None:
            """
            Wait until the autopilot sends a MAVLink heartbeat.

            Heartbeat confirms that:
            - MAVLink connection is alive
            - autopilot is running
            """
            msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=60)

            if msg is None:
                raise RuntimeError("Heartbeat timeout (60s)")
            
            print(f"[FAKE_GCS] Heartbeat OK (sys={msg.get_srcSystem()}, comp={msg.get_srcComponent()})")


    # Wait for autopilot heartbeat
    _wait_heartbeat(m)

    # Optional short wait so the UDP socket is ready
    time.sleep(0.5)

    # clock loop timer
    last_print = 0.0
    rx_count = 0

    while not stop["flag"]:
        try:
            # 1) emptying UDP receive queue, PX4 sends hearbeat thru 14550
            while True:
                msg = m.recv_match(blocking=False)
                if msg is None:
                    break
                rx_count += 1

                now = time.time()

            # 2) send fake GCS heartbeat
            m.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                mavutil.mavlink.MAV_STATE_ACTIVE,
            )

            if now - last_print > 1.0:
                print(f"[FAKE_GCS] heartbeat sent, drained_rx={rx_count}", flush=True)
                rx_count = 0
                last_print = now

        except Exception as e:
            print(f"[FAKE_GCS] loop failed: {e}", file=sys.stderr)

        time.sleep(period)

    print("[FAKE_GCS] stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())