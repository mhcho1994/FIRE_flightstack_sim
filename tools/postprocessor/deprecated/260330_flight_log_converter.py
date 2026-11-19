#!/usr/bin/env python3
"""
Convert flight log artifacts into processed CSV files.

Supported conversions
---------------------
- PX4 ULog (*.ulg) -> CSV
- ArduPilot BIN (*.BIN) -> CSV
- ROS 2 rosbag2 (MCAP storage) odometry topics -> CSV

Expected ArduPilot ROS bag layout
---------------------------------
/data/flight_logs/ardu_logs/
  raw/
    <bag_dir>/
      metadata.yaml
      *.mcap
  processed/

This module extracts mocap/odometry position from ROS 2 bag topics such as:
- /camera/odom
- /ardupilot/odom

and writes them into CSV files under:
- /data/flight_logs/ardu_logs/processed/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String


def _find_rosbag2_dirs(raw_dir: Path) -> list[Path]:
    """
    Find rosbag2 bag directories under a raw log directory.

    A rosbag2 bag directory is assumed to contain:
    - metadata.yaml
    - one or more *.mcap files

    Parameters
    ----------
    raw_dir : Path
        Directory such as /data/flight_logs/ardu_logs/raw

    Returns
    -------
    list[Path]
        Sorted list of bag directories.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []

    bag_dirs: list[Path] = []
    for metadata_file in raw_dir.rglob("metadata.yaml"):
        bag_dir = metadata_file.parent
        if any(bag_dir.glob("*.mcap")):
            bag_dirs.append(bag_dir)

    return sorted(set(bag_dirs))


def _topic_name_to_filename(topic_name: str) -> str:
    """
    Convert a ROS topic name into a safe CSV filename suffix.

    Example
    -------
    /camera/odom -> camera_odom
    """
    return topic_name.strip("/").replace("/", "_")


def _read_rosbag2_topic_rows(
    bag_dir: Path,
    topic_name: str,
) -> list[dict[str, float]]:
    """
    Read one Odometry topic from a rosbag2 bag directory.

    Parameters
    ----------
    bag_dir : Path
        Path to rosbag2 bag directory containing metadata.yaml.
    topic_name : str
        Topic name to extract, e.g. /camera/odom.

    Returns
    -------
    list[dict[str, float]]
        Extracted rows. Each row contains:
        - bag_time_ns
        - header_stamp_ns
        - x, y, z
        - qx, qy, qz, qw
        - vx, vy, vz
        - wx, wy, wz
    """
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir),
        storage_id="mcap",
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    if topic_name not in type_map:
        return []

    msg_type = get_message(type_map[topic_name])

    rows: list[dict[str, float]] = []

    while reader.has_next():
        current_topic, data, bag_time_ns = reader.read_next()
        if current_topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)

        header_stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )

        rows.append(
            {
                "bag_time_ns": int(bag_time_ns),
                "header_stamp_ns": header_stamp_ns,
                "x": float(msg.pose.pose.position.x),
                "y": float(msg.pose.pose.position.y),
                "z": float(msg.pose.pose.position.z),
                "qx": float(msg.pose.pose.orientation.x),
                "qy": float(msg.pose.pose.orientation.y),
                "qz": float(msg.pose.pose.orientation.z),
                "qw": float(msg.pose.pose.orientation.w),
                "vx": float(msg.twist.twist.linear.x),
                "vy": float(msg.twist.twist.linear.y),
                "vz": float(msg.twist.twist.linear.z),
                "wx": float(msg.twist.twist.angular.x),
                "wy": float(msg.twist.twist.angular.y),
                "wz": float(msg.twist.twist.angular.z),
            }
        )

    return rows


def _write_rows_to_csv(csv_path: Path, rows: list[dict[str, float]]) -> None:
    """
    Write extracted rows to CSV.

    Parameters
    ----------
    csv_path : Path
        Output CSV file path.
    rows : list[dict[str, float]]
        Rows to write.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Create an empty file with header for consistency.
        fieldnames = [
            "bag_time_ns",
            "header_stamp_ns",
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
        ]
    else:
        fieldnames = list(rows[0].keys())

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_rosbag2_mocap_to_csv(
    ardu_logs_dir: Path,
    topic_names: Iterable[str] = ("/camera/odom", "/ardupilot/odom"),
    skip_existing: bool = False,
) -> list[Path]:
    """
    Extract mocap/odometry topics from rosbag2 MCAP bags into CSV files.

    Directory convention
    --------------------
    Input:
        <ardu_logs_dir>/raw/<bag_dir>/metadata.yaml
        <ardu_logs_dir>/raw/<bag_dir>/*.mcap

    Output:
        <ardu_logs_dir>/processed/<bag_dir>__<topic>.csv

    Parameters
    ----------
    ardu_logs_dir : Path
        Path such as /data/flight_logs/ardu_logs
    topic_names : Iterable[str], optional
        Topic names to extract.
    skip_existing : bool, optional
        Skip writing if all expected CSV files already exist.

    Returns
    -------
    list[Path]
        Paths of generated CSV files.
    """
    ardu_logs_dir = Path(ardu_logs_dir)
    raw_dir = ardu_logs_dir / "raw"
    processed_dir = ardu_logs_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    bag_dirs = _find_rosbag2_dirs(raw_dir)
    if not bag_dirs:
        print(f"[WARN] No rosbag2 MCAP directories found in: {raw_dir}")
        return []

    generated: list[Path] = []

    for bag_dir in bag_dirs:
        bag_name = bag_dir.name
        print(f"[INFO] Processing rosbag2 bag: {bag_dir}")

        expected_paths = [
            processed_dir / f"{bag_name}__{_topic_name_to_filename(topic)}.csv"
            for topic in topic_names
        ]

        if skip_existing and all(path.exists() for path in expected_paths):
            print(f"[SKIP] Existing mocap CSVs found for: {bag_name}")
            generated.extend(expected_paths)
            continue

        for topic_name in topic_names:
            rows = _read_rosbag2_topic_rows(bag_dir=bag_dir, topic_name=topic_name)
            out_csv = processed_dir / f"{bag_name}__{_topic_name_to_filename(topic_name)}.csv"

            if not rows:
                print(f"[WARN] Topic not found or empty: {topic_name} in {bag_dir}")
                continue

            _write_rows_to_csv(out_csv, rows)
            print(f"[INFO] Wrote {len(rows)} rows -> {out_csv}")
            generated.append(out_csv)

    return generated


def main() -> int:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Extract mocap odometry from rosbag2 MCAP bags into CSV."
    )
    parser.add_argument(
        "--ardu-logs-dir",
        type=Path,
        required=True,
        help="Path like /data/flight_logs/ardu_logs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip bag conversion if expected CSV files already exist",
    )
    args = parser.parse_args()

    try:
        extract_rosbag2_mocap_to_csv(
            ardu_logs_dir=args.ardu_logs_dir,
            skip_existing=args.skip_existing,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())