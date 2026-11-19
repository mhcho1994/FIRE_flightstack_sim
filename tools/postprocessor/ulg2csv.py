#!/usr/bin/env python3
"""Convert PX4 ULog files from px4_logs/raw into processed CSV files."""

from __future__ import annotations

from pathlib import Path

from pyulog.ulog2csv import convert_ulog2csv


def convert_px4_ulog_to_csv(px4_logs_dir: Path, skip_existing: bool = False) -> list[Path]:
    """
    Convert all PX4 ULog files in a run directory into CSV files.

    This function searches for ``*.ulg`` files under:
        <run_dir>/px4_logs/raw/

    and writes the converted CSV files under:
        <run_dir>/px4_logs/processed/

    Parameters
    ----------
    px4_logs_dir : Path
        Path to the PX4 log directory, e.g. ``/data/run_000/px4_logs``.
    skip_existing : bool, optional
        If True and the processed directory already contains CSV files,
        conversion is skipped.

    Returns
    -------
    list[Path]
        A sorted list of generated CSV files in the processed directory.

    Notes
    -----
    - Uses ``pyulog.ulog2csv.convert_ulog2csv(...)``.
    - All ULog files found in the directory are converted.
    - Each ULog typically expands into multiple CSV files, one per topic.
    """
    px4_logs_dir = Path(px4_logs_dir)
    logs_dir = px4_logs_dir / "raw"
    processed_dir = px4_logs_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    existing_csvs = sorted(processed_dir.glob("*.csv"))
    if skip_existing and existing_csvs:
        print(f"[SKIP] PX4 conversion already exists: {processed_dir}")
        return existing_csvs

    ulg_files = sorted(logs_dir.glob("*.ulg"))
    if not ulg_files:
        print(f"[WARN] No PX4 ULog files found in: {logs_dir}")
        return []

    for ulg_file in ulg_files:
        print(f"[INFO] Converting PX4 ULog: {ulg_file}")
        # Convert the ULog into one or more CSV files in processed_dir.
        convert_ulog2csv(
            str(ulg_file),   # ulog_file_name
            None,            # messages: convert all messages
            str(processed_dir),  # output directory
            ",",             # delimiter
            None,            # time_s
            None,            # time_e
            False,           # disable_str_exceptions
        )

    return sorted(processed_dir.glob("*.csv"))
