#!/usr/bin/env python3
"""Convert ArduPilot BIN files from ardu_logs/raw/logs into processed CSV files."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def convert_ardupilot_bin_to_csv(ardu_logs_dir: Path, skip_existing: bool = False) -> list[Path]:
    """
    Convert all ArduPilot BIN log files in a run directory into CSV files.

    This function searches for ``*.BIN`` files under:
        <run_dir>/ardu_logs/raw/logs/

    and writes the converted CSV files under:
        <run_dir>/ardu_logs/processed/

    Parameters
    ----------
    ardu_logs_dir : Path
        Path to the ArduPilot log directory, e.g. ``/data/run_000/ardu_logs``.
    skip_existing : bool, optional
        If True and the processed directory already contains CSV files,
        conversion is skipped.

    Returns
    -------
    list[Path]
        A sorted list of generated CSV files in the processed directory.

    Raises
    ------
    RuntimeError
        If the ``bin2csv`` executable is not available or conversion fails.

    Notes
    -----
    - Public pybinlog documentation exposes ``bin2csv`` as a command-line tool,
      so this function calls that installed entry point directly.
    - All ``*.BIN`` files in ``ardu_logs/raw/logs`` are converted.
    """
    ardu_logs_dir = Path(ardu_logs_dir)
    logs_dir = ardu_logs_dir / "raw" / "logs"
    processed_dir = ardu_logs_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    existing_csvs = sorted(processed_dir.glob("*.csv"))
    if skip_existing and existing_csvs:
        print(f"[SKIP] ArduPilot conversion already exists: {processed_dir}")
        return existing_csvs

    if shutil.which("bin2csv") is None:
        raise RuntimeError(
            "Could not find 'bin2csv' in PATH. "
            "Please install pybinlog so that the CLI entry point is available."
        )

    bin_files = sorted(logs_dir.glob("*.BIN"))
    if not bin_files:
        print(f"[WARN] No ArduPilot BIN files found in: {logs_dir}")
        return []

    for bin_file in bin_files:
        print(f"[INFO] Converting ArduPilot BIN: {bin_file}")

        # pybinlog documents bin2csv as:
        #   bin2csv [-h] [-m Messages] [-o output_directory] [filepath]
        cmd = [
            "bin2csv",
            "-o",
            str(processed_dir),
            str(bin_file),
        ]

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"bin2csv failed for {bin_file}\n"
                f"stdout:\n{result.stdout}\n\n"
                f"stderr:\n{result.stderr}"
            )

    return sorted(processed_dir.glob("*.csv"))
