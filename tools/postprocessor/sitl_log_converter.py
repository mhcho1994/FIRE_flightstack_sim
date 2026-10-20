#!/usr/bin/env python3
"""
Convert PX4 ULog and ArduPilot BIN logs into CSV files for one run directory
or for all batch runs under a data root.

Expected directory layout
-------------------------
<data_root>/
  run_000/
    px4_logs/
      raw/
        *.ulg
    ardu_logs/
      raw/
        logs/
          000001.BIN
  run_001/
    ...

Outputs
-------
PX4:
  /data/run_000/px4_logs/processed/*.csv

ArduPilot:
  /data/run_000/ardu_logs/processed/*.csv

Usage examples
--------------
Process one run:
    python tools/postprocessor/sitl_log_converter.py --run-dir /data/run_000

Process all runs:
    python tools/postprocessor/sitl_log_converter.py --data-root /data

PX4 only:
    python tools/postprocessor/sitl_log_converter.py --data-root /data --px4-only

ArduPilot only:
    python tools/postprocessor/sitl_log_converter.py --run-dir /data/run_000 --ardu-only

Skip already processed directories:
    python tools/postprocessor/sitl_log_converter.py --data-root /data --skip-existing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

if __package__:
    from .ulg2csv import convert_px4_ulog_to_csv
    from .bin2csv import convert_ardupilot_bin_to_csv
else:
    from ulg2csv import convert_px4_ulog_to_csv
    from bin2csv import convert_ardupilot_bin_to_csv


def process_run_directory(
    run_dir: Path,
    process_px4: bool = True,
    process_ardu: bool = True,
    skip_existing: bool = False,
) -> dict[str, list[Path]]:
    """
    Process one batch run directory.

    Parameters
    ----------
    run_dir : Path
        Path to a run directory such as ``/data/run_000``.
    process_px4 : bool, optional
        Whether to process PX4 logs.
    process_ardu : bool, optional
        Whether to process ArduPilot logs.
    skip_existing : bool, optional
        If True, skip conversion when processed CSV files already exist.

    Returns
    -------
    dict[str, list[Path]]
        Dictionary with keys:
        - ``"px4"``  : generated PX4 CSV files
        - ``"ardu"`` : generated ArduPilot CSV files
    """
    run_dir = Path(run_dir)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    results: dict[str, list[Path]] = {"px4": [], "ardu": []}

    if process_px4:
        px4_logs_dir = run_dir / "px4_logs"
        if px4_logs_dir.exists():
            results["px4"] = convert_px4_ulog_to_csv(
                px4_logs_dir=px4_logs_dir,
                skip_existing=skip_existing,
            )
        else:
            print(f"[WARN] Missing PX4 log directory: {px4_logs_dir}")

    if process_ardu:
        ardu_logs_dir = run_dir / "ardu_logs"
        if ardu_logs_dir.exists():
            results["ardu"] = convert_ardupilot_bin_to_csv(
                ardu_logs_dir=ardu_logs_dir,
                skip_existing=skip_existing,
            )
        else:
            print(f"[WARN] Missing ArduPilot log directory: {ardu_logs_dir}")

    return results


def _iter_run_dirs(data_root: Path) -> Iterable[Path]:
    """
    Yield batch run directories under the given data root.

    A valid run directory is assumed to match:
        run_*

    Parameters
    ----------
    data_root : Path
        Root batch directory, e.g. ``/data``.

    Yields
    ------
    Path
        Run directory paths sorted lexicographically.
    """
    yield from sorted(p for p in data_root.glob("run_*") if p.is_dir())


def process_all_runs(
    data_root: Path,
    process_px4: bool = True,
    process_ardu: bool = True,
    skip_existing: bool = False,
) -> dict[str, dict[str, list[Path]]]:
    """
    Process all run directories under a batch data root.

    Parameters
    ----------
    data_root : Path
        Root directory containing ``run_XXX`` subdirectories.
    process_px4 : bool, optional
        Whether to process PX4 logs.
    process_ardu : bool, optional
        Whether to process ArduPilot logs.
    skip_existing : bool, optional
        If True, skip conversion when processed CSV files already exist.

    Returns
    -------
    dict[str, dict[str, list[Path]]]
        Nested result dictionary of the form:

        {
            "run_000": {"px4": [...], "ardu": [...]},
            "run_001": {"px4": [...], "ardu": [...]},
            ...
        }
    """
    data_root = Path(data_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    results: dict[str, dict[str, list[Path]]] = {}

    run_dirs = list(_iter_run_dirs(data_root))
    if not run_dirs:
        print(f"[WARN] No run_* directories found in: {data_root}")
        return results

    for run_dir in run_dirs:
        print(f"[INFO] Processing run directory: {run_dir}")
        results[run_dir.name] = process_run_directory(
            run_dir=run_dir,
            process_px4=process_px4,
            process_ardu=process_ardu,
            skip_existing=skip_existing,
        )

    return results



def main() -> int:
    """
    CLI entry point.

    Returns
    -------
    int
        Process exit code.
    """

    # Configured parser for the conversion script
    parser = argparse.ArgumentParser(
        description="Convert PX4 ULog and ArduPilot BIN logs to CSV."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-dir",
        type=Path,
        help="Process a single run directory such as /data/run_000",
    )
    group.add_argument(
        "--data-root",
        type=Path,
        help="Process all run_* directories under a data root such as /data",
    )

    parser.add_argument(
        "--px4-only",
        action="store_true",
        help="Process only PX4 logs",
    )
    parser.add_argument(
        "--ardu-only",
        action="store_true",
        help="Process only ArduPilot logs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip conversion if processed CSV files already exist",
    )
    args = parser.parse_args()

    # Resolve which log families to process.
    process_px4 = not args.ardu_only
    process_ardu = not args.px4_only

    try:
        if args.run_dir is not None:
            process_run_directory(
                run_dir=args.run_dir,
                process_px4=process_px4,
                process_ardu=process_ardu,
                skip_existing=args.skip_existing,
            )
        else:
            process_all_runs(
                data_root=args.data_root,
                process_px4=process_px4,
                process_ardu=process_ardu,
                skip_existing=args.skip_existing,
            )

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
