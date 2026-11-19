from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def sort_run_key(run_name: str):
    """
    Sort key for run labels like 'run1', 'run2', ...
    """
    m = re.search(r"(\d+)$", str(run_name))
    return int(m.group(1)) if m else str(run_name)


def build_output_dataframe(df_run: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize columns and rename them to the requested schema.
    """
    out = df_run.loc[
        :,
        ["frame", "time_s", "x_smooth", "y_smooth", "z_smooth", "gt_x", "gt_y", "gt_z"],
    ].copy()

    out = out.rename(
        columns={
            "time_s": "timestamp",
            "x_smooth": "xsmooth",
            "y_smooth": "ysmooth",
            "z_smooth": "zsmooth",
            "gt_x": "gtx",
            "gt_y": "gty",
            "gt_z": "gtz",
        }
    )

    return out


def interpolate_processed(df_raw_out: pd.DataFrame) -> pd.DataFrame:
    """
    Linearly interpolate missing values for processed output.
    Interpolate smoothed trajectory and GT states.
    """
    df_processed = df_raw_out.copy()

    interp_cols = [
        "xsmooth",
        "ysmooth",
        "zsmooth",
        "gtx",
        "gty",
        "gtz",
    ]

    # ensure numeric
    for c in interp_cols:
        df_processed[c] = pd.to_numeric(df_processed[c], errors="coerce")

    # linear interpolation
    df_processed[interp_cols] = (
        df_processed[interp_cols]
        .interpolate(
            method="linear",
            limit_direction="both"
        )
    )

    return df_processed


def split_all_runs(
    input_csv: str | Path,
    output_root: str | Path,
    raw_filename: str = "trajectory.csv",
    processed_filename: str = "trajectory.csv",
) -> None:
    """
    Split a combined all-runs CSV into per-run raw/processed CSV files.

    Parameters
    ----------
    input_csv : str | Path
        Path to combined CSV containing columns:
        run, frame, time_s, x_smooth, y_smooth, z_smooth, gt_x, gt_y, gt_z
    output_root : str | Path
        Root directory under which run_000, run_001, ... will be created.
    raw_filename : str
        Output filename for raw CSV.
    processed_filename : str
        Output filename for processed CSV.
    """
    input_csv = Path(input_csv)
    output_root = Path(output_root)

    df = pd.read_csv(input_csv)

    required_cols = {
        "run",
        "frame",
        "time_s",
        "x_smooth",
        "y_smooth",
        "z_smooth",
        "gt_x",
        "gt_y",
        "gt_z",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    run_names = sorted(df["run"].dropna().unique(), key=sort_run_key)

    for idx, run_name in enumerate(run_names):
        df_run = df[df["run"] == run_name].copy()
        df_run = df_run.sort_values("frame").reset_index(drop=True)

        run_dir = output_root / f"run_{idx:03d}" / "ardu_logs"
        raw_dir = run_dir / "raw"
        processed_dir = run_dir / "processed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        df_raw_out = build_output_dataframe(df_run)
        df_processed_out = interpolate_processed(df_raw_out)

        raw_path = raw_dir / raw_filename
        processed_path = processed_dir / processed_filename

        df_raw_out.to_csv(raw_path, index=False)
        df_processed_out.to_csv(processed_path, index=False)

        print(f"[Saved] {run_name} -> {raw_path}")
        print(f"[Saved] {run_name} -> {processed_path}")


if __name__ == "__main__":
    # Example usage:
    # input_csv = Path("all_runs/ardu_logs/raw/ardupilot_all_runs_trajectory.csv")
    # output_root = Path(".")
    #
    # ./run_000/ardu_logs/raw/trajectory.csv
    # ./run_000/ardu_logs/processed/trajectory.csv
    # ...

    input_csv = Path("./data/260424_flight_logs/all_runs/ardupilot_all_runs_trajectory.csv")
    output_root = Path("./data/260424_flight_logs")

    split_all_runs(input_csv=input_csv, output_root=output_root)