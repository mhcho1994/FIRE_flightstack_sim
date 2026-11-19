from __future__ import annotations

from pathlib import Path
import pandas as pd


def interpolate_trajectory(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing trajectory values using linear interpolation.
    """
    df = df_raw.copy()

    interp_cols = [
        "x_smooth",
        "y_smooth",
        "z_smooth",
        "gt_x",
        "gt_y",
        "gt_z",
    ]

    missing_cols = [c for c in interp_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing interpolation columns: {missing_cols}")

    # Ensure numeric
    for c in interp_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional: keep timestamp/frame numeric too
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    if "frame" in df.columns:
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce")

    # Linear interpolation, including beginning/end NaNs
    df[interp_cols] = df[interp_cols].interpolate(
        method="linear",
        limit_direction="both",
    )

    return df


def process_all_runs(
    data_root: str | Path,
    raw_filename: str = "trajectory.csv",
    processed_filename: str = "trajectory.csv",
) -> None:
    """
    For each run_XXX directory, read raw trajectory and save interpolated processed trajectory.

    Expected input:
        data_root/run_000/ardu_logs/raw/trajectory.csv

    Output:
        data_root/run_000/ardu_logs/processed/trajectory.csv
    """
    data_root = Path(data_root)

    run_dirs = sorted(data_root.glob("run_*"))

    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under: {data_root}")

    for run_dir in run_dirs:
        raw_path = run_dir / "ardu_logs" / "raw" / raw_filename
        processed_dir = run_dir / "ardu_logs" / "processed"
        processed_path = processed_dir / processed_filename

        if not raw_path.exists():
            print(f"[Skip] Raw file not found: {raw_path}")
            continue

        df_raw = pd.read_csv(raw_path)

        if "frame" in df_raw.columns:
            df_raw = df_raw.sort_values("frame").reset_index(drop=True)
        elif "timestamp" in df_raw.columns:
            df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

        df_processed = interpolate_trajectory(df_raw)

        processed_dir.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(processed_path, index=False)

        print(f"[Saved] {processed_path}")


if __name__ == "__main__":
    data_root = Path("./data/260424_flight_logs")
    process_all_runs(data_root)