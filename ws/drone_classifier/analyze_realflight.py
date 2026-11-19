"""
Analyze and visualize all real-flight CSV trajectories.

For each file:
  - Resolves position columns across all known naming conventions
  - Computes: duration, distance, speed stats, altitude stats, velocity stats
  - Saves a 3D trajectory PNG

Outputs:
  realflight_analysis/          -- one PNG per CSV
  realflight_summary.csv        -- stats table
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

REALFLIGHT_DIR = Path("/home/gayeonslee/FIRE/flightstack_sim/data/realflight")
OUT_DIR        = Path("realflight_analysis")
OUT_DIR.mkdir(exist_ok=True)


# ── column resolver (mirrors trajectory_processor logic) ─────────────────────

def _resolve_columns(df):
    """Return (t, x, y, z, vx, vy, vz) arrays or raise ValueError."""
    cols = set(df.columns)

    # time
    if "bag_time_ns" in cols:
        t = df["bag_time_ns"].values / 1e9
    elif "mocap_time_s" in cols:
        t = df["mocap_time_s"].values
    elif "time_s" in cols:
        t = df["time_s"].values
    elif "timestamp" in cols:
        t = df["timestamp"].values
    else:
        raise ValueError("No time column")

    # position — prefer smoothed/gt columns over raw
    if "x_smooth" in cols:
        x, y, z = df["x_smooth"].values, df["y_smooth"].values, df["z_smooth"].values
    elif "xsmooth" in cols:
        x, y, z = df["xsmooth"].values, df["ysmooth"].values, df["zsmooth"].values
    elif "gt_x" in cols:
        x, y, z = df["gt_x"].values, df["gt_y"].values, df["gt_z"].values
    elif "gtx" in cols:
        x, y, z = df["gtx"].values, df["gty"].values, df["gtz"].values
    elif "x" in cols and "y" in cols and "z" in cols:
        x, y, z = df["x"].values, df["y"].values, df["z"].values
    else:
        raise ValueError("No position columns")

    # NED→ENU if needed
    if np.nanmedian(z) < 0:
        z = -z

    # remove NaN rows
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z) | np.isnan(t))
    t, x, y, z = t[valid], x[valid], y[valid], z[valid]
    if len(t) < 10:
        raise ValueError("Too few valid rows")

    # velocity — direct if available, else numerical diff
    if "vx" in cols and "vy" in cols and "vz" in cols:
        vx = df["vx"].values[valid]
        vy = df["vy"].values[valid]
        vz = df["vz"].values[valid]
    else:
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        vz = np.gradient(z, t)

    # clip velocity outliers (numerical diff noise)
    for v in (vx, vy, vz):
        np.clip(v, -30, 30, out=v)

    return t, x, y, z, vx, vy, vz


# ── statistics ────────────────────────────────────────────────────────────────

def _compute_stats(t, x, y, z, vx, vy, vz):
    dt          = np.diff(t)
    dt          = dt[dt > 0]
    duration    = float(t[-1] - t[0])

    speed_xy    = np.sqrt(vx**2 + vy**2)
    speed_3d    = np.sqrt(vx**2 + vy**2 + vz**2)
    step_dist   = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    distance    = float(step_dist.sum())

    return {
        "n_samples":      len(t),
        "duration_s":     round(duration, 2),
        "distance_m":     round(distance, 2),
        "avg_speed_ms":   round(float(np.mean(speed_3d)), 3),
        "max_speed_ms":   round(float(np.max(speed_3d)),  3),
        "avg_speed_xy":   round(float(np.mean(speed_xy)), 3),
        "max_speed_xy":   round(float(np.max(speed_xy)),  3),
        "vz_mean":        round(float(np.mean(vz)),       3),
        "vz_std":         round(float(np.std(vz)),        3),
        "alt_min":        round(float(np.min(z)),         3),
        "alt_max":        round(float(np.max(z)),         3),
        "alt_mean":       round(float(np.mean(z)),        3),
        "alt_range":      round(float(np.max(z) - np.min(z)), 3),
        "x_range":        round(float(np.max(x) - np.min(x)), 3),
        "y_range":        round(float(np.max(y) - np.min(y)), 3),
    }


# ── 3D trajectory plot ────────────────────────────────────────────────────────

def _plot_3d(t, x, y, z, vx, vy, vz, stats, fname, out_path):
    speed_3d = np.sqrt(vx**2 + vy**2 + vz**2)
    speed_xy = np.sqrt(vx**2 + vy**2)

    # determine autopilot from filename heuristic
    fl = fname.lower()
    if "ardupilot" in fl or "ardu" in fl:
        ap_label = "ArduPilot"
        ap_color = "#e07b39"
    elif "px4" in fl:
        ap_label = "PX4"
        ap_color = "#3a7ebf"
    elif "cognipilot" in fl:
        ap_label = "CogniPilot"
        ap_color = "#2ca02c"
    else:
        ap_label = "Unknown"
        ap_color = "#888888"

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"{fname}   [{ap_label}]", fontsize=13, fontweight="bold", y=0.99)

    # ── 3D trajectory (large, left) ───────────────────────────────────────────
    ax3d = fig.add_axes([0.03, 0.18, 0.48, 0.76], projection="3d")
    sc = ax3d.scatter(x, y, z, c=speed_3d, cmap="plasma",
                      s=4, linewidths=0, vmin=0, vmax=max(speed_3d.max(), 0.1))
    ax3d.plot(x, y, z, color="gray", alpha=0.25, linewidth=0.6)
    ax3d.scatter([x[0]], [y[0]], [z[0]], color="green", s=60, zorder=5, label="Start")
    ax3d.scatter([x[-1]], [y[-1]], [z[-1]], color="red",   s=60, zorder=5, label="End")
    ax3d.set_xlabel("X (m)", labelpad=4, fontsize=9)
    ax3d.set_ylabel("Y (m)", labelpad=4, fontsize=9)
    ax3d.set_zlabel("Z / Alt (m)", labelpad=4, fontsize=9)
    ax3d.set_title("3D Trajectory (colour = speed)", fontsize=10)
    ax3d.legend(fontsize=8, loc="upper left")
    cbar = fig.colorbar(sc, ax=ax3d, pad=0.05, shrink=0.55)
    cbar.set_label("Speed 3D (m/s)", fontsize=8)

    t_rel = t - t[0]

    # ── speed vs time ─────────────────────────────────────────────────────────
    ax1 = fig.add_axes([0.57, 0.72, 0.40, 0.22])
    ax1.plot(t_rel, speed_xy,  color=ap_color, linewidth=1.0, label="XY speed")
    ax1.plot(t_rel, speed_3d,  color="gray",   linewidth=0.8, alpha=0.6, label="3D speed")
    ax1.axhline(stats["avg_speed_xy"], color=ap_color, linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("Speed (m/s)", fontsize=9)
    ax1.set_title("Speed vs Time", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── altitude vs time ──────────────────────────────────────────────────────
    ax2 = fig.add_axes([0.57, 0.44, 0.40, 0.22])
    ax2.plot(t_rel, z, color="#d62728", linewidth=1.0)
    ax2.fill_between(t_rel, z, alpha=0.15, color="#d62728")
    ax2.axhline(stats["alt_mean"], color="#d62728", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Altitude (m)", fontsize=9)
    ax2.set_title("Altitude vs Time", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── vertical velocity vs time ─────────────────────────────────────────────
    ax3 = fig.add_axes([0.57, 0.16, 0.40, 0.22])
    ax3.plot(t_rel, vz, color="#9467bd", linewidth=0.9)
    ax3.axhline(0, color="black", linewidth=0.6, alpha=0.4)
    ax3.set_xlabel("Time (s)", fontsize=9)
    ax3.set_ylabel("Vz (m/s)", fontsize=9)
    ax3.set_title("Vertical Velocity vs Time", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── stats text box ────────────────────────────────────────────────────────
    stat_lines = [
        f"Duration:   {stats['duration_s']:.1f} s",
        f"Distance:   {stats['distance_m']:.1f} m",
        f"Samples:    {stats['n_samples']}",
        f"Avg speed:  {stats['avg_speed_ms']:.2f} m/s",
        f"Max speed:  {stats['max_speed_ms']:.2f} m/s",
        f"Avg XY spd: {stats['avg_speed_xy']:.2f} m/s",
        f"Alt range:  {stats['alt_min']:.2f} – {stats['alt_max']:.2f} m",
        f"X range:    {stats['x_range']:.2f} m",
        f"Y range:    {stats['y_range']:.2f} m",
        f"Vz std:     {stats['vz_std']:.3f} m/s",
    ]
    fig.text(0.03, 0.12, "\n".join(stat_lines),
             fontsize=8.5, family="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.85))

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    csv_files = sorted(REALFLIGHT_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {REALFLIGHT_DIR}\n")

    all_stats = []
    ok, fail = 0, 0

    for csv_path in csv_files:
        fname = csv_path.name
        stem  = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
            t, x, y, z, vx, vy, vz = _resolve_columns(df)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")
            fail += 1
            continue

        stats = _compute_stats(t, x, y, z, vx, vy, vz)
        stats["file"] = fname

        png_path = OUT_DIR / f"{stem}.png"
        _plot_3d(t, x, y, z, vx, vy, vz, stats, fname, png_path)

        print(f"  [OK]  {fname}")
        print(f"        {stats['duration_s']:.1f}s  "
              f"{stats['distance_m']:.1f}m  "
              f"spd={stats['avg_speed_ms']:.2f}m/s (max={stats['max_speed_ms']:.2f})  "
              f"alt={stats['alt_min']:.2f}–{stats['alt_max']:.2f}m")
        all_stats.append(stats)
        ok += 1

    print(f"\n{'='*60}")
    print(f"  Processed {ok} files, skipped {fail}")
    print(f"  PNGs saved to: {OUT_DIR}/")

    if all_stats:
        col_order = ["file", "duration_s", "distance_m", "n_samples",
                     "avg_speed_ms", "max_speed_ms", "avg_speed_xy", "max_speed_xy",
                     "vz_mean", "vz_std",
                     "alt_min", "alt_max", "alt_mean", "alt_range",
                     "x_range", "y_range"]
        df_out = pd.DataFrame(all_stats)[col_order]
        csv_out = Path("realflight_summary.csv")
        df_out.to_csv(csv_out, index=False)
        print(f"  Summary CSV:   {csv_out}")
        print()
        print(df_out.drop(columns=["file"]).describe().round(3).to_string())


if __name__ == "__main__":
    main()
