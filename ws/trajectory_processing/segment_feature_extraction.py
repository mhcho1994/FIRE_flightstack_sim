# =====================================================================
# SEGMENT-LEVEL FEATURE EXTRACTION
# =====================================================================
def _nan_safe_stats(x, prefix):
    """
    Return a dict of common statistics for a 1D array.
    """
    x = np.asarray(x)
    if len(x) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_rms": np.nan,
        }

    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_p10": float(np.percentile(x, 10)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
        f"{prefix}_range": float(np.max(x) - np.min(x)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(x)))),
    }


def _segment_heading_features(heading):
    """
    Heading-specific summary features.
    """
    heading = np.asarray(heading)
    if len(heading) == 0:
        return {
            "heading_net_change": np.nan,
            "heading_abs_change_sum": np.nan,
            "heading_direction": 0,
        }

    d_heading = np.diff(heading)
    net_change = heading[-1] - heading[0]
    abs_change_sum = np.sum(np.abs(d_heading)) if len(d_heading) > 0 else 0.0

    direction = 0
    if net_change > 1e-3:
        direction = 1
    elif net_change < -1e-3:
        direction = -1

    return {
        "heading_net_change": float(net_change),
        "heading_abs_change_sum": float(abs_change_sum),
        "heading_direction": int(direction),
    }


def _segment_time_shape_features(t, speed_xy, yaw_rate, curvature, a_long):
    """
    Shape features describing where peaks occur within the segment.
    """
    n = len(t)
    if n < 2:
        return {
            "peak_speed_rel_idx": np.nan,
            "peak_abs_yaw_rate_rel_idx": np.nan,
            "peak_curvature_rel_idx": np.nan,
            "peak_abs_a_long_rel_idx": np.nan,
        }

    def _rel_idx(arr):
        if len(arr) == 0:
            return np.nan
        idx = int(np.argmax(arr))
        return float(idx / max(len(arr) - 1, 1))

    return {
        "peak_speed_rel_idx": _rel_idx(speed_xy),
        "peak_abs_yaw_rate_rel_idx": _rel_idx(np.abs(yaw_rate)),
        "peak_curvature_rel_idx": _rel_idx(curvature),
        "peak_abs_a_long_rel_idx": _rel_idx(np.abs(a_long)),
    }


def extract_segment_features(
    segment,
    run_id,
    autopilot,
    segment_group=None,
):
    """
    Extract summary features from a single segment dictionary.

    Parameters
    ----------
    segment : dict
        Segment returned by extract_flight_segments_v2().
    run_id : str
        Run identifier, e.g. 'run_000'.
    autopilot : str
        'px4' or 'ardupilot'.
    segment_group : str or None
        Optional parent group if needed later. Usually same as segment['label'].

    Returns
    -------
    dict
        Flat feature row for use in a pandas DataFrame.
    """
    if segment is None:
        return None

    t = np.asarray(segment["time"])
    feat = np.asarray(segment["features"])
    label = segment["label"]

    if len(t) < 2 or len(feat) < 2:
        return None

    # feature columns
    altitude = feat[:, 0]
    heading = feat[:, 1]
    v_alt = feat[:, 2]
    speed_xy = feat[:, 3]
    a_alt = feat[:, 4]
    acc_norm_xy = feat[:, 5]
    j_alt = feat[:, 6]
    jerk_norm_xy = feat[:, 7]
    curvature = feat[:, 8]
    yaw_rate = feat[:, 9]

    dt_local = float(np.median(np.diff(t))) if len(t) >= 2 else np.nan
    duration = float(t[-1] - t[0]) if len(t) >= 2 else 0.0

    # longitudinal acceleration estimated from horizontal speed derivative
    speed_xy_smooth = _safe_savgol(speed_xy, window_length=11, polyorder=2)
    a_long = np.gradient(speed_xy_smooth, dt_local) if len(speed_xy_smooth) >= 2 else np.zeros_like(speed_xy_smooth)

    # lateral accel approximation
    a_lat_approx = speed_xy * np.abs(yaw_rate)

    # path length approximation in XY
    path_length_xy = float(np.trapz(speed_xy, t))

    row = {
        "run_id": run_id,
        "autopilot": autopilot,
        "segment_label": label,
        "segment_group": segment_group if segment_group is not None else label,
        "t_start": float(t[0]),
        "t_end": float(t[-1]),
        "duration": duration,
        "n_samples": int(len(t)),
        "dt_mean": dt_local,
        "path_length_xy": path_length_xy,
    }

    # generic statistics
    row.update(_nan_safe_stats(altitude, "altitude"))
    row.update(_nan_safe_stats(v_alt, "v_alt"))
    row.update(_nan_safe_stats(speed_xy, "speed_xy"))
    row.update(_nan_safe_stats(a_alt, "a_alt"))
    row.update(_nan_safe_stats(acc_norm_xy, "acc_xy_norm"))
    row.update(_nan_safe_stats(j_alt, "j_alt"))
    row.update(_nan_safe_stats(jerk_norm_xy, "jerk_xy_norm"))
    row.update(_nan_safe_stats(curvature, "curvature"))
    row.update(_nan_safe_stats(yaw_rate, "yaw_rate"))
    row.update(_nan_safe_stats(a_long, "a_long"))
    row.update(_nan_safe_stats(a_lat_approx, "a_lat_approx"))

    # heading-related features
    row.update(_segment_heading_features(heading))

    # shape / temporal-location features
    row.update(_segment_time_shape_features(t, speed_xy, yaw_rate, curvature, a_long))

    # signed / semantic features
    row.update({
        "speed_delta": float(speed_xy[-1] - speed_xy[0]),
        "altitude_delta": float(altitude[-1] - altitude[0]),
        "mean_signed_yaw_rate": float(np.mean(yaw_rate)),
        "mean_abs_yaw_rate": float(np.mean(np.abs(yaw_rate))),
        "mean_abs_curvature": float(np.mean(np.abs(curvature))),
        "mean_abs_a_long": float(np.mean(np.abs(a_long))),
        "mean_abs_a_lat_approx": float(np.mean(np.abs(a_lat_approx))),
        "integral_abs_yaw_rate": float(np.trapz(np.abs(yaw_rate), t)),
        "integral_curvature": float(np.trapz(np.abs(curvature), t)),
        "integral_abs_a_long": float(np.trapz(np.abs(a_long), t)),
        "integral_abs_jerk_xy": float(np.trapz(np.abs(jerk_norm_xy), t)),
    })

    # ratios that can be discriminative
    speed_mean = np.mean(speed_xy)
    yaw_abs_mean = np.mean(np.abs(yaw_rate))
    curvature_mean = np.mean(np.abs(curvature))
    a_long_abs_mean = np.mean(np.abs(a_long))

    row.update({
        "yaw_rate_to_speed_ratio": float(yaw_abs_mean / (speed_mean + 1e-6)),
        "curvature_to_speed_ratio": float(curvature_mean / (speed_mean + 1e-6)),
        "a_long_to_speed_ratio": float(a_long_abs_mean / (speed_mean + 1e-6)),
    })

    # label-oriented indicators
    row.update({
        "is_takeoff": int(label == "takeoff"),
        "is_landing": int(label == "landing"),
        "is_turn": int(label == "turn"),
        "is_straight": int(label.startswith("straight")),
        "is_straight_const": int(label == "straight_const"),
        "is_straight_accel": int(label == "straight_accel"),
        "is_straight_decel": int(label == "straight_decel"),
        "is_unknown": int(label == "unknown"),
    })

    return row


# ======================================================================================

def build_segment_dataframe(segments, run_id, autopilot):
    """
    Convert a segmented flight into a segment-level pandas DataFrame.

    Parameters
    ----------
    segments : dict
        Output 'segments' from extract_flight_segments_v2().
    run_id : str
        Example: 'run_000'
    autopilot : str
        'px4' or 'ardupilot'

    Returns
    -------
    pd.DataFrame
        One row per segment.
    """
    rows = []
    if not segments:
        return pd.DataFrame()

    ordered_keys = [
        "takeoff",
        "landing",
        "turn",
        "straight_const",
        "straight_accel",
        "straight_decel",
        "unknown",
    ]

    for key in ordered_keys:
        seg_obj = segments.get(key)

        if seg_obj is None:
            continue

        if isinstance(seg_obj, dict):
            row = extract_segment_features(
                segment=seg_obj,
                run_id=run_id,
                autopilot=autopilot,
                segment_group=key,
            )
            if row is not None:
                rows.append(row)

        elif isinstance(seg_obj, list):
            for seg in seg_obj:
                row = extract_segment_features(
                    segment=seg,
                    run_id=run_id,
                    autopilot=autopilot,
                    segment_group=key,
                )
                if row is not None:
                    rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # stable sort for readability
    if "t_start" in df.columns:
        df = df.sort_values(["run_id", "autopilot", "t_start"]).reset_index(drop=True)

    return df

# ======================================================================================

def collect_all_runs_segment_dataframe(base_data_dir=Path("data"), max_runs=100):
    """
    Collect segment-level features from all PX4 and ArduPilot runs.

    Parameters
    ----------
    base_data_dir : Path
        Root directory containing run_000, run_001, ...
    max_runs : int
        Maximum run count to scan.

    Returns
    -------
    pd.DataFrame
        Combined segment-level dataset for both autopilots.
    """
    # all_dfs = []

    # for i in range(max_runs):
    #     run_folder = f"run_{i:03d}"
    #     run_dir = base_data_dir / run_folder
    #     if not run_dir.exists():
    #         continue

    #     px4_dir = run_dir / "px4_logs" / "raw"
    #     ardu_dir = run_dir / "ardu_logs" / "raw" / "logs"

    #     # PX4
    #     if px4_dir.exists():
    #         for file in os.listdir(px4_dir):
    #             if file.lower().endswith(".ulg"):
                    x_px4, y_px4, t_px4, feat_px4, segments_px4, spans_px4 = process_px4_flight_data(str(px4_dir / file))
                    if segments_px4:
                        df_px4 = build_segment_dataframe(
                            segments=segments_px4,
                            run_id=run_folder,
                            autopilot="px4",
                        )
                        if not df_px4.empty:
                            df_px4["source_file"] = file
                            all_dfs.append(df_px4)
                    break

        # ArduPilot
        if ardu_dir.exists():
            for file in os.listdir(ardu_dir):
                if file.lower().endswith(".bin"):
                    x_ardu, y_ardu, t_ardu, feat_ardu, segments_ardu, spans_ardu = process_ardu_flight_data(str(ardu_dir / file))
                    if segments_ardu:
                        df_ardu = build_segment_dataframe(
                            segments=segments_ardu,
                            run_id=run_folder,
                            autopilot="ardupilot",
                        )
                        if not df_ardu.empty:
                            df_ardu["source_file"] = file
                            all_dfs.append(df_ardu)
                    break

    if not all_dfs:
        return pd.DataFrame()

    df_all = pd.concat(all_dfs, ignore_index=True)

    # optional convenience label for classification
    df_all["autopilot_label"] = df_all["autopilot"].map({"px4": 0, "ardupilot": 1})

    return df_all

# ======================================================================================
def filter_segment_dataset(df, segment_labels=None, min_duration=None):
    """
    Filter a segment-level dataset by label and/or duration.

    Parameters
    ----------
    df : pd.DataFrame
    segment_labels : list[str] or None
        Example: ['turn'] or ['straight_accel', 'straight_decel']
    min_duration : float or None
        Minimum segment duration in seconds.

    Returns
    -------
    pd.DataFrame
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    if segment_labels is not None:
        out = out[out["segment_label"].isin(segment_labels)]

    if min_duration is not None and "duration" in out.columns:
        out = out[out["duration"] >= min_duration]

    return out.reset_index(drop=True)

# ======================================================================================

def get_ml_feature_columns(df):
    """
    Return numeric feature columns suitable for ML, excluding metadata columns.
    """
    exclude = {
        "run_id",
        "autopilot",
        "autopilot_label",
        "segment_label",
        "segment_group",
        "source_file",
        "t_start",
        "t_end",
    }

    numeric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    return numeric_cols

# ======================================================================================

    # -------------------------------------------------------------
    # Segment-level dataset export for PX4 vs ArduPilot classification
    # -------------------------------------------------------------
    df_segments = collect_all_runs_segment_dataframe(BASE_DATA_DIR, max_runs=100)

    if not df_segments.empty:
        out_csv = BASE_DATA_DIR / "segment_feature_dataset.csv"
        df_segments.to_csv(out_csv, index=False)
        print(f"[Info] Segment feature dataset saved to: {out_csv}")

        # Example filtered exports
        df_turn = filter_segment_dataset(df_segments, segment_labels=["turn"], min_duration=0.5)
        if not df_turn.empty:
            turn_csv = BASE_DATA_DIR / "segment_feature_dataset_turn_only.csv"
            df_turn.to_csv(turn_csv, index=False)
            print(f"[Info] Turn-only dataset saved to: {turn_csv}")

        df_straight_dyn = filter_segment_dataset(
            df_segments,
            segment_labels=["straight_accel", "straight_decel"],
            min_duration=0.4,
        )
        if not df_straight_dyn.empty:
            straight_dyn_csv = BASE_DATA_DIR / "segment_feature_dataset_straight_dyn.csv"
            df_straight_dyn.to_csv(straight_dyn_csv, index=False)
            print(f"[Info] Straight dynamic dataset saved to: {straight_dyn_csv}")