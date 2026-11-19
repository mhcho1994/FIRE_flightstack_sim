"""
Real flight classification using segment-based LightGBM.

Usage:
    python test_segment_realflight.py [model.pkl] [features.json]
"""
import sys
import json
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_processor import process_rosbag_flight_data
from train_segment_lgbm import build_10col, segment_flight, segments_to_rows


def auto_select(pattern, label):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {label} matching '{pattern}'")
    chosen = files[-1]
    print(f"[AUTO] {label}: {chosen}")
    return chosen


def main():
    if len(sys.argv) == 3:
        model_path = sys.argv[1]
        feat_path  = sys.argv[2]
    else:
        model_path = auto_select("segment_lgbm_*.pkl",     "model")
        feat_path  = auto_select("segment_features_*.json", "feature columns")

    print(f"\n{'='*80}")
    print("  Segment LightGBM — Real Flight Classification")
    print(f"{'='*80}\n")

    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    with open(feat_path) as f:
        feature_cols = json.load(f)

    print(f"[OK] Model: {model_path}")
    print(f"[OK] {len(feature_cols)} features\n")

    csv_files = sorted(glob.glob(
        "/home/gayeonslee/FIRE/flightstack_sim/data/realflight/*.csv"
    ))
    print(f"Found {len(csv_files)} CSV files\n")

    all_results = []

    for csv_file in csv_files:
        fname = Path(csv_file).name
        processed = process_rosbag_flight_data(csv_file)
        if not processed:
            print(f"[FAIL] {fname}")
            continue

        file_rows = []
        for seg_data in processed:
            seg_idx = seg_data["segment_index"]
            t_res, traj_res, t_resampled, traj_resampled, feat7, _, _ = seg_data["data"]

            if feat7 is None or len(feat7) < 50:
                continue

            feat10 = build_10col(traj_resampled, feat7)
            dt = float(np.median(np.diff(t_resampled))) if len(t_resampled) >= 2 else 0.02
            segs = segment_flight(t_resampled, feat10, dt=dt)

            rows = segments_to_rows(segs, label=-1)  # unknown label
            if not rows:
                print(f"  {fname} seg{seg_idx}: no segments found")
                continue

            df = pd.DataFrame(rows).drop(columns=["seg_type", "label"], errors="ignore")
            df = df.reindex(columns=feature_cols, fill_value=0.0).fillna(0.0)

            probs      = clf.predict_proba(df.values)   # (N_segs, 2)
            ardu_prob  = float(probs[:, 1].mean())
            prediction = "ArduPilot" if ardu_prob > 0.5 else "PX4"

            # segment-level breakdown
            seg_labels = [r.get("seg_type", "?") for r in rows]
            seg_preds  = ["ArduPilot" if p > 0.5 else "PX4"
                          for p in probs[:, 1].tolist()]
            breakdown  = ", ".join(f"{l}→{p}" for l, p in zip(seg_labels, seg_preds))

            print(f"  {fname} seg{seg_idx}  "
                  f"ArduPilot={ardu_prob*100:.1f}%  → {prediction}  "
                  f"({len(rows)} segs: {breakdown})")

            all_results.append({
                "file":       fname,
                "segment":    seg_idx,
                "n_segs":     len(rows),
                "ardu_prob":  ardu_prob,
                "prediction": prediction,
            })

    print(f"\n{'='*80}")
    ardu_cnt = sum(1 for r in all_results if r["prediction"] == "ArduPilot")
    px4_cnt  = sum(1 for r in all_results if r["prediction"] == "PX4")
    print(f"Total: {len(all_results)} file-segments")
    print(f"  ArduPilot: {ardu_cnt}")
    print(f"  PX4:       {px4_cnt}")

    out = "segment_realflight_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Saved: {out}")


if __name__ == "__main__":
    main()
