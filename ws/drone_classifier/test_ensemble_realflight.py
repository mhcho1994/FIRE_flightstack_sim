"""
Ensemble: DWT+LightGBM + Segment LightGBM.

Combination modes (set MODE below):
  "max"   — max(p_dwt, p_seg)          best when models complement each other
  "weighted" — DWT_WEIGHT*p_dwt + (1-DWT_WEIGHT)*p_seg
  "dwtonly"  — p_dwt only
  "segonly"  — p_seg only

Usage:
    python test_ensemble_realflight.py [dwt_model.pkl] [seg_model.pkl] [seg_features.json]
    python test_ensemble_realflight.py                 # auto-selects latest of each
"""
import sys
import json
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_processor import process_rosbag_flight_data
from train_dwt_lgbm import extract_turns, extract_dwt_features
from train_segment_lgbm import build_10col, segment_flight, segments_to_rows

MODE       = "max"   # "max" | "weighted" | "dwtonly" | "segonly"
DWT_WEIGHT = 0.6    # only used when MODE="weighted"
REALFLIGHT_DIR = "/home/gayeonslee/FIRE/flightstack_sim/data/realflight"


def auto_select(pattern, label):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {label} matching '{pattern}'")
    chosen = files[-1]
    print(f"[AUTO] {label}: {chosen}")
    return chosen


def dwt_prob(clf_dwt, t_res, feat7, traj_res):
    """Return ArduPilot probability from DWT model, or None if no turns found."""
    if feat7 is None or len(feat7) < 25:
        return None
    dt = float(np.median(np.diff(t_res))) if len(t_res) >= 2 else 0.02
    turns = extract_turns(t_res, feat7, traj_res, dt=dt)
    if not turns:
        return None
    feats = np.array([extract_dwt_features(t) for t in turns], dtype=np.float32)
    probs = clf_dwt.predict_proba(feats)
    return float(probs[:, 1].mean()), len(turns), [round(float(p), 3) for p in probs[:, 1]]


def seg_prob(clf_seg, feature_cols, t_res, feat7, traj_res):
    """Return ArduPilot probability from Segment model, or None if no segments found."""
    if feat7 is None or len(feat7) < 50:
        return None
    feat10 = build_10col(traj_res, feat7)
    dt = float(np.median(np.diff(t_res))) if len(t_res) >= 2 else 0.02
    segs = segment_flight(t_res, feat10, dt=dt)
    rows = segments_to_rows(segs, label=-1)
    if not rows:
        return None
    df = pd.DataFrame(rows).drop(columns=["seg_type", "label"], errors="ignore")
    df = df.reindex(columns=feature_cols, fill_value=0.0).fillna(0.0)
    probs = clf_seg.predict_proba(df.values)
    return float(probs[:, 1].mean()), len(rows)


def main():
    if len(sys.argv) == 4:
        dwt_path, seg_path, feat_path = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        dwt_path  = auto_select("dwt_lgbm_*.pkl",        "DWT model")
        seg_path  = auto_select("segment_lgbm_*.pkl",    "Segment model")
        feat_path = auto_select("segment_features_*.json", "Segment features")

    print(f"\n{'='*70}")
    print(f"  Ensemble: DWT-LightGBM + Segment-LightGBM  (mode={MODE})")
    if MODE == "weighted":
        print(f"  DWT weight={DWT_WEIGHT:.1f}  Segment weight={1-DWT_WEIGHT:.1f}")
    print(f"{'='*70}\n")

    with open(dwt_path,  "rb") as f: clf_dwt = pickle.load(f)
    with open(seg_path,  "rb") as f: clf_seg = pickle.load(f)
    with open(feat_path)      as f: feature_cols = json.load(f)

    print(f"[OK] DWT model:     {dwt_path}")
    print(f"[OK] Segment model: {seg_path}")
    print(f"[OK] {len(feature_cols)} segment features\n")

    csv_files = sorted(glob.glob(f"{REALFLIGHT_DIR}/*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")

    all_results = []

    for csv_file in csv_files:
        fname = Path(csv_file).name
        processed = process_rosbag_flight_data(csv_file)
        if not processed:
            print(f"[FAIL] {fname}")
            continue

        for seg_data in processed:
            seg_idx = seg_data["segment_index"]
            t_res, traj_res, t_resampled, traj_resampled, feat7, _, _ = seg_data["data"]

            # ── DWT score ──────────────────────────────────────────────────────
            dwt_result = dwt_prob(clf_dwt, t_resampled, feat7, traj_resampled)
            if dwt_result is not None:
                p_dwt, n_turns, turn_probs = dwt_result
            else:
                p_dwt, n_turns, turn_probs = None, 0, []

            # ── Segment score ──────────────────────────────────────────────────
            seg_result = seg_prob(clf_seg, feature_cols, t_resampled, feat7, traj_resampled)
            if seg_result is not None:
                p_seg, n_segs = seg_result
            else:
                p_seg, n_segs = None, 0

            # ── Combine ────────────────────────────────────────────────────────
            if p_dwt is None and p_seg is None:
                print(f"  {fname} seg{seg_idx}: no features extracted, skip")
                continue
            elif p_dwt is None:
                final_prob = p_seg
                note = f"seg-only ({n_segs} segs)"
            elif p_seg is None:
                final_prob = p_dwt
                note = f"dwt-only ({n_turns} turns)"
            else:
                if MODE == "max":
                    final_prob = max(p_dwt, p_seg)
                    chosen = "dwt" if p_dwt >= p_seg else "seg"
                    note = f"max({chosen})  dwt={p_dwt*100:.1f}% seg={p_seg*100:.1f}% ({n_turns}t/{n_segs}s)"
                elif MODE == "weighted":
                    final_prob = DWT_WEIGHT * p_dwt + (1 - DWT_WEIGHT) * p_seg
                    note = f"dwt={p_dwt*100:.1f}% seg={p_seg*100:.1f}% ({n_turns}t/{n_segs}s)"
                elif MODE == "dwtonly":
                    final_prob = p_dwt
                    note = f"dwt={p_dwt*100:.1f}% ({n_turns}t)"
                else:  # segonly
                    final_prob = p_seg
                    note = f"seg={p_seg*100:.1f}% ({n_segs}s)"

            prediction = "ArduPilot" if final_prob > 0.5 else "PX4"
            print(f"  {fname} seg{seg_idx}  "
                  f"ArduPilot={final_prob*100:.1f}%  → {prediction}  [{note}]")

            all_results.append({
                "file": fname, "segment": seg_idx,
                "ardu_prob_dwt":     round(p_dwt,  3) if p_dwt  is not None else None,
                "ardu_prob_seg":     round(p_seg,  3) if p_seg  is not None else None,
                "ardu_prob_final":   round(final_prob, 3),
                "prediction":        prediction,
                "n_turns":           n_turns,
                "n_segs":            n_segs,
                "per_turn_ardu_prob": turn_probs,
            })

    print(f"\n{'='*70}")
    ardu = sum(1 for r in all_results if r["prediction"] == "ArduPilot")
    px4  = sum(1 for r in all_results if r["prediction"] == "PX4")
    print(f"Total: {len(all_results)} file-segments")
    print(f"  ArduPilot: {ardu}")
    print(f"  PX4:       {px4}")

    # per-model breakdown
    if any(r["ardu_prob_dwt"] is not None for r in all_results):
        dwt_ardu = sum(1 for r in all_results if r["ardu_prob_dwt"] is not None and r["ardu_prob_dwt"] > 0.5)
        print(f"\n  DWT-only:  ArduPilot={dwt_ardu}/{len(all_results)}")
    if any(r["ardu_prob_seg"] is not None for r in all_results):
        seg_ardu = sum(1 for r in all_results if r["ardu_prob_seg"] is not None and r["ardu_prob_seg"] > 0.5)
        print(f"  Seg-only:  ArduPilot={seg_ardu}/{len(all_results)}")

    with open("ensemble_realflight_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Saved: ensemble_realflight_results.json")


if __name__ == "__main__":
    main()
