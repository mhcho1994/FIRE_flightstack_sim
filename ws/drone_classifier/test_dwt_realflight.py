"""
Real flight classification using DWT + LightGBM.
Usage: python test_dwt_realflight.py [model.pkl]
"""
import sys, glob, pickle, json
from pathlib import Path
import numpy as np

from trajectory_processor import process_rosbag_flight_data
from train_dwt_lgbm import extract_turns, extract_dwt_features

def auto_select(pattern, label):
    files = sorted(glob.glob(pattern))
    if not files: raise FileNotFoundError(f"No {label} matching '{pattern}'")
    chosen = files[-1]
    print(f"[AUTO] {label}: {chosen}")
    return chosen

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else auto_select("dwt_lgbm_*.pkl", "model")

    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    print(f"\n[OK] Model: {model_path}\n")

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

        for seg_data in processed:
            seg_idx = seg_data["segment_index"]
            _, _, t_res, traj_res, feat7, _, _ = seg_data["data"]

            if feat7 is None or len(feat7) < 25:
                continue

            dt = float(np.median(np.diff(t_res))) if len(t_res) >= 2 else 0.02
            turns = extract_turns(t_res, feat7, traj_res, dt=dt)

            if not turns:
                print(f"  {fname} seg{seg_idx}: no turn segments found")
                continue

            feats = np.array([extract_dwt_features(t) for t in turns], dtype=np.float32)
            probs = clf.predict_proba(feats)          # (N_turns, 2)
            ardu_prob = float(probs[:, 1].mean())
            prediction = "ArduPilot" if ardu_prob > 0.5 else "PX4"

            # per-turn breakdown
            per_turn = ["ArduPilot" if p > 0.5 else "PX4" for p in probs[:, 1]]
            breakdown = ", ".join(f"{p:.2f}" for p in probs[:, 1])

            print(f"  {fname} seg{seg_idx}  "
                  f"ArduPilot={ardu_prob*100:.1f}%  → {prediction}  "
                  f"({len(turns)} turns: [{breakdown}])")

            all_results.append({
                "file": fname, "segment": seg_idx,
                "n_turns": len(turns),
                "ardu_prob": ardu_prob,
                "prediction": prediction,
                "per_turn_ardu_prob": [round(float(p), 3) for p in probs[:, 1]],
            })

    print(f"\n{'='*70}")
    ardu = sum(1 for r in all_results if r["prediction"] == "ArduPilot")
    px4  = sum(1 for r in all_results if r["prediction"] == "PX4")
    print(f"Total: {len(all_results)} file-segments")
    print(f"  ArduPilot: {ardu}")
    print(f"  PX4:       {px4}")

    with open("dwt_realflight_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[OK] Saved: dwt_realflight_results.json")

if __name__ == "__main__":
    main()
