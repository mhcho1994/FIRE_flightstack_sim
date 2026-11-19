"""
DWT + Random Forest drone autopilot classifier.
Same pipeline as train_dwt_lgbm.py but uses RandomForest instead of LightGBM.
"""

import random
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from train_dwt_lgbm import extract_dwt_features, extract_turns, MIN_SEG_LEN

PX4_FOLDER  = "../../data/px4_logs"
ARDU_FOLDER = "../../data/ardu_logs"
TEST_RATIO  = 0.2


def load_turns_from_files(file_list):
    from trajectory_processor import process_px4_flight_data, process_ardu_flight_data
    X, y = [], []
    for path, label in file_list:
        path = str(path)
        result = process_px4_flight_data(path) if label == 0 else process_ardu_flight_data(path)
        if result is None or result[4] is None:
            continue
        _, _, t_res, traj_res, feat7, _, _ = result
        if len(feat7) < MIN_SEG_LEN:
            continue
        for turn in extract_turns(t_res, feat7, traj_res, dt=1/50):
            if turn["n"] < MIN_SEG_LEN:
                continue
            X.append(extract_dwt_features(turn))
            y.append(label)
    return (np.array(X, dtype=np.float32) if X else np.empty((0, 72))), np.array(y)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_out = f"dwt_rf_{ts}.pkl"

    print(f"\n{'='*70}")
    print(f"  DWT + Random Forest Trainer  ({ts})")
    print(f"{'='*70}\n")

    px4_files  = [(p, 0) for p in sorted(Path(PX4_FOLDER).glob("*.ulg"))]
    ardu_files = [(p, 1) for p in sorted(Path(ARDU_FOLDER).glob("*.bin"))]
    random.shuffle(px4_files); random.shuffle(ardu_files)

    px4_test  = max(1, int(len(px4_files)  * TEST_RATIO)) if len(px4_files)  > 1 else 0
    ardu_test = max(1, int(len(ardu_files) * TEST_RATIO)) if len(ardu_files) > 1 else 0

    train_files = px4_files[px4_test:]  + ardu_files[ardu_test:]
    test_files  = px4_files[:px4_test]  + ardu_files[:ardu_test]

    print(f"Files — train: {len(train_files)}  test: {len(test_files)}")
    print("⏳ Extracting DWT features from train set...")
    X_train, y_train = load_turns_from_files(train_files)
    print("⏳ Extracting DWT features from test set...")
    X_test,  y_test  = load_turns_from_files(test_files)

    print(f"\nTurn segments — train: {len(X_train)}  test: {len(X_test)}")
    unique, counts = np.unique(y_train, return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"  {'PX4' if lbl==0 else 'ArduPilot'}: {cnt} ({100*cnt/len(y_train):.1f}%)")

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"\nTest accuracy (SITL): {accuracy_score(y_test, y_pred)*100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["PX4", "ArduPilot"]))

    with open(model_out, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved: {model_out}")

    feat_names = [
        f"ch{c}_{'cA3' if k==0 else 'cD3' if k==1 else 'cD2'}_{s}"
        for c in range(3)
        for k in range(3)
        for s in ["mean","std","energy","max","min","kurt","peak_max","peak_min"]
    ]
    importances = sorted(zip(feat_names, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    ch_map = {0: "speed_xy", 1: "ah", 2: "yaw_rate"}
    print("\nTop-15 features:")
    for name, imp in importances[:15]:
        ch_idx = int(name[2])
        print(f"  {imp:.4f}  {ch_map[ch_idx]}__{name[4:]}")


if __name__ == "__main__":
    main()
