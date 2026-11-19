import os
from pathlib import Path

import config
import flight_segmenter
import kinematic_processor
import numpy as np
# Import our modular pipeline components
from data_extractor import parse_ardu_bin, parse_px4_ulog, parse_real_csv


def process_dataset_folder(base_folder, is_sitl=True, measurement_type='mocap', max_runs=None):
    """
    Crawls folders, runs the ETL pipeline (Extract -> Kinematics -> Segment), 
    and returns extracted Turn segments.
    """
    base_dir = Path("data") / base_folder
    X_ts, T_ts, y, runs = [], [], [], []
    
    if not base_dir.exists():
        print(f"[Warning] Directory not found: {base_dir}")
        return X_ts, T_ts, np.array(y), runs

    run_folders = sorted([f for f in os.listdir(base_dir) if f.startswith("run_") and (base_dir / f).is_dir()])
    
    if max_runs is not None:
        print(f"[Info] Limiting processing to {max_runs} runs for {base_folder} (out of {len(run_folders)}).")
        run_folders = run_folders[:max_runs]
    
    # AI Classification Target Features
    target_features = config.TARGET_FEATURES
    target_indices = [config.FEATURE_MAP[f] for f in target_features]

    for run_folder in run_folders:
        run_dir = base_dir / run_folder
        
        for fw_config in config.get_fw_configs(is_sitl):
            # Skip Cogni if it's SITL data
            if is_sitl and fw_config.name == 'cogni': continue
            
            fw_dir = config.find_fw_dir(run_dir, fw_config.name, fw_config.sub_paths)
            if not fw_dir:
                continue
                
            for file in os.listdir(fw_dir):
                if file.lower().endswith(fw_config.ext):
                    file_path = str(fw_dir / file)
                    
                    # [Step 1: Extract Raw Data]
                    if is_sitl and fw_config.name == 'px4': raw_data = parse_px4_ulog(file_path)
                    elif is_sitl and fw_config.name == 'ardu': raw_data = parse_ardu_bin(file_path)
                    else: raw_data = parse_real_csv(file_path, measurement_type)
                    
                    if raw_data is None: continue
                    
                    # [Step 2 & 3: Transform (Kinematics -> Segment)]
                    # Use new kinematic processor for HMM-based segmentation
                    kinematic_features = kinematic_processor.compute_kinematics_pca(raw_data)
                    if kinematic_features is None: continue
                    segs, spans = flight_segmenter.extract_segments(kinematic_features)
                    
                    # Use diff-based kinematic processor for AI features extraction
                    t_full, feat_full = kinematic_processor.compute_kinematics_diff(raw_data)
                    if t_full is None or feat_full is None: continue

                    # [Step 4: Load Turn Features]
                    count_in_run = 0
                    turn_spans = spans.get('turn_left', []) + spans.get('turn_right', [])
                    
                    if len(turn_spans) > 0:
                        for span in turn_spans:
                            # Extract corresponding indices using time spans from the new segmenter
                            indices = np.where((t_full >= span[0]) & (t_full <= span[1]))[0]
                            if len(indices) == 0: continue
                            
                            feat_turn = feat_full[indices][:, target_indices]
                            time_turn = t_full[indices] - t_full[indices][0]
                            X_ts.append(feat_turn)
                            T_ts.append(time_turn)
                            y.append(fw_config.class_label)
                            runs.append(f"{base_folder}/{run_folder}")
                            count_in_run += 1
                            
                    if count_in_run > 0:
                        print(f"    - {run_folder} [{fw_config.name.upper()}]: {count_in_run} turn segments extracted.")
                    break # Process only the first valid file per firmware

    return X_ts, T_ts, np.array(y), runs
