import os
import sys
import argparse
import concurrent.futures
from pathlib import Path

import numpy as np
import pywt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

matplotlib.use('Agg')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import config
import kinematic_processor
from data_extractor import parse_px4_ulog, parse_ardu_bin, parse_real_csv
from kinematic_processor import compute_kinematics_diff
from flight_segmenter import extract_segments, get_segmentation_details
from visualization_utils import (
    plot_3d_trajectory_comparison,
    plot_state_variables,
    plot_full_trajectory_with_spans,
    plot_turn_segment_features,
    plot_dwt_features,
    plot_hmm_viterbi_states,
    plot_turn_context_comparison
)

def generate_comparison_plots(data_store, run_folder, save_path_run, base_folder_name, selected_indices, all_feature_names):
    """Generates comparison plots (Type 0, Type 1, and Custom Type 6) for SITL data."""
    # Type 0: Trajectory 3D Plot (Combined)
    p = data_store['px4']['raw_xy']
    a = data_store['ardu']['raw_xy']
    plot_3d_trajectory_comparison(
        p[0], p[1], p[2], a[0], a[1], a[2], ['PX4', 'ArduPilot'],
        f"3D Trajectory Comparison ({run_folder})",
        str(save_path_run / f"{run_folder}_0_combined_traj3d.png")
    )
    
    # Type 1: State Variables Comparison
    plot_state_variables(
        [data_store['px4']['t_raw'], data_store['ardu']['t_raw']],
        [data_store['px4']['states'], data_store['ardu']['states']],
        ['PX4', 'ArduPilot'], ['tab:green', 'tab:orange'],
        f"State Variables Comparison ({run_folder})",
        str(save_path_run / f"{run_folder}_1_combined_states.png")
    )

    # Type 6: First Turn Context Comparison (Only for specific folder)
    if base_folder_name == "sitl_logs":
        heading_idx = config.FEATURE_MAP.get('Heading', 1)
        xyspeed_idx = config.FEATURE_MAP.get('XY-Speed', 3)
        
        type6_indices = [heading_idx, xyspeed_idx] + [idx for idx in selected_indices if idx not in (heading_idx, xyspeed_idx)]
        
        plot_turn_context_comparison(
            data_store['px4'], 
            data_store['ardu'], 
            type6_indices, 
            all_feature_names, 
            f"First Turn Context Comparison ({run_folder})",
            str(save_path_run / f"{run_folder}_6_turn_context_comparison.png")
        )

def generate_individual_plots(data_store, run_folder, save_path_run, selected_indices, all_feature_names, include_trajectory=True):
    """Generates individual plots (Type 0-5) for each firmware."""
    for fw, d in data_store.items():
        if include_trajectory:
            # Type 0: Trajectory 3D Plot
            x, y, z = d['raw_xy']
            plot_3d_trajectory_comparison(
                x1=x, y1=y, z1=z, 
                x2=None, y2=None, z2=None, 
                labels=[fw.upper(), ''], 
                title=f"3D Trajectory: {fw.upper()} ({run_folder})",
                save_path=str(save_path_run / f"{run_folder}_0_{fw}_traj3d.png"),
                colors=[d['color'], 'k'] 
            )
            # Type 1: State Variables
            plot_state_variables(
                [d['t_raw']], [d['states']], [fw.upper()], [d['color']],
                f"State Variables: {fw.upper()} ({run_folder})",
                str(save_path_run / f"{run_folder}_1_{fw}_states.png")
            )

        # Type 2: HMM Segmentation
        plot_hmm_viterbi_states(
            t=d['kinematic_t'], probs=d['probs'], viterbi_labels=d['viterbi_labels'],
            title=f"HMM Emission Probabilities & Viterbi Decoding: {fw.upper()} ({run_folder})",
            save_path=str(save_path_run / f"{run_folder}_2_{fw}_hmm_viterbi.png")
        )

        # Type 3: Segment Check (Full Highlight)
        spans_for_plot = d['spans'].copy()
        if 'turn_left' in spans_for_plot or 'turn_right' in spans_for_plot:
            spans_for_plot['turn'] = spans_for_plot.pop('turn_left', []) + spans_for_plot.pop('turn_right', [])
        plot_full_trajectory_with_spans(
            t=d['t_full'], features=d['feat'], spans=spans_for_plot,
            title=f"Segment Check: {fw.upper()} ({run_folder})",
            save_path=str(save_path_run / f"{run_folder}_3_{fw}_seg_check.png"),
            line_color=d['color']
        )

        # Type 4 & 5: Turn Segments
        if d['segs']:
            turn_segments = d['segs'].get('turn_left', []) + d['segs'].get('turn_right', [])
            for i, seg in enumerate(turn_segments):
                seg_id = f"seg{i+1}"
                span_start, span_end = seg['span']
                idx = (d['t_full'] >= span_start) & (d['t_full'] <= span_end)
                seg_t = d['t_full'][idx]
                seg_feat = d['feat'][idx]

                # Type 4: Isolated Segment Features
                plot_turn_segment_features(
                    t=seg_t, features=seg_feat,
                    title=f"Turn Segment Features: {fw.upper()} {seg_id}",
                    save_path=str(save_path_run / f"{run_folder}_4_{fw}_{seg_id}_features.png"),
                    line_color=d['color']
                )
                
                # Type 5: DWT Result
                plot_dwt_features(
                    t=seg_t, data=seg_feat,
                    target_indices=selected_indices, feature_names=all_feature_names,
                    title=f"DWT Decomposition: {fw.upper()} {seg_id}",
                    save_path=str(save_path_run / f"{run_folder}_5_{fw}_{seg_id}_dwt.png"),
                    color=d['color']
                )

def process_single_run(run_folder, base_folder_name, is_sitl, selected_indices, all_feature_names):
    BASE_DATA_DIR = Path("data") / base_folder_name
    SAVE_BASE_DIR = Path("results") / f"{base_folder_name}_viz"
    
    print(f"[Processing] {run_folder} in {base_folder_name}...")
    run_path = BASE_DATA_DIR / run_folder
    save_path_run = SAVE_BASE_DIR / run_folder
    save_path_run.mkdir(parents=True, exist_ok=True)

    # 1. Load Data (PX4, ArduPilot)
    data_store = {}
    
    for fw_config in config.get_fw_configs(is_sitl):
        if fw_config.name == 'cogni': continue # visualization script doesn't support cogni yet
        
        sub_dir = config.find_fw_dir(run_path, fw_config.name, fw_config.sub_paths)
        
        if sub_dir:
            for file in os.listdir(sub_dir):
                if file.lower().endswith(fw_config.ext):
                    path = str(sub_dir / file)
                    
                    # Step 1: Extract Raw Data
                    if fw_config.name == 'px4' and is_sitl: raw_data = parse_px4_ulog(path)
                    elif fw_config.name == 'ardu' and is_sitl: raw_data = parse_ardu_bin(path)
                    else: raw_data = parse_real_csv(path, measurement_type='mocap')
                    
                    # Step 2 & 3: Kinematics and Segmentation
                    if raw_data is not None:
                        t_full, feat_full = compute_kinematics_diff(raw_data)
                        kinematic_features = kinematic_processor.compute_kinematics_pca(raw_data)
                        segs, spans = extract_segments(kinematic_features)
                        
                        probs, viterbi_labels = get_segmentation_details(kinematic_features)

                        states = np.vstack((raw_data['x'], raw_data['y'], raw_data['z'], 
                                            raw_data['vx'], raw_data['vy'], raw_data['vz'])).T
                        
                        data_store[fw_config.name] = {
                            't_raw': raw_data['t'], 't_full': t_full, 'states': states, 'feat': feat_full, 
                            'segs': segs, 'spans': spans, 'color': fw_config.color,
                            'raw_xy': (raw_data['x'], raw_data['y'], raw_data['z']),
                            'probs': probs, 'viterbi_labels': viterbi_labels, 'kinematic_t': kinematic_features['t_window']
                        }
                    break

    # 2. Generate Plots
    if is_sitl and 'px4' in data_store and 'ardu' in data_store:
        generate_comparison_plots(data_store, run_folder, save_path_run, base_folder_name, selected_indices, all_feature_names)
        generate_individual_plots(data_store, run_folder, save_path_run, selected_indices, all_feature_names, include_trajectory=False)
    else:
        generate_individual_plots(data_store, run_folder, save_path_run, selected_indices, all_feature_names, include_trajectory=True)

def run_visualization_pipeline(base_folder_name, is_sitl=True, max_runs=None):
    BASE_DATA_DIR = Path("data") / base_folder_name
    
    if not BASE_DATA_DIR.exists():
        print(f"[Error] Folder not found: {BASE_DATA_DIR}")
        return

    run_folders = sorted([f for f in os.listdir(BASE_DATA_DIR) 
                         if f.startswith("run_") and (BASE_DATA_DIR / f).is_dir()])

    if max_runs is not None:
        print(f"[Info] Limiting visualization to {max_runs} runs (out of {len(run_folders)}).")
        run_folders = run_folders[:max_runs]

    # Get feature names and indices directly from config definitions
    all_feature_names = [feat.plot_label for feat in config.FEATURE_DEFINITIONS]
    target_features_dwt = config.TARGET_FEATURES
    selected_indices = [config.FEATURE_MAP[f] for f in target_features_dwt]

    print(f"\n[Info] Starting multiprocessing for {len(run_folders)} runs...")
    
    # Process runs in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_folder = {
            executor.submit(
                process_single_run, 
                run_folder, 
                base_folder_name, 
                is_sitl, 
                selected_indices, 
                all_feature_names
            ): run_folder 
            for run_folder in run_folders
        }
        
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(future_to_folder):
            run_folder = future_to_folder[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[Error] Exception generated while processing {run_folder}: {exc}")

def main():
    parser = argparse.ArgumentParser(description="Generate flight visualizations from logs.")
    parser.add_argument("log_folder", type=str, nargs='?', default="260615_sitl_logs", help="Name of the log folder (e.g., '260615_sitl_logs')")
    parser.add_argument("--sitl", action="store_true", help="Set this flag if the logs are SITL logs")
    parser.add_argument("--real", dest="sitl", action="store_false", help="Set this flag if the logs are real flight logs")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit the number of runs to visualize")
    parser.set_defaults(sitl=True)
    
    args = parser.parse_args()
    
    print(f"[Info] Starting visualization pipeline for folder: {args.log_folder} (SITL={args.sitl})")
    run_visualization_pipeline(args.log_folder, is_sitl=args.sitl, max_runs=args.max_runs)

    print("\n[Success] All visualizations generated successfully.")

if __name__ == "__main__":
    main()