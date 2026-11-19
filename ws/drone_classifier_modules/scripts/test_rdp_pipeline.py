import os
import sys
import argparse
import concurrent.futures
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

# Add src to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import config
from data_extractor import parse_px4_ulog, parse_ardu_bin, parse_real_csv
from rdp_utils import generate_rdp_plots

def process_single_run(run_folder, base_folder_name, is_sitl, epsilon):
    BASE_DATA_DIR = Path("data") / base_folder_name
    SAVE_BASE_DIR = Path("results") / f"{base_folder_name}_viz"
    
    print(f"[Processing RDP] {run_folder} in {base_folder_name}...")
    run_path = BASE_DATA_DIR / run_folder
    save_path_run = SAVE_BASE_DIR / run_folder
    save_path_run.mkdir(parents=True, exist_ok=True)

    for fw_config in config.get_fw_configs(is_sitl):
        if fw_config.name == 'cogni': continue
        
        sub_dir = config.find_fw_dir(run_path, fw_config.name, fw_config.sub_paths)
        
        if sub_dir:
            for file in os.listdir(sub_dir):
                if file.lower().endswith(fw_config.ext):
                    path = str(sub_dir / file)
                    
                    if fw_config.name == 'px4' and is_sitl: raw_data = parse_px4_ulog(path)
                    elif fw_config.name == 'ardu' and is_sitl: raw_data = parse_ardu_bin(path)
                    else: raw_data = parse_real_csv(path, measurement_type='mocap')
                    
                    if raw_data is not None:
                        out_prefix = save_path_run / f"{run_folder}_rdp_{fw_config.name}_eps{epsilon}"
                        title_prefix = f"{fw_config.name.upper()} ({run_folder})"
                        generate_rdp_plots(raw_data, epsilon, out_prefix, title_prefix)
                    break

def run_rdp_pipeline(base_folder_name, is_sitl, epsilon, max_runs=None):
    BASE_DATA_DIR = Path("data") / base_folder_name
    
    if not BASE_DATA_DIR.exists():
        print(f"[Error] Folder not found: {BASE_DATA_DIR}")
        return

    run_folders = sorted([f for f in os.listdir(BASE_DATA_DIR) if f.startswith("run_") and (BASE_DATA_DIR / f).is_dir()])

    if max_runs is not None:
        print(f"[Info] Limiting RDP to {max_runs} runs (out of {len(run_folders)}).")
        run_folders = run_folders[:max_runs]

    print(f"\n[Info] Starting multiprocessing for {len(run_folders)} runs...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_folder = {
            executor.submit(process_single_run, run_folder, base_folder_name, is_sitl, epsilon): run_folder 
            for run_folder in run_folders
        }
        
        for future in concurrent.futures.as_completed(future_to_folder):
            run_folder = future_to_folder[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[Error] Exception generated while processing {run_folder}: {exc}")

def main():
    parser = argparse.ArgumentParser(description="Apply RDP to flight trajectories.")
    parser.add_argument("log_folder", type=str, nargs='?', default="sitl_logs", help="Name of the log folder (e.g., 'sitl_logs')")
    parser.add_argument("--sitl", action="store_true", help="Set this flag if the logs are SITL logs")
    parser.add_argument("--real", dest="sitl", action="store_false", help="Set this flag if the logs are real flight logs")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Epsilon parameter for RDP")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit the number of runs to process")
    parser.set_defaults(sitl=True)
    
    args = parser.parse_args()
    
    print(f"[Info] Starting RDP pipeline for folder: {args.log_folder} (SITL={args.sitl}, eps={args.epsilon})")
    run_rdp_pipeline(args.log_folder, is_sitl=args.sitl, epsilon=args.epsilon, max_runs=args.max_runs)
    print("\n[Success] All RDP visualizations generated successfully.")

if __name__ == "__main__":
    main()
