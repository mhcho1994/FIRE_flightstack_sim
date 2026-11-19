import sys
import os
import argparse
import matplotlib
matplotlib.use('Agg')

# Add src to sys.path to import data_extractor and rdp_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_extractor import parse_px4_ulog, parse_ardu_bin
from rdp_utils import generate_rdp_plots
import pathlib

def main():
    parser = argparse.ArgumentParser(description="Apply RDP to trajectory directly from log")
    parser.add_argument("--log", type=str, required=True, help="Path to the ULG or BIN log file")
    parser.add_argument("--log-type", type=str, choices=['px4', 'ardu'], default='px4', help="Type of the log (px4 or ardu)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Epsilon parameter for RDP")
    parser.add_argument("--output", type=str, default="rdp_trajectory.png", help="Output plot image path (used as base prefix)")
    args = parser.parse_args()

    # Load data using data_extractor
    if args.log_type == 'px4':
        data = parse_px4_ulog(args.log)
    else:
        data = parse_ardu_bin(args.log)
        
    if data is None:
        print("Failed to parse log file.")
        return
        
    out_path = pathlib.Path(args.output)
    base_prefix = str(out_path.with_suffix(''))
    title_prefix = f"Raw Log ({args.log_type.upper()})"
    
    generate_rdp_plots(data, args.epsilon, base_prefix, title_prefix)
    print(f"RDP plots generated with prefix: {base_prefix}")

if __name__ == "__main__":
    main()
