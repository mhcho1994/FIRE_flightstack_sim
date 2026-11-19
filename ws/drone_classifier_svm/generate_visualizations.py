import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pywt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Use 'Agg' backend for non-interactive environments (WSL, Servers)
import matplotlib
matplotlib.use('Agg')

# Import your existing processing functions
from signal_processor import (
    process_px4_flight_data,
    process_ardu_flight_data,
    process_real_flight_data,
    FEATURE_MAP
)

# =====================================================================
# 1. Helper Functions for New Plot Types
# =====================================================================

def plot_3d_trajectory_comparison(x1, y1, z1, x2, y2, z2, labels, title, save_path, colors=None):
    """Generates a 3D comparison of two trajectories, or a single one if x2 is None."""
    if colors is None:
        colors = ['tab:green', 'tab:orange']
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot first trajectory
    if x1 is not None:
        ax.plot(y1, x1, z1, color=colors[0], label=labels[0], linewidth=1.5)
        ax.scatter(y1[0], x1[0], z1[0], color=colors[0], marker='o', s=80, edgecolor='k') # Start point
    
    # Plot second trajectory 
    if x2 is not None:
        ax.plot(y2, x2, z2, color=colors[1], label=labels[1], linewidth=1.5)
        ax.scatter(y2[0], x2[0], z2[0], color=colors[1], marker='o', s=80, edgecolor='k') # Start point

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('East (Y) [m]')
    ax.set_ylabel('North (X) [m]')
    ax.set_zlabel('Altitude (Z) [m]')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_state_variables(t_list, states_list, labels, colors, title, save_path):
    """Plots x, y, z, vx, vy, vz for comparison or single flight."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    var_names = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 
                 'VX Velocity (m/s)', 'VY Velocity (m/s)', 'VZ Velocity (m/s)']
    
    for i in range(6):
        ax = axes[i//2, i%2]
        for t, states, label, color in zip(t_list, states_list, labels, colors):
            ax.plot(t, states[:, i], color=color, label=label, alpha=0.8)
        
        ax.set_title(var_names[i], fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0: ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_full_trajectory_with_spans(t, features, spans, title, save_path, line_color):
    if features is None or len(features) == 0: return

    feature_names = ['Altitude (m)', 'Heading (rad)', 'Z-Axis Velocity (m/s)', 'XY-Plane Speed (m/s)', 'Z-Axis Acceleration (m/s²)', 'XY-Plane Accel Norm (m/s²)', 'Z-Axis Jerk (m/s³)', 'XY-Plane Jerk Norm (m/s³)', 'Curvature (1/m)', 'Yaw rate (rad/s)']
    
    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    axes_flat = axes.flatten()
    
    span_colors = {'takeoff': 'orange', 'straight': 'mediumseagreen', 'turn': 'crimson', 'landing': 'mediumpurple'}

    for i in range(10):
        ax = axes_flat[i]
        ax.plot(t, features[:, i], color=line_color, linewidth=1.5, zorder=2)
        ax.set_title(feature_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
        
        if i == 6: ax.set_ylim(np.nanpercentile(features[:, i], 2), np.nanpercentile(features[:, i], 98))
        elif i == 7: ax.set_ylim(-1, np.nanpercentile(features[:, i], 98))
        elif i == 8: ax.set_ylim(-0.1, np.nanpercentile(features[:, i], 98))

        if spans:
            if spans.get('takeoff'): ax.axvspan(*spans['takeoff'], color=span_colors['takeoff'], alpha=0.2, zorder=0)
            if spans.get('landing'): ax.axvspan(*spans['landing'], color=span_colors['landing'], alpha=0.2, zorder=0)
            for s, e in spans.get('straight', []): ax.axvspan(s, e, color=span_colors['straight'], alpha=0.2, zorder=0)
            for s, e in spans.get('turn', []): ax.axvspan(s, e, color=span_colors['turn'], alpha=0.4, zorder=0) 
    
    if spans:
        legend_patches = [
            mpatches.Patch(color=span_colors['takeoff'], alpha=0.2, label='Take-off'),
            mpatches.Patch(color=span_colors['straight'], alpha=0.2, label='Straight'),
            mpatches.Patch(color=span_colors['turn'], alpha=0.4, label='Turn (Target)'),
            mpatches.Patch(color=span_colors['landing'], alpha=0.2, label='Landing')
        ]
        fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.95, 0.98), ncol=4, fontsize=12)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_turn_segment_features(t, features, title, save_path, line_color):
    if features is None or len(features) == 0: return

    feature_names = ['Altitude (m)', 'Heading (rad)', 'Z-Axis Velocity (m/s)', 'XY-Plane Speed (m/s)', 'Z-Axis Acceleration (m/s²)', 'XY-Plane Accel Norm (m/s²)', 'Z-Axis Jerk (m/s³)', 'XY-Plane Jerk Norm (m/s³)', 'Curvature (1/m)', 'Yaw rate (rad/s)']
    
    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    axes_flat = axes.flatten()
    
    for i in range(10):
        ax = axes_flat[i]
        ax.plot(t, features[:, i], color=line_color, linewidth=2.0)
        ax.set_title(feature_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Absolute Time (s)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 6: ax.set_ylim(np.nanpercentile(features[:, i], 2), np.nanpercentile(features[:, i], 98))
        elif i == 7: ax.set_ylim(-1, np.nanpercentile(features[:, i], 98))
        elif i == 8: ax.set_ylim(-0.1, np.nanpercentile(features[:, i], 98))
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_dwt_features(t, data, target_indices, feature_names, title, save_path, color, wavelet='db4', level=3):
    """Visualizes DWT decomposition for selected features of a segment."""
    n_feat = len(target_indices)
    fig, axes = plt.subplots(n_feat, level + 1, figsize=(4 * (level+1), 3 * n_feat))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    t_rel = t - t[0]
    
    for i, f_idx in enumerate(target_indices):
        sig = data[:, f_idx]
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        
        # Original Signal
        axes[i, 0].plot(t_rel, sig, color=color)
        axes[i, 0].set_ylabel(feature_names[f_idx], fontweight='bold')
        if i == 0: axes[i, 0].set_title("Original")
        
        # Details (cD)
        for l in range(level):
            ax = axes[i, l+1]
            # Rescale time for coefficients
            t_coeff = np.linspace(0, t_rel[-1], len(coeffs[level-l]))
            ax.plot(t_coeff, coeffs[level-l], color=color, alpha=0.7)
            if i == 0: ax.set_title(f"Detail Level {l+1}")
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()

# =====================================================================
# 2. Main Visualization Engine
# =====================================================================

def run_visualization_pipeline(base_folder_name, is_sitl=True):
    BASE_DATA_DIR = Path("data") / base_folder_name
    SAVE_BASE_DIR = Path("data") / f"{base_folder_name}_visualization"
    if not BASE_DATA_DIR.exists():
        print(f"[Error] Folder not found: {BASE_DATA_DIR}")
        return

    if not BASE_DATA_DIR.exists():
        print(f"[Error] Folder not found: {BASE_DATA_DIR}")
        return

    run_folders = sorted([f for f in os.listdir(BASE_DATA_DIR) 
                         if f.startswith("run_") and (BASE_DATA_DIR / f).is_dir()])

    # All feature names for reference
    all_feature_names = ['Alt', 'Heading', 'VZ', 'XY-Speed', 'AZ', 'XY-Accel', 'JZ', 'XY-Jerk', 'Curvature', 'YawRate', 'Slip']
    target_features_dwt = ['XY-Accel', 'XY-Jerk', 'Curvature']
    selected_indices = [FEATURE_MAP[f] for f in target_features_dwt]

    for run_folder in run_folders:
        print(f"\n[Processing] {run_folder} in {base_folder_name}...")
        run_path = BASE_DATA_DIR / run_folder
        save_path_run = SAVE_BASE_DIR / run_folder
        save_path_run.mkdir(parents=True, exist_ok=True)

        # 1. Load Data (PX4, ArduPilot, Cogni)
        data_store = {}
        
        # Firmware search logic
        configs = [
            ('px4', run_path / "px4_logs", '.ulg' if is_sitl else '.csv', 'tab:green'),
            ('ardu', run_path / "ardu_logs", '.bin' if is_sitl else '.csv', 'tab:orange'),
            ('cogni', run_path / "cogni_logs", '.csv', 'tab:blue')
        ]

        for fw, fw_path, ext, color in configs:
            sub_dir = fw_path / ("raw" if is_sitl else "processed")
            if fw == 'ardu' and is_sitl: sub_dir = fw_path / "raw" / "logs"
            
            if sub_dir.exists():
                for file in os.listdir(sub_dir):
                    if file.lower().endswith(ext):
                        path = str(sub_dir / file)
                        if fw == 'px4' and is_sitl: res = process_px4_flight_data(path)
                        elif fw == 'ardu' and is_sitl: res = process_ardu_flight_data(path)
                        else: res = process_real_flight_data(path, measurement_type='mocap')
                        
                        if res[0] is not None:
                            raw_data, t_full, feat_full, segs, spans = res

                            states = np.vstack((raw_data['x'], raw_data['y'], raw_data['z'], 
                                                raw_data['vx'], raw_data['vy'], raw_data['vz'])).T
                            
                            data_store[fw] = {
                                't_raw': raw_data['t'], 't_full': t_full, 'states': states, 'feat': feat_full, 
                                'segs': segs, 'spans': spans, 'color': color,
                                'raw_xy': (raw_data['x'], raw_data['y'], raw_data['z'])
                            }
                        break

        # =============================================================
        # Type 0: Trajectory 3D Plot (Combined or Individual)
        # =============================================
        if is_sitl and 'px4' in data_store and 'ardu' in data_store:
            # SITL: Overlap 3D
            p = data_store['px4']['raw_xy']
            a = data_store['ardu']['raw_xy']
            plot_3d_trajectory_comparison(
                p[0], p[1], p[2], a[0], a[1], a[2], ['PX4', 'ArduPilot'],
                f"3D Trajectory Comparison ({run_folder})",
                str(save_path_run / f"{run_folder}_0_traj_3d_combined.png")
            )
        else:
            # Real Flight (or incomplete SITL): Separate 3D per firmware
            for fw, d in data_store.items():
                x, y, z = d['raw_xy']
                plot_3d_trajectory_comparison(
                    x1=x, y1=y, z1=z, 
                    x2=None, y2=None, z2=None, 
                    labels=[fw.upper(), ''], 
                    title=f"3D Trajectory: {fw.upper()} ({run_folder})",
                    save_path=str(save_path_run / f"{run_folder}_0_traj_3d_{fw}.png"),
                    colors=[d['color'], 'k'] # fw 고유 색상 적용
                )

        # =============================================================
        # Type 1: State Variables Comparison (x,y,z,vx,vy,vz)
        # =============================================
        if is_sitl and 'px4' in data_store and 'ardu' in data_store:
            # SITL: Overlap
            plot_state_variables(
                [data_store['px4']['t_raw'], data_store['ardu']['t_raw']],
                [data_store['px4']['states'], data_store['ardu']['states']],
                ['PX4', 'ArduPilot'], ['tab:green', 'tab:orange'],
                f"State Variables Comparison ({run_folder})",
                str(save_path_run / f"{run_folder}_1_states_compare.png")
            )
        else:
            # Real: Separate per firmware
            for fw, d in data_store.items():
                plot_state_variables(
                    [d['t_raw']], [d['states']], [fw.upper()], [d['color']],
                    f"State Variables: {fw.upper()} ({run_folder})",
                    str(save_path_run / f"{run_folder}_1_{fw}_states.png")
                )

        # =============================================================
        # Type 2 & 3 & 4: Segments, Features, and DWT
        # =============================================
        for fw, d in data_store.items():
            # Type 2: Segment Check (Full Highlight)
            plot_full_trajectory_with_spans(
                t=d['t_full'], features=d['feat'], spans=d['spans'],
                title=f"Segment Check: {fw.upper()} ({run_folder})",
                save_path=str(save_path_run / f"{run_folder}_2_{fw}_seg_check.png"),
                line_color=d['color']
            )

            # Type 3 & 4: Turn Segments
            for i, seg in enumerate(d['segs']['turn']):
                seg_id = f"seg{i+1}"
                # Type 3: Isolated Segment Features
                plot_turn_segment_features(
                    t=seg['time'], features=seg['features'],
                    title=f"Turn Segment Features: {fw.upper()} {seg_id}",
                    save_path=str(save_path_run / f"{run_folder}_3_{fw}_{seg_id}_features.png"),
                    line_color=d['color']
                )
                
                # Type 4: DWT Result (Added as per brainstorming)
                plot_dwt_features(
                    t=seg['time'], data=seg['features'],
                    target_indices=selected_indices, feature_names=all_feature_names,
                    title=f"DWT Decomposition: {fw.upper()} {seg_id}",
                    save_path=str(save_path_run / f"{run_folder}_4_{fw}_{seg_id}_dwt.png"),
                    color=d['color']
                )

def main():
    # 1. Process SITL Logs
    # run_visualization_pipeline("sitl_logs", is_sitl=True)
    
    # 2. Process Real Flight Logs (Update folder name as needed)
    run_visualization_pipeline("260417_flight_logs", is_sitl=False)
    run_visualization_pipeline("260424_flight_logs", is_sitl=False)
    run_visualization_pipeline("260501_flight_logs_old", is_sitl=False)

    print("\n[Success] All visualizations generated in 'results_svm_figs/' directory.")

if __name__ == "__main__":
    main()