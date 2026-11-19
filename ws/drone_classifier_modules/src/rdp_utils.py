import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from rdp import rdp
from pathlib import Path
import config
import kinematic_processor

def generate_rdp_plots(raw_data, epsilon, output_prefix, title_prefix):
    """
    Generates 3 plots (states, traj3d, features) for RDP simplified trajectory.
    output_prefix: str or Path, e.g., 'results/.../run_000_rdp_px4_eps0.01'
    title_prefix: str, e.g., 'PX4 (run_000)'
    """
    x, y, z = raw_data['x'], raw_data['y'], raw_data['z']
    vx, vy, vz = raw_data['vx'], raw_data['vy'], raw_data['vz']
    t = raw_data['t']
    
    xyz = np.column_stack((x, y, z))
    mask = rdp(xyz, epsilon=epsilon, return_mask=True)
    simplified_xyz = xyz[mask]
    simplified_t = t[mask]
    
    out_prefix = str(output_prefix)
    
    # 1. State Variables Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'RDP Trajectory Simplification: {title_prefix} (eps={epsilon})', fontsize=16, fontweight='bold')
    var_names = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 
                 'VX Velocity (m/s)', 'VY Velocity (m/s)', 'VZ Velocity (m/s)']
    states = np.column_stack((x, y, z, vx, vy, vz))
    simplified_states = np.column_stack((x[mask], y[mask], z[mask], vx[mask], vy[mask], vz[mask]))
    
    for i in range(6):
        ax = axes[i//2, i%2]
        ax.plot(t, states[:, i], color='tab:blue', label='Original', alpha=0.8)
        ax.scatter(simplified_t, simplified_states[:, i], color='red', s=30, label='RDP Waypoint', zorder=5)
        for rdp_t in simplified_t:
            ax.axvline(x=rdp_t, color='red', alpha=0.1, linestyle='--')
        ax.set_title(var_names[i], fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0: ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{out_prefix}_states.png", dpi=300)
    plt.close(fig)
    
    # 2. 3D Trajectory Plot
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=f'Original ({len(xyz)} pts)', color='blue', alpha=0.6)
    ax3d.plot(simplified_xyz[:, 0], simplified_xyz[:, 1], simplified_xyz[:, 2], 
              label=f'RDP (eps={epsilon}, {len(simplified_xyz)} pts)', color='red', marker='o', markersize=4, linewidth=2)
    ax3d.set_title(f'3D RDP Trajectory: {title_prefix}')
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.legend()
    plt.grid(True)
    limits = np.array([ax3d.get_xlim3d(), ax3d.get_ylim3d(), ax3d.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax3d.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax3d.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax3d.set_zlim3d([origin[2] - radius, origin[2] + radius])
    plt.savefig(f"{out_prefix}_traj3d.png", dpi=300)
    plt.close(fig3d)
    
    # 3. Features Plot
    t_full, feat_full = kinematic_processor.compute_kinematics_diff(raw_data)
    if feat_full is not None and len(feat_full) > 0:
        feature_names = [feat.plot_label for feat in config.FEATURE_DEFINITIONS]
        num_features = len(feature_names)
        num_rows = (num_features + 1) // 2
        
        fig_feat, axes_feat = plt.subplots(num_rows, 2, figsize=(16, 3 * num_rows + 3))
        fig_feat.suptitle(f"Kinematic Features with RDP Waypoints: {title_prefix} (eps={epsilon})", fontsize=18, fontweight='bold', y=0.98)
        axes_flat = axes_feat.flatten()
        
        for i in range(num_features):
            ax = axes_flat[i]
            ax.plot(t_full, feat_full[:, i], color='tab:green', linewidth=1.5, zorder=2)
            ax.set_title(feature_names[i], fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
            
            feat_id = config.FEATURE_DEFINITIONS[i].id
            if feat_id == 'JZ': ax.set_ylim(np.nanpercentile(feat_full[:, i], 2), np.nanpercentile(feat_full[:, i], 98))
            elif feat_id == 'XY-Jerk': ax.set_ylim(-1, np.nanpercentile(feat_full[:, i], 98))
            elif feat_id == 'Curvature': ax.set_ylim(-0.1, np.nanpercentile(feat_full[:, i], 98))

            for rdp_t in simplified_t:
                ax.axvline(x=rdp_t, color='red', alpha=0.3, linestyle='--', zorder=0)
        
        for j in range(num_features, len(axes_flat)):
            fig_feat.delaxes(axes_flat[j])
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(f"{out_prefix}_features.png", dpi=300)
        plt.close(fig_feat)
