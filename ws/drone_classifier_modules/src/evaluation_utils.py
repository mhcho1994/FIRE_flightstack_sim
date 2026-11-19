import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Use 'Agg' backend for non-interactive environments
matplotlib.use('Agg')

def plot_confusion_matrix(y_true, y_pred, target_names, model_name="SVM", save_dir="results/svm_figs"):
    """
    Visualizes the model's prediction results as a Confusion Matrix.
    """
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(f"{model_name} Classification Confusion Matrix", fontweight='bold')
    
    plt.tight_layout()
    file_path = f"{save_dir}/{model_name.lower()}_confusion_matrix.png"
    plt.savefig(file_path, dpi=150)
    plt.close()
    print(f"[Info] Saved Confusion Matrix plot to '{file_path}'")

def plot_pca_2d_projection(X_train, X_test, y_train, y_test, X_real=None, y_real=None, y_real_pred=None, model_name="SVM", save_dir="results/svm_figs"):
    """
    Visualizes the distribution of the DWT feature space and actual prediction results 
    by projecting them onto a 2D PCA plane.
    """
    os.makedirs(save_dir, exist_ok=True)
    class_colors = {0: 'tab:green', 1: 'tab:orange', 2: 'tab:blue'}
    label_map = {0: "PX4", 1: "ArduPilot", 2: "Cogni"}

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    fig, ax = plt.subplots(figsize=(11, 7))
    
    # 1. SITL Data (Train / Test)
    for cls in [0, 1]:
        mask_train = (y_train == cls)
        ax.scatter(X_train_pca[mask_train, 0], X_train_pca[mask_train, 1], 
                   c=class_colors[cls], alpha=0.15, marker='o', label=f'SITL {label_map[cls]} (Train)')
        
        mask_test = (y_test == cls)
        ax.scatter(X_test_pca[mask_test, 0], X_test_pca[mask_test, 1], 
                   c=class_colors[cls], alpha=0.6, marker='o', s=80, edgecolor='k', label=f'SITL {label_map[cls]} (Test)')

    # 2. Real Flight Data (Matches / Fails)
    if X_real is not None and len(X_real) > 0 and y_real is not None and y_real_pred is not None:
        X_real_pca = pca.transform(X_real)
        y_real_array = np.array(y_real)
        
        for cls in [0, 1, 2]:
            cls_mask = (y_real_array == cls)
            if not np.any(cls_mask): continue
            
            correct_mask = cls_mask & (y_real_array == y_real_pred)
            if np.any(correct_mask):
                ax.scatter(X_real_pca[correct_mask, 0], X_real_pca[correct_mask, 1], 
                           c=class_colors[cls], marker='*', s=350, edgecolor='k', linewidths=0.8,
                           zorder=5, label=f'Real {label_map[cls]} (Match)')
            
            incorrect_mask = cls_mask & (y_real_array != y_real_pred)
            if np.any(incorrect_mask):
                ax.scatter(X_real_pca[incorrect_mask, 0], X_real_pca[incorrect_mask, 1], 
                           c=class_colors[cls], marker='X', s=200, edgecolor='red', linewidths=1.5,
                           zorder=6, label=f'Real {label_map[cls]} (Fail)')

    ax.set_title(f"PCA 2D Projection & Prediction Status ({model_name})", fontsize=15, fontweight='bold')
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    file_path = f"{save_dir}/{model_name.lower()}_pca_scatter.png"
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Info] Saved PCA Scatter plot to '{file_path}'")

def print_detailed_prediction_map(y_real, y_real_pred, runs_real):
    """
    Outputs the detailed prediction results of real flight data in a console table format.
    """
    label_map = {0: "PX4", 1: "ArduPilot", 2: "Cogni"}
    
    print("\n[Detailed Prediction Map]")
    print("-" * 85)
    print(f"{'No.':<4} | {'Run Folder':<30} | {'True Label':<12} | {'Prediction':<12} | {'Status'}")
    print("-" * 85)
    
    for i in range(len(y_real)):
        run_name = runs_real[i]
        true_name = label_map.get(y_real[i], "Unknown")
        pred_name = label_map.get(y_real_pred[i], "Unknown")
        status = "Match" if true_name == pred_name else "Fail"
        print(f"{i+1:<4} | {run_name:<30} | {true_name:<12} | {pred_name:<12} | {status}")
    print("-" * 85)