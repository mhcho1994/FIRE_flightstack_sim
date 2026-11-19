import torch
import torch.nn as nn
import glob
import numpy as np
import matplotlib.pyplot as plt
from model import DroneTrajectoryCNN, DroneTrajectoryCNNLSTM
import os
from pathlib import Path

def analyze_feature_importance(model_type="cnn"):
    """기존 trained 모델의 feature importance 분석
    
    Args:
        model_type: "cnn" or "lstm"
    """

    model = None
    model_file = None
    
    # ==========================================
    # 1. Find available model files
    # ==========================================
    print("\nSearching for model files...\n")
    
    model_files = {
        "cnn": sorted(glob.glob('drone_cnn_*.pth')),
        "lstm": sorted(glob.glob('drone_cnn_lstm_*.pth'))
    }
    
    if model_type.lower() not in model_files:
        print(f"❌ Error: Unknown model type '{model_type}'")
        print(f"   Available types: {list(model_files.keys())}")
        return
    
    available_models = model_files[model_type.lower()]
    
    if not available_models:
        print(f"❌ No {model_type.upper()} model files found!")
        print(f"   Looking for: drone_{model_type.lower()}_*.pth")
        return
    
    print(f"Available {model_type.upper()} models ({len(available_models)}):")
    for i, f in enumerate(available_models):
        print(f"  [{i}] {f}")
    
    # User selects CNN model
    while True:
        try:
            cnn_idx = int(input(f"\nSelect CNN model [0-{len(available_models)-1}]: "))
            if 0 <= cnn_idx < len(available_models):
                pth_file = available_models[cnn_idx]
                break
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

    # User selects CNN-LSTM model
    while True:
        try:
            cnn_lstm_idx = int(input(f"Select CNN-LSTM model [0-{len(available_models)-1}]: "))
            if 0 <= cnn_lstm_idx < len(available_models):
                cnn_lstm_path = available_models[cnn_lstm_idx]
                break
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

    # ==========================================
    # 2. Load trained model
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    try:
        if model_type.lower() == "cnn":
            model = DroneTrajectoryCNN(num_features=10)
        else:  # lstm
            model = DroneTrajectoryCNNLSTM(num_features=10, lstm_hidden_size=32, num_lstm_layers=1)
        
        model.load_state_dict(torch.load(pth_file, map_location=device))
        model.to(device)
        print(f"✓ Model loaded successfully from: {pth_file}\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    model.eval()

    feature_names = ['h_smooth', 'heading', 'vh_smooth', 'speed_xy', 'ah', 'acc_norm_xy', 'j_alt', 'jerk_norm_xy', 'curvature_smooth', 'yaw_rate_smooth']
    print(f"{'='*70}")
    print(f"🔍 FEATURE IMPORTANCE ANALYSIS ({model_type.upper()})")
    print(f"   Model: {pth_file}")
    print(f"{'='*70}\n")

    # ==========================================
    # 3. Analyze Conv1d weights
    # ==========================================
    print("📈 Conv1d Layer Weight Analysis\n")
    
    conv1_weights = model.conv1.weight.data  # (out_channels, in_channels, kernel_size)
    # 각 입력 특성별로 가중치의 절댓값 평균
    feature_magnitude = np.mean(np.abs(conv1_weights.cpu().numpy()), axis=(0, 2))
    
    # 정규화 (0-100)
    feature_magnitude_norm = (feature_magnitude / feature_magnitude.max()) * 100
    
    print("Average weight magnitude per feature in Conv1d:")
    print(f"{'─'*50}")
    
    # 중요도 순서로 정렬
    sorted_indices = np.argsort(feature_magnitude_norm)[::-1]
    
    for rank, idx in enumerate(sorted_indices, 1):
        bar = "█" * int(feature_magnitude_norm[idx] / 5)
        print(f"{rank}. {feature_names[idx]:20s}: {feature_magnitude_norm[idx]:6.2f} {bar}")
    
    print(f"{'─'*50}\n")
    
    # ==========================================
    # 4. Visualize
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1) Weight Magnitude (정렬)
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_magnitudes = [feature_magnitude_norm[i] for i in sorted_indices]
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_magnitudes)))
    axes[0].barh(sorted_names, sorted_magnitudes, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Normalized Importance Score', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Feature Importance ({model_type.upper()})\nBased on Conv1d Weights', 
                      fontsize=13, fontweight='bold')
    axes[0].set_xlim(0, 105)
    for i, v in enumerate(sorted_magnitudes):
        axes[0].text(v + 1, i, f'{v:.1f}', va='center', fontweight='bold')
    axes[0].grid(alpha=0.3, axis='x')
    
    # 2) Weight Distribution by Feature
    bp = axes[1].boxplot([conv1_weights.cpu().numpy()[:, i, :].flatten() for i in range(10)],
                         labels=feature_names,
                         patch_artist=True,
                         notch=True)
    
    for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, 10))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1].set_ylabel('Weight Value', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Weight Distribution per Feature\n({model_type.upper()})', 
                      fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 저장
    output_file = f'feature_importance_{model_type.lower()}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Feature importance visualization saved to '{output_file}'")
    plt.close(fig)
    
    # ==========================================
    # 5. Summary
    # ==========================================
    print(f"\n{'='*70}")
    print("💡 SUMMARY")
    print(f"{'='*70}\n")
    
    top_feature = sorted_names[0]
    top_importance = sorted_magnitudes[0]
    
    print(f"🏆 Most important feature: {top_feature}")
    print(f"   Importance score: {top_importance:.2f}/100")
    print(f"\n📊 Top 3 features:")
    for i in range(min(3, len(sorted_names))):
        print(f"   {i+1}. {sorted_names[i]:20s} ({sorted_magnitudes[i]:6.2f})")
    
if __name__ == "__main__":
    import sys
    
    model_type = "cnn"  # 기본값
    
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
    
    print(f"\nUsage: python3 feature_importance.py [cnn|lstm]\n")
    
    analyze_feature_importance(model_type)
