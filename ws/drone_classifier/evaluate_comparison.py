import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from model import DroneTrajectoryCNN, DroneTrajectoryCNNLSTM
from dataset import build_training_pipeline

def evaluate_model(model, test_loader, model_name):
    """
    Evaluate a single model on test set
    
    Returns:
        dict: Evaluation metrics for the model
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    
    # Prediction confidence
    confidence = np.max(all_probs, axis=1)
    mean_confidence = np.mean(confidence)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'confidence': confidence,
        'mean_confidence': mean_confidence,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }

def main():
    """Compare all three models"""
    print(f"\n{'='*80}")
    print(f"🤖 DRONE FLIGHT CLASSIFIER - MODEL COMPARISON EVALUATION")
    print(f"{'='*80}\n")
    
    # Load data (same for all models)
    print("⏳ Loading test data...")
    train_loader, test_loader = build_training_pipeline(
        px4_dir="../../data/px4_logs", 
        ardu_dir="../../data/ardu_logs", 
        batch_size=32, 
        test_ratio=0.2,
        window_size=100,
        step_size=50
    )
    
    # Model definitions
    models_to_eval = [
        ("Pure CNN", DroneTrajectoryCNN(num_features=10), "drone_cnn_20260421_2153.pth"),
        ("CNN-LSTM", DroneTrajectoryCNNLSTM(num_features=10), "drone_cnn_lstm_20260421_2159.pth"),
        # ("CNN-LSTM V2", DroneTrajectoryCNNLSTMV2(num_features=7), "drone_cnn_lstm_v2.pth"),
    ]
    
    results = []
    
    # Evaluate each model
    for model_name, model, model_path in models_to_eval:
        print(f"\n📊 Evaluating {model_name}...")
        
        # Load weights
        try:
            model.load_state_dict(torch.load(model_path))
        except FileNotFoundError:
            print(f"  ⚠️  WARNING: {model_path} not found, skipping...")
            continue
        
        # Get total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, model_name)
        metrics['parameters'] = total_params
        results.append(metrics)
        
        print(f"  ✅ {model_name}:")
        print(f"     - Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"     - Precision: {metrics['precision']*100:.2f}%")
        print(f"     - Recall:    {metrics['recall']*100:.2f}%")
        print(f"     - F1-Score:  {metrics['f1']*100:.2f}%")
        print(f"     - Parameters: {total_params:,}")
    
    # ===== COMPARISON TABLE =====
    print(f"\n\n{'='*80}")
    print(f"📋 COMPREHENSIVE COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    comparison_data = []
    for res in results:
        comparison_data.append({
            'Model': res['model_name'],
            'Accuracy': f"{res['accuracy']*100:.2f}%",
            'Precision': f"{res['precision']*100:.2f}%",
            'Recall': f"{res['recall']*100:.2f}%",
            'F1-Score': f"{res['f1']*100:.2f}%",
            'Parameters': f"{res['parameters']:,}",
            'Confidence': f"{res['mean_confidence']*100:.2f}%",
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # ===== DETAILED REPORT FOR EACH MODEL =====
    print(f"\n\n{'='*80}")
    print(f"📝 DETAILED CLASSIFICATION REPORTS")
    print(f"{'='*80}\n")
    
    for res in results:
        print(f"\n{'='*80}")
        print(f"🔍 {res['model_name']}")
        print(f"{'='*80}")
        
        # Confusion Matrix
        cm = res['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  TN (Correct PX4):     {cm[0,0]}")
        print(f"  FP (PX4→Ardu):        {cm[0,1]}")
        print(f"  FN (Ardu→PX4):        {cm[1,0]}")
        print(f"  TP (Correct Ardu):    {cm[1,1]}")
        
        # Classification Report
        print(f"\nClassification Report:")
        report = classification_report(res['targets'], res['predictions'],
                                      target_names=['PX4', 'ArduPilot'],
                                      digits=4)
        print(report)
        
        # Confidence Analysis
        confidence = res['confidence']
        print(f"\nPrediction Confidence Analysis:")
        print(f"  Average Confidence: {np.mean(confidence)*100:.2f}%")
        print(f"  Min Confidence:     {np.min(confidence)*100:.2f}%")
        print(f"  Max Confidence:     {np.max(confidence)*100:.2f}%")
        print(f"  Std Deviation:      {np.std(confidence)*100:.2f}%")
        
        # Uncertain predictions (low confidence)
        uncertain_idx = np.argsort(confidence)[:3]
        if len(uncertain_idx) > 0:
            print(f"\n  Most uncertain predictions (lowest 3):")
            for idx in uncertain_idx:
                true_label = "PX4" if res['targets'][idx] == 0 else "Ardu"
                pred_label = "PX4" if res['predictions'][idx] == 0 else "Ardu"
                conf = confidence[idx] * 100
                status = "✅" if res['targets'][idx] == res['predictions'][idx] else "❌"
                print(f"    {status} True: {true_label:5s} | Pred: {pred_label:5s} | Conf: {conf:.2f}%")
    
    # ===== VISUALIZATION =====
    print(f"\n\n{'='*80}")
    print(f"📈 Generating visualizations...")
    print(f"{'='*80}\n")
    
    # 1. Metrics Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    model_names = [r['model_name'] for r in results]
    accuracies = [r['accuracy']*100 for r in results]
    precisions = [r['precision']*100 for r in results]
    recalls = [r['recall']*100 for r in results]
    f1s = [r['f1']*100 for r in results]
    
    # ensure color list matches number of models
    default_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    if len(results) <= len(default_colors):
        colors = default_colors[:len(results)]
    else:
        cmap = plt.get_cmap('tab20')
        colors = default_colors[:]
        for i in range(len(default_colors), len(results)):
            colors.append(cmap(i % 20))
    
    # Accuracy
    axes[0, 0].bar(model_names, accuracies, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0, 0].set_title('Test Accuracy Comparison')
    axes[0, 0].set_ylim([0, 105])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Precision
    axes[0, 1].bar(model_names, precisions, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Precision (%)', fontweight='bold')
    axes[0, 1].set_title('Precision Comparison')
    axes[0, 1].set_ylim([0, 105])
    for i, v in enumerate(precisions):
        axes[0, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Recall
    axes[1, 0].bar(model_names, recalls, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Recall (%)', fontweight='bold')
    axes[1, 0].set_title('Recall Comparison')
    axes[1, 0].set_ylim([0, 105])
    for i, v in enumerate(recalls):
        axes[1, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # F1-Score
    axes[1, 1].bar(model_names, f1s, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('F1-Score (%)', fontweight='bold')
    axes[1, 1].set_title('F1-Score Comparison')
    axes[1, 1].set_ylim([0, 105])
    for i, v in enumerate(f1s):
        axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_metrics_comparison.png")
    plt.close()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
    
    for idx, res in enumerate(results):
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['PX4', 'Ardu'],
                   yticklabels=['PX4', 'Ardu'],
                   cbar=False)
        axes[idx].set_title(res['model_name'], fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: confusion_matrices_comparison.png")
    plt.close()
    
    # 3. Model Complexity vs Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    
    params = [r['parameters'] for r in results]
    accs = [r['accuracy']*100 for r in results]
    
    scatter = ax.scatter(params, accs, s=500, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    
    for i, model_name in enumerate(model_names):
        ax.annotate(model_name, (params[i], accs[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))
    
    ax.set_xlabel('Total Parameters', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Model Complexity vs Performance', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([98, 101])
    
    plt.tight_layout()
    plt.savefig('model_complexity_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: model_complexity_comparison.png")
    plt.close()
    
    # 4. Confidence Distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    
    for idx, res in enumerate(results):
        confidence = res['confidence']
        axes[idx].hist(confidence, bins=20, color=colors[idx], alpha=0.7, edgecolor='black')
        axes[idx].set_title(res['model_name'], fontweight='bold')
        axes[idx].set_xlabel('Confidence Score')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(np.mean(confidence), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidence):.3f}')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('confidence_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: confidence_distribution_comparison.png")
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"✅ EVALUATION COMPLETE!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
