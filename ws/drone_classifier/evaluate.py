import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from model import DroneTrajectoryCNN
from dataset import build_training_pipeline

def evaluate_model():
    """평가 및 confusion matrix 생성"""
    # 데이터 로드
    train_loader, test_loader = build_training_pipeline(
        px4_dir="../../data/px4_logs", 
        ardu_dir="../../data/ardu_logs", 
        batch_size=32, 
        test_ratio=0.2,
        window_size=100,
        step_size=50
    )

    # 모델 로드
    model = DroneTrajectoryCNN(num_features=7)
    model.load_state_dict(torch.load("drone_real_model.pth"))
    model.eval()

    # 테스트 데이터로 예측
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

    # 정확도
    accuracy = accuracy_score(all_targets, all_preds)
    print(f"\n{'='*60}")
    print(f"📊 TEST ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*60}\n")
    
    # 클래스 분포 확인
    print(f"Test Set Class Distribution:")
    print(f"  PX4 (0): {np.sum(all_targets == 0)} samples")
    print(f"  Ardu (1): {np.sum(all_targets == 1)} samples")
    print(f"\nPredicted Class Distribution:")
    print(f"  PX4 (0): {np.sum(all_preds == 0)} samples")
    print(f"  Ardu (1): {np.sum(all_preds == 1)} samples\n")

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    print("🔍 Confusion Matrix:")
    print(cm)
    if cm.size == 1:
        print(f"\n⚠️ WARNING: Only one class predicted! All samples classified as ONE class!")
        if np.sum(all_preds == 0) == len(all_preds):
            print("   All predictions: PX4 (0)")
        else:
            print("   All predictions: ArduPilot (1)")
    print(f"\nPX4  로 예측된 샘플 중 실제 PX4: {cm[0,0]} / {cm[0,0]+cm[0,1]}")
    print(f"Ardu로 예측된 샘플 중 실제 Ardu: {cm[1,1]} / {cm[1,0]+cm[1,1]}")

    # Classification Report
    print("\n" + "="*60)
    print("📋 Classification Report:")
    print("="*60)
    report = classification_report(all_targets, all_preds, 
                                   target_names=['PX4', 'ArduPilot'])
    print(report)

    # Confidence 분석
    print("\n" + "="*60)
    print("🎯 Prediction Confidence Analysis:")
    print("="*60)
    confidence = np.max(all_probs, axis=1)
    print(f"평균 확신도: {np.mean(confidence)*100:.2f}%")
    print(f"최소 확신도: {np.min(confidence)*100:.2f}%")
    print(f"최대 확신도: {np.max(confidence)*100:.2f}%")
    print(f"표준편차: {np.std(confidence)*100:.2f}%")
    
    # 확신도가 낮은 샘플 (실수하기 쉬운 샘플들)
    uncertain_idx = np.argsort(confidence)[:5]
    print(f"\n가장 낮은 확신도 (상위 5개):")
    for idx in uncertain_idx:
        true_label = "PX4" if all_targets[idx] == 0 else "Ardu"
        pred_label = "PX4" if all_preds[idx] == 0 else "Ardu"
        confidence_pct = confidence[idx] * 100
        print(f"  - 실제: {true_label:5s}, 예측: {pred_label:5s}, 확신도: {confidence_pct:.2f}%")

    # Confusion Matrix 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['PX4', 'ArduPilot'],
                yticklabels=['PX4', 'ArduPilot'])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("\n✅ Confusion matrix saved to 'confusion_matrix.png'")

    # 신뢰도 분포 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(confidence, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Prediction Confidence')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Prediction Confidence')
    axes[0].axvline(np.mean(confidence), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(confidence):.4f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 클래스별 신뢰도
    px4_conf = confidence[all_targets == 0]
    ardu_conf = confidence[all_targets == 1]
    axes[1].boxplot([px4_conf, ardu_conf], labels=['PX4', 'ArduPilot'])
    axes[1].set_ylabel('Confidence')
    axes[1].set_title('Confidence by True Class')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=150)
    print("✅ Confidence analysis saved to 'confidence_analysis.png'")
    
    return accuracy, cm

if __name__ == "__main__":
    evaluate_model()
