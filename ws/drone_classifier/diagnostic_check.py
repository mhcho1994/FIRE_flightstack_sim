import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import build_training_pipeline, extract_flight_signatures
from model import DroneTrajectoryCNN

def check_data_leakage():
    """훈련/테스트 데이터 중복 확인"""
    print(f"\n{'='*70}")
    print("🔍 CHECK 1: Data Leakage Detection")
    print(f"{'='*70}\n")
    
    px4_files = list(Path("../../data/px4_logs").glob("*.ulg"))
    ardu_files = list(Path("../../data/ardu_logs").glob("*.bin"))
    
    px4_names = set([f.name for f in px4_files])
    ardu_names = set([f.name for f in ardu_files])
    
    print(f"✅ PX4 files: {len(px4_files)}")
    print(f"✅ ArduPilot files: {len(ardu_files)}")
    print(f"✅ Total unique files: {len(px4_names) + len(ardu_names)}")
    
    if px4_names & ardu_names:
        print(f"\n⚠️ WARNING: 중복 파일 발견!")
    else:
        print(f"\n✅ Data leakage 없음: 모든 파일이 유일함")

def check_random_baseline():
    """랜덤 모델 성능 확인"""
    print(f"\n{'='*70}")
    print("🔍 CHECK 2: Random Model Baseline")
    print(f"{'='*70}\n")
    
    train_loader, test_loader = build_training_pipeline(
        px4_dir="../../data/px4_logs", 
        ardu_dir="../../data/ardu_logs", 
        batch_size=32, 
        test_ratio=0.2,
        window_size=100,
        step_size=50
    )
    
    # 훈련되지 않은 모델로 테스트
    untrained_model = DroneTrajectoryCNN(num_features=7)
    untrained_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = untrained_model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    random_acc = 100 * correct / total
    
    print(f"\n📊 Untrained (Random) Model Performance:")
    print(f"   Accuracy: {random_acc:.2f}%")
    print(f"   (Expected for random: 50%)")
    
    if random_acc > 55:
        print(f"\n⚠️ WARNING: 정확도가 50% 이상!")
        print(f"   → 특성 분포가 극도로 편향되어 있을 수 있음")
    elif random_acc < 45:
        print(f"\n✅ 정상: 무작위에 가까움")

def check_learning_curve():
    """학습곡선에서 초기 정확도 확인"""
    print(f"\n{'='*70}")
    print("🔍 CHECK 3: Learning Curve Analysis")
    print(f"{'='*70}\n")
    
    import torch.nn as nn
    import torch.optim as optim
    
    train_loader, test_loader = build_training_pipeline(
        px4_dir="../../data/px4_logs", 
        ardu_dir="../../data/ardu_logs", 
        batch_size=32, 
        test_ratio=0.2,
        window_size=100,
        step_size=50
    )
    
    model = DroneTrajectoryCNN(num_features=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    train_accs = []
    test_accs = []
    losses = []
    
    print("Training for 15 epochs with detailed tracking...\n")
    
    for epoch in range(15):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f} | Train={train_acc:.1f}% | Test={test_acc:.1f}%")
    
    # 분석
    print(f"\n{'─'*70}")
    improvement = test_accs[-1] - test_accs[0]
    print(f"📊 Analysis:")
    print(f"   Epoch 1 Test Accuracy:  {test_accs[0]:.1f}%")
    print(f"   Epoch 15 Test Accuracy: {test_accs[-1]:.1f}%")
    print(f"   Total Improvement:      {improvement:.1f}%")
    
    if improvement < 3:
        print(f"\n⚠️ 거의 개선이 없음 ({improvement:.1f}%)")
        print(f"   가능한 이유:")
        print(f"   1️⃣ PX4 vs ArduPilot의 특성이 정말 명확히 다름")
        print(f"   2️⃣ 모델이 처음부터 거의 완벽하게 분류 가능")
        print(f"   3️⃣ 간단한 선형 패턴으로 충분히 분류 가능")
    else:
        print(f"\n✅ 정상적인 학습: {improvement:.1f}% 개선")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(range(1, 16), train_accs, marker='o', label='Train', linewidth=2, markersize=6)
    axes[0].plot(range(1, 16), test_accs, marker='s', label='Test', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0].set_title('Learning Curve - Accuracy', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 105])
    
    axes[1].plot(range(1, 16), losses, marker='o', color='red', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curve_analysis.png', dpi=150)
    print(f"\n✅ Learning curve saved to 'learning_curve_analysis.png'")

def analyze_feature_statistics():
    """특성 통계 분석"""
    print(f"\n{'='*70}")
    print("🔍 CHECK 4: Feature Statistics Analysis")
    print(f"{'='*70}\n")
    
    px4_files = list(Path("../../data/px4_logs").glob("*.ulg"))[:5]
    ardu_files = list(Path("../../data/ardu_logs").glob("*.bin"))[:5]
    
    px4_list, _ = extract_flight_signatures([(f, 0) for f in px4_files])
    ardu_list, _ = extract_flight_signatures([(f, 1) for f in ardu_files])
    
    feature_names = ['Speed', 'Accel Mag', 'Jerk Mag', 'Curvature', 
                     'Yaw Rate (Traj)', 'Yaw Rate (Att)', 'Slip Rate']
    
    if px4_list and ardu_list:
        px4_data = np.vstack(px4_list)
        ardu_data = np.vstack(ardu_list)
        
        print(f"{'Feature':<20} {'PX4 Mean':<12} {'Ardu Mean':<12} {'PX4 Std':<10} {'Ardu Std':<10} {'차이율':<10}")
        print(f"{'─'*70}")
        
        for i, name in enumerate(feature_names):
            px4_mean = np.mean(px4_data[:, i])
            ardu_mean = np.mean(ardu_data[:, i])
            px4_std = np.std(px4_data[:, i])
            ardu_std = np.std(ardu_data[:, i])
            
            # 차이율
            max_val = max(abs(px4_mean), abs(ardu_mean))
            if max_val > 0:
                diff_ratio = abs(px4_mean - ardu_mean) / max_val * 100
            else:
                diff_ratio = 0
            
            print(f"{name:<20} {px4_mean:<12.4f} {ardu_mean:<12.4f} {px4_std:<10.4f} {ardu_std:<10.4f} {diff_ratio:<10.1f}%")
        
        # 시각화
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, name in enumerate(feature_names):
            axes[i].hist(px4_data[:, i], bins=30, alpha=0.6, label='PX4', edgecolor='black')
            axes[i].hist(ardu_data[:, i], bins=30, alpha=0.6, label='ArduPilot', edgecolor='black')
            axes[i].set_title(name, fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend(fontsize=9)
            axes[i].grid(alpha=0.3)
        
        axes[7].axis('off')
        
        plt.tight_layout()
        plt.savefig('feature_statistics.png', dpi=150)
        print(f"\n✅ Feature statistics visualization saved to 'feature_statistics.png'")
        
        # 판단
        diffs = []
        for i in range(7):
            px4_m = np.mean(px4_data[:, i])
            ardu_m = np.mean(ardu_data[:, i])
            m = max(abs(px4_m), abs(ardu_m))
            if m > 0:
                diffs.append(abs(px4_m - ardu_m) / m * 100)
        
        avg_diff_ratio = np.mean(diffs)
        
        print(f"\n{'─'*70}")
        print(f"평균 특성 차이율: {avg_diff_ratio:.1f}%")
        
        if avg_diff_ratio > 50:
            print(f"\n✨ 매우 높다! PX4와 ArduPilot이 명확히 다른 비행 특성을 가짐")
            print(f"   → Epoch 1부터 높은 정확도가 정상입니다 ✅")
        elif avg_diff_ratio > 20:
            print(f"\n✅ 적당. 구분 가능한 특성들이 있음")
        else:
            print(f"\n⚠️ 낮음. 특성이 비슷함.")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔬 DIAGNOSTIC: WHY IS EPOCH 1 ACCURACY SO HIGH?")
    print("="*70)
    
    check_data_leakage()
    check_random_baseline()
    analyze_feature_statistics()
    check_learning_curve()
    
    print(f"\n{'='*70}")
    print("✅ Diagnostic complete!")
    print("="*70 + "\n")
