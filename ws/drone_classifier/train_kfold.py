import torch
import torch.nn as nn
import torch.optim as optim
from model import DroneTrajectoryCNN
from dataset import extract_flight_signatures, DroneTrajectoryDataset
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import random

PX4_FOLDER = "../../data/px4_logs"
ARDU_FOLDER = "../../data/ardu_logs"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
WINDOW_SIZE = 100
STEP_SIZE = 50
K_FOLDS = 5

def kfold_cross_validation():
    """K-Fold Cross-Validation으로 모델 훈련"""
    
    # 전체 파일 로드
    px4_files = list(Path(PX4_FOLDER).glob("*.ulg"))
    ardu_files = list(Path(ARDU_FOLDER).glob("*.bin"))
    
    all_files = [(f, 0) for f in px4_files] + [(f, 1) for f in ardu_files]
    random.shuffle(all_files)
    
    print(f"\n{'='*70}")
    print(f"🔄 K-Fold Cross-Validation (K={K_FOLDS})")
    print(f"{'='*70}")
    print(f"Total files: {len(all_files)} (PX4: {len(px4_files)}, Ardu: {len(ardu_files)})")
    print(f"{'='*70}\n")
    
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_files)):
        print(f"\n{'─'*70}")
        print(f"📍 FOLD {fold + 1}/{K_FOLDS}")
        print(f"{'─'*70}")
        
        # 데이터 분할
        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        
        print(f"Training samples: {len(train_files)} | Validation samples: {len(val_files)}")
        
        # 특성 추출
        print("⏳ Extracting training features...")
        train_data, train_labels = extract_flight_signatures(train_files)
        
        print("⏳ Extracting validation features...")
        val_data, val_labels = extract_flight_signatures(val_files)
        
        # 데이터셋 생성
        train_dataset = DroneTrajectoryDataset(train_data, train_labels, WINDOW_SIZE, STEP_SIZE)
        val_dataset = DroneTrajectoryDataset(val_data, val_labels, WINDOW_SIZE, STEP_SIZE)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"Training tensors: {len(train_dataset)} | Validation tensors: {len(val_dataset)}\n")
        
        # 모델 초기화
        model = DroneTrajectoryCNN(num_features=7)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        
        best_val_acc = 0
        patience = 8
        patience_counter = 0
        fold_best_loss = float('inf')
        
        # 훈련
        for epoch in range(NUM_EPOCHS):
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
            
            # 검증
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = 100 * val_correct / val_total
            
            if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
                print(f"  Epoch [{epoch+1:2d}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | "
                      f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  ⚠️ Early stopping at epoch {epoch+1}")
                    break
        
        fold_accuracies.append(best_val_acc)
        fold_losses.append(avg_loss)
        
        print(f"\n✅ Fold {fold + 1} Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # 결과 정리
    print(f"\n{'='*70}")
    print(f"📊 CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"\nFold Accuracies:")
    for i, acc in enumerate(fold_accuracies):
        print(f"  Fold {i+1}: {acc:.2f}%")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"\n{'─'*70}")
    print(f"🎯 Average Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"{'─'*70}\n")
    
    # 전체 데이터로 최종 모델 훈련
    print("\n" + "="*70)
    print("🔄 Training Final Model on All Data...")
    print("="*70)
    
    all_data, all_labels = extract_flight_signatures(all_files)
    final_dataset = DroneTrajectoryDataset(all_data, all_labels, WINDOW_SIZE, STEP_SIZE)
    final_loader = DataLoader(final_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    final_model = DroneTrajectoryCNN(num_features=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    for epoch in range(NUM_EPOCHS):
        final_model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for batch_x, batch_y in final_loader:
            optimizer.zero_grad()
            outputs = final_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(final_loader)
        
        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch [{epoch+1:2d}/{NUM_EPOCHS}] Loss: {avg_loss:.4f} | Accuracy: {train_acc:.1f}%")
    
    torch.save(final_model.state_dict(), "drone_real_model.pth")
    print("\n✅ Final model saved to 'drone_real_model.pth'")

if __name__ == "__main__":
    kfold_cross_validation()
