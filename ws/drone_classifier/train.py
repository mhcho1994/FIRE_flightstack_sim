import torch
import torch.nn as nn
import torch.optim as optim
from model import DroneTrajectoryCNN, DroneTrajectoryCNNLSTM, DroneTrajectorMLP
from dataset import build_training_pipeline 
import sys
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

PX4_FOLDER = "../../data/px4_logs"
ARDU_FOLDER = "../../data/ardu_logs"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
WINDOW_SIZE = 25 #0.5 seconds at 50Hz
STEP_SIZE = 50

def main(model_type="cnn"):
    """
    Train drone trajectory classification model
    
    Args:
        model_type: 'cnn' (default) or 'cnn_lstm'
    """
    # Generate timestamp for model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    robust_stats_path = f"robust_stats_{timestamp}.json"
    
    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*70}\n")
    
    use_relative = (model_type.lower() == "mlp")
    # Camera/mocap noise range: 5–20 mm.  Applied only during MLP training to
    # bridge the SITL→real-flight domain gap via position noise augmentation.
    noise_std = 0.005 if use_relative else 0.0

    train_loader, test_loader = build_training_pipeline(
        px4_dir=PX4_FOLDER,
        ardu_dir=ARDU_FOLDER,
        batch_size=BATCH_SIZE,
        test_ratio=0.2,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        robust_stats_path=robust_stats_path,
        use_relative_features=use_relative,
        pos_noise_std=noise_std,
    )

    # Model selection
    if model_type.lower() == "cnn":
        model = DroneTrajectoryCNN(num_features=7)
        model_name = "Pure CNN"
    elif model_type.lower() == "cnn_lstm":
        model = DroneTrajectoryCNNLSTM(num_features=7, lstm_hidden_size=32, num_lstm_layers=1)
        model_name = "CNN-LSTM Hybrid"
    elif model_type.lower() == "mlp":
        model = DroneTrajectorMLP(num_features=7)
        model_name = "MLP (relative features)"
    else:
        print(f"Unknown model type: {model_type}")
        print("Available options: 'cnn', 'cnn_lstm', 'mlp'")
        return
    
    print(f"Model: {model_name}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # ==========================================
    # Compute class weights for balanced learning
    # ==========================================
    print("📊 Computing class weights for balanced learning...\n")
    
    all_labels = []
    for batch_x, batch_y in train_loader:
        all_labels.extend(batch_y.cpu().numpy().tolist())
    all_labels = np.array(all_labels)
    
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f"Class distribution in training data:")
    for label, count in zip(unique, counts):
        class_name = "PX4" if label == 0 else "ArduPilot"
        pct = 100 * count / len(all_labels)
        print(f"  {class_name}: {count:5d} ({pct:6.2f}%)")
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.FloatTensor(class_weights)
    
    print(f"\nComputed class weights:")
    print(f"  PX4 weight:       {class_weights[0]:.4f}")
    print(f"  ArduPilot weight: {class_weights[1]:.4f}")
    print(f"  → ArduPilot weighted {class_weights[1]/class_weights[0]:.2f}x higher\n")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    best_test_acc = 0
    patience = 10  # Early stopping patience
    patience_counter = 0

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
        
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                
        test_acc = 100 * test_correct / test_total
        
        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"Loss: {avg_loss:.4f} | "
              f"Train Accuracy: {train_acc:.1f}% | "
              f"🏆 Test Accuracy: {test_acc:.1f}%")
        
        # Early stopping check
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            # Save model with type-specific filename and timestamp
            model_filename = f"drone_{model_type}_{timestamp}.pth"
            torch.save(model.state_dict(), model_filename)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    model_filename = f"drone_{model_type}_{timestamp}.pth"
    print(f"\n Training Complete - Best Test Accuracy: {best_test_acc:.1f}%")
    print(f"Weight saved to '{model_filename}'")
    print(f"Robust scaler stats saved to '{robust_stats_path}'")

if __name__ == "__main__":
    # Default: Pure CNN
    # Usage: python train.py [cnn|cnn_lstm]
    model_type = "cnn"
    
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    
    main(model_type)
