import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import config
# Import your existing evaluation utilities
from evaluation_utils import (plot_confusion_matrix, plot_pca_2d_projection,
                              print_detailed_prediction_map)


# =====================================================================
# 1. Define 1D-CNN Model Architecture
# =====================================================================
class Drone1DCNN(nn.Module):
    def __init__(self, num_features=3, num_classes=2):
        super(Drone1DCNN, self).__init__()
        
        # Expected input shape: (Batch_Size, Channels/Features, Sequence_Length)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Global Average Pooling ensures output is (Batch, 32, 1) regardless of sequence length
            nn.AdaptiveAvgPool1d(1) 
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(), # Flattens to (Batch, 32)
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        features = self.conv_block(x)
        out = self.classifier(features)
        # Return both predictions and the 32D latent features for PCA visualization
        return out, features.view(features.size(0), -1) 

# =====================================================================
# 2. Helper Function for 3D Sequence Scaling
# =====================================================================
def scale_3d_sequences(X_train, X_test, X_real=None):
    """
    Flattens 3D sequences (Batch, Length, Features) for standard scaling, 
    then restores the original 3D shape.
    """
    scaler = StandardScaler()
    
    # Scale Train data
    B, L, F = X_train.shape
    X_train_flat = X_train.reshape(-1, F)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(B, L, F)
    
    # Scale Test data
    B_t, L_t, F_t = X_test.shape
    X_test_scaled = scaler.transform(X_test.reshape(-1, F)).reshape(B_t, L_t, F)
    
    # Scale Real data if provided
    X_real_scaled = None
    if X_real is not None:
        B_r, L_r, F_r = X_real.shape
        X_real_scaled = scaler.transform(X_real.reshape(-1, F)).reshape(B_r, L_r, F)
        
    return X_train_scaled, X_test_scaled, X_real_scaled

import argparse


# =====================================================================
# 3. Main Execution Pipeline
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Train 1D-CNN Model.")
    parser.add_argument("--sitl-folder", type=str, default=config.SITL_FOLDER, help="Name of the SITL folder.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--no-real", action="store_true", help="Disable evaluation on real flight data.")
    args = parser.parse_args()

    # Adjust cache directory if necessary based on your folder structure
    cache_dir = config.CACHE_DIR 
    sitl_cache = cache_dir / f"{args.sitl_folder}_features.npz"

    if not sitl_cache.exists():
        print("[Error] Cache file not found. Please run 'build_features.py' first.")
        return

    # 1. Load Padded Sequence Data
    print("\n[Info] Loading cached sequence data...")
    sitl_data = np.load(sitl_cache)
    X_sitl_seq, y_sitl = sitl_data['X_seq'], sitl_data['y'] 
    
    # 2. Load multiple real flight datasets if available
    X_real_seq_list, y_real_list, runs_real_list = [], [], []
    if not args.no_real:
        test_folders = config.REAL_FLIGHT_FOLDERS
        for folder in test_folders:
            real_cache = cache_dir / f"{folder}_features.npz"
            if real_cache.exists():
                real_data = np.load(real_cache)
                X_real_seq_list.append(real_data['X_seq'])
                y_real_list.append(real_data['y'])
                runs_real_list.append(real_data['runs'])
                print(f"  -> Loaded '{folder}' features.")

    if len(X_real_seq_list) > 0:
        # Concatenate multiple real datasets and ensure consistent sequence lengths by padding
        max_len = max(x.shape[1] for x in X_real_seq_list)
        for i in range(len(X_real_seq_list)):
            x = X_real_seq_list[i]
            if x.shape[1] < max_len:
                pad_width = max_len - x.shape[1]
                X_real_seq_list[i] = np.pad(x, ((0,0), (0, pad_width), (0,0)), mode='constant')
        
        X_real_seq = np.vstack(X_real_seq_list)
        y_real = np.concatenate(y_real_list)
        runs_real = np.concatenate(runs_real_list)
    else:
        X_real_seq, y_real, runs_real = None, None, None

    # 2. Train/Test Split & Standard Scaling
    X_train, X_test, y_train, y_test = train_test_split(X_sitl_seq, y_sitl, test_size=0.1, random_state=42)
    X_train_s, X_test_s, X_real_s = scale_3d_sequences(X_train, X_test, X_real_seq)

    # Convert Numpy arrays to PyTorch tensors and transpose shapes to (Batch, Channels, Length)
    X_train_t = torch.tensor(X_train_s.transpose(0, 2, 1), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test_s.transpose(0, 2, 1), dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Set up DataLoader for batch processing
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Model Initialization and Training Loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Info] Training 1D-CNN model on {device}...")
    
    # Initialize model with 2 classes (PX4, ArduPilot)
    model = Drone1DCNN(num_features=3, num_classes=2).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 4. Evaluate Model on SITL Test Set
    model.eval()
    with torch.no_grad():
        # Extract features for training data (for PCA visualization)
        _, train_cnn_features = model(X_train_t.to(device)) 
        # Evaluate on test data
        test_outputs, test_cnn_features = model(X_test_t.to(device))
        
        _, y_pred = torch.max(test_outputs, 1)
        y_pred = y_pred.cpu().numpy()

    target_names_sitl = ['PX4', 'ArduPilot']
    print("\n================ SITL Classification Results ================")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=target_names_sitl, zero_division=0))

    # 5. Evaluate Model on Real Flight Data
    if X_real_s is not None:
        # Filter out Cogni (Class 2) from Real data to avoid dimension mismatch
        valid_real_idx = (y_real != 2)
        X_real_s = X_real_s[valid_real_idx]
        y_real = y_real[valid_real_idx]
        runs_real = runs_real[valid_real_idx]

        X_real_t = torch.tensor(X_real_s.transpose(0, 2, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            real_outputs, real_cnn_features = model(X_real_t)
            _, y_real_pred = torch.max(real_outputs, 1)
            y_real_pred = y_real_pred.cpu().numpy()

        correct_count = np.sum(y_real_pred == y_real)
        print(f"\n[Info] Final Real Data Accuracy: {(correct_count / len(y_real)) * 100:.2f}%")
        
        print_detailed_prediction_map(y_real, y_real_pred, runs_real)
        print("\n[Real Data Classification Report]")
        print(classification_report(y_real, y_real_pred, labels=[0, 1], target_names=['PX4', 'ArduPilot'], zero_division=0))

        # Generate plots using CNN latent features for PCA
        plot_confusion_matrix(y_real, y_real_pred, target_names=['PX4', 'ArduPilot'], model_name="1D-CNN")
        plot_pca_2d_projection(
            train_cnn_features.cpu().numpy(), test_cnn_features.cpu().numpy(), 
            y_train, y_test, 
            real_cnn_features.cpu().numpy(), y_real, y_real_pred, 
            model_name="1D-CNN"
        )
    else:
        plot_confusion_matrix(y_test, y_pred, target_names=target_names_sitl, model_name="1D-CNN")

if __name__ == "__main__":
    main()