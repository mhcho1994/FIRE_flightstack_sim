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
# Import common project evaluation utilities
from evaluation_utils import (plot_confusion_matrix, plot_pca_2d_projection,
                              print_detailed_prediction_map)


# =====================================================================
# 1. Define CNN-LSTM Hybrid Model Architecture
# =====================================================================
class DroneCNNLSTM(nn.Module):
    def __init__(self, num_features=3, num_classes=2, lstm_hidden_size=64, num_lstm_layers=1):
        super(DroneCNNLSTM, self).__init__()
        
        # Stage 1: 1D-CNN Feature Extraction Block
        # Extracts local kinematic patterns and squashes features
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Reduces sequence length by half
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
            # AdaptiveAvgPool1d is omitted to preserve the temporal dimension for the LSTM stage
        )
        
        # Stage 2: LSTM Temporal Sequence Learning Block
        # Expects input feature size of 32 from the final Conv1d layer output channels
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=lstm_hidden_size, 
            num_layers=num_lstm_layers, 
            batch_first=True
        )
        
        # Stage 3: Dense Fully-Connected Classifier Block
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden_size, num_classes)
        )

    def forward(self, x):
        # Input x shape: (Batch, Channels/Features, Sequence_Length)
        conv_out = self.conv_block(x) # Output shape: (Batch, 32, New_Sequence_Length)
        
        # Permute shape to (Batch, New_Sequence_Length, 32) to match LSTM expectations
        lstm_in = conv_out.permute(0, 2, 1)
        
        # out shape: (Batch, New_Sequence_Length, lstm_hidden_size)
        # h_n shape: (num_layers, Batch, lstm_hidden_size)
        out, (h_n, c_n) = self.lstm(lstm_in)
        
        # Extract the final hidden state from the last layer of LSTM as latent representation
        latent_features = h_n[-1] # Shape: (Batch, lstm_hidden_size)
        
        logits = self.classifier(latent_features)
        return logits, latent_features

# =====================================================================
# 2. Helper Function for 3D Sequence Scaling (Unchanged from CNN)
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
# 3. Main Training & Evaluation Pipeline
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM Hybrid Model.")
    parser.add_argument("--sitl-folder", type=str, default=config.SITL_FOLDER, help="Name of the SITL folder.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--no-real", action="store_true", help="Disable evaluation on real flight data.")
    args = parser.parse_args()

    cache_dir = config.CACHE_DIR 
    sitl_cache = cache_dir / f"{args.sitl_folder}_features.npz"

    if not sitl_cache.exists():
        print("[Error] Feature cache not found. Please execute 'build_features.py' first.")
        return

    # 1. Load Padded Sequence Datasets
    print("\n[Info] Loading cached sequence dataset for CNN-LSTM Hybrid model...")
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

    # 2. Train/Test Data Split & Step-by-Step 3D Scaling
    X_train, X_test, y_train, y_test = train_test_split(X_sitl_seq, y_sitl, test_size=0.1, random_state=42)
    X_train_s, X_test_s, X_real_s = scale_3d_sequences(X_train, X_test, X_real_seq)

    # Convert Numpy arrays into PyTorch Tensors and transpose to (Batch, Channels, Length)
    X_train_t = torch.tensor(X_train_s.transpose(0, 2, 1), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test_s.transpose(0, 2, 1), dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Establish PyTorch DataLoader for optimized batching
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Model Initialization & Supervised Learning Loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Info] Initializing and training CNN-LSTM model on environment: {device}...")
    
    # Target binary classification (0: PX4, 1: ArduPilot)
    model = DroneCNNLSTM(num_features=3, num_classes=2, lstm_hidden_size=64).to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # 4. Model Evaluation on SITL Simulated Test Sets
    model.eval()
    with torch.no_grad():
        # Capture internal representations for PCA projection mapping
        _, train_latent_features = model(X_train_t.to(device)) 
        test_outputs, test_latent_features = model(X_test_t.to(device))
        
        _, y_pred = torch.max(test_outputs, 1)
        y_pred = y_pred.cpu().numpy()

    target_names_sitl = ['PX4', 'ArduPilot']
    print("\n================ SITL CNN-LSTM Classification Results ================")
    print(f"Accuracy on Simulated Test Set: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=target_names_sitl, zero_division=0))

    # 5. Model Generalization Validation on Real World Flight Data
    if X_real_s is not None:
        # Filter out anomalous Cogni labels (Class 2) to maintain binary metrics integrity
        valid_real_idx = (y_real != 2)
        X_real_s = X_real_s[valid_real_idx]
        y_real = y_real[valid_real_idx]
        runs_real = runs_real[valid_real_idx]

        X_real_t = torch.tensor(X_real_s.transpose(0, 2, 1), dtype=torch.float32).to(device)
        with torch.no_grad():
            real_outputs, real_latent_features = model(X_real_t)
            _, y_real_pred = torch.max(real_outputs, 1)
            y_real_pred = y_real_pred.cpu().numpy()

        correct_count = np.sum(y_real_pred == y_real)
        print(f"\n[Info] Final Real World Generalization Accuracy: {(correct_count / len(y_real)) * 100:.2f}%")
        
        print_detailed_prediction_map(y_real, y_real_pred, runs_real)
        print("\n[Real Data CNN-LSTM Classification Report]")
        print(classification_report(y_real, y_real_pred, labels=[0, 1], target_names=['PX4', 'ArduPilot'], zero_division=0))

        # Render visualizations using hidden features discovered by the LSTM block
        plot_confusion_matrix(y_real, y_real_pred, target_names=['PX4', 'ArduPilot'], model_name="CNN-LSTM")
        plot_pca_2d_projection(
            train_latent_features.cpu().numpy(), test_latent_features.cpu().numpy(), 
            y_train, y_test, 
            real_latent_features.cpu().numpy(), y_real, y_real_pred, 
            model_name="CNN-LSTM"
        )
    else:
        plot_confusion_matrix(y_test, y_pred, target_names=target_names_sitl, model_name="CNN-LSTM")

if __name__ == "__main__":
    main()