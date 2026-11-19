import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import random
from signal_processor import process_px4_ulog, process_ardu_bin

class DroneTrajectoryDataset(Dataset):
    def __init__(self, data_list, labels_list, window_size, step_size):
        self.windows = []
        self.labels = []
        
        for data, label in zip(data_list, labels_list):
            num_steps = len(data) 
            
            for i in range(0, num_steps - window_size + 1, step_size):
                window = data[i : i + window_size] # (100, 7)
                
                self.windows.append(window.T)
                self.labels.append(label)
                
        self.windows = torch.tensor(np.array(self.windows), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

def extract_flight_signatures(file_list):
    data_list = []
    labels_list = []
    
    for file_path, label in file_list:
        if label == 0:
            result = process_px4_ulog(str(file_path))
        else:
            result = process_ardu_bin(str(file_path))
            
        if result is not None:
            features = result[0] 
            
            if len(features) > 0: 
                data_list.append(features)
                labels_list.append(label)
            
    return data_list, labels_list

def build_training_pipeline(px4_dir, ardu_dir, batch_size, test_ratio, window_size, step_size):
    px4_files = list(Path(px4_dir).glob("*.ulg"))
    ardu_files = list(Path(ardu_dir).glob("*.bin"))
    
    if not px4_files and not ardu_files:
        raise ValueError("No log files found in the specified directories.")
        
    random.shuffle(px4_files)
    random.shuffle(ardu_files)
    
    px4_test_cnt = max(1, int(len(px4_files) * test_ratio)) if len(px4_files) > 1 else 0
    ardu_test_cnt = max(1, int(len(ardu_files) * test_ratio)) if len(ardu_files) > 1 else 0
    
    train_files = [(f, 0) for f in px4_files[px4_test_cnt:]] + [(f, 1) for f in ardu_files[ardu_test_cnt:]]
    test_files = [(f, 0) for f in px4_files[:px4_test_cnt]] + [(f, 1) for f in ardu_files[:ardu_test_cnt]]
    
    print(f"Files divided - Training: {len(train_files)} | Test: {len(test_files)}")
    
    print("⏳ Loading Train Data...")
    train_data, train_labels = extract_flight_signatures(train_files)
    
    print("⏳ Loading Test Data...")
    test_data, test_labels = extract_flight_signatures(test_files)
    
    train_dataset = DroneTrajectoryDataset(train_data, train_labels, window_size, step_size)
    test_dataset = DroneTrajectoryDataset(test_data, test_labels, window_size, step_size)
    
    print(f"✅ Tensor Loading Complete - Training Tensors: {len(train_dataset)} | Test Tensors: {len(test_dataset)}\n")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader