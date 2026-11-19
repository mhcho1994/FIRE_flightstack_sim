import torch
import torch.nn as nn

class DroneTrajectoryCNN(nn.Module):
    """Pure CNN model for drone flight classification"""
    def __init__(self, num_features):
        super().__init__()
        
        # Enlarged kernel size for noise insensitivity
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=8, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(8)
        self.relu1 = nn.ReLU() 
        self.dropout1 = nn.Dropout(0.5)
        # Increased pooling size to blur/smooth details
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Larger kernel for second conv layer
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.flatten = nn.Flatten()
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(in_features=16, out_features=8)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.4)
        
        # Final output: PX4 (0) vs ArduPilot (1)
        self.fc2 = nn.Linear(in_features=8, out_features=2)

    def forward(self, x):
        # x shape: (batch_size, num_features, sequence_length)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        out = self.fc2(x)
        return out

class DroneTrajectoryCNNLSTM(nn.Module):
    """
    CNN-LSTM Hybrid model for trajectory classification (Tuned for Real-world Odom)
    """
    # 💡 [수정 1] LSTM 다이어트: hidden_size 64->32, num_layers 2->1 로 롤백
    def __init__(self, num_features, lstm_hidden_size=32, num_lstm_layers=1):
        super().__init__()
        
        # ===== CNN Part =====
        # 💡 [수정 2] Conv1 Kernel 크기 확대 (5->7): 노이즈를 더 넓게 보고 무시하도록 변경
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        # 1차 풀링 유지 (데이터를 살짝 뭉개서 노이즈 감소)
        self.pool1 = nn.MaxPool1d(kernel_size=2) 
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # 💡 [수정 3] pool2 완전히 삭제! 
        # 이유: 여기서 또 풀링을 하면 시퀀스 길이가 너무 짧아져서 LSTM이 '흐름'을 읽지 못합니다.
        # self.pool2 = nn.MaxPool1d(kernel_size=2) <- 삭제됨
        
        # ===== LSTM Part =====
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden_size,  # 32로 축소 (과적합 방지)
            num_layers=num_lstm_layers,    # 1로 축소 (과적합 방지)
            batch_first=True,
            # num_layers가 1일 때는 PyTorch 내부 규정상 dropout을 0으로 설정해야 에러가 안 납니다.
            dropout=0.0, 
            bidirectional=False
        )
        
        # ===== FC Part =====
        # LSTM이 가벼워졌으므로 FC 레이어의 노드 수도 줄이고, Dropout을 강하게 줍니다.
        self.fc1 = nn.Linear(in_features=lstm_hidden_size, out_features=16)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.4)  # 💡 [수정 4] 0.2 -> 0.4 로 올려서 과확신(99% 오답) 방지
        
        self.fc2 = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        # x shape: (batch_size, num_features, sequence_length)
        
        # ===== CNN part =====
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.pool1(x)  # 시퀀스 길이 절반으로 감소 (적당한 뭉개짐)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        # 두 번째 풀링이 사라졌으므로, 시퀀스 길이는 절반 크기로 잘 보존됨
        
        # ===== Reshape for LSTM =====
        # CNN output: (batch_size, 32, sequence_length // 2)
        # LSTM input: (batch_size, sequence_length // 2, 32)
        x = x.transpose(1, 2)
        
        # ===== LSTM part =====
        lstm_out, (h_n, c_n) = self.lstm(x)

        x = lstm_out.mean(dim=1)
        
        # ===== FC part =====
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        out = self.fc2(x)
        return out
