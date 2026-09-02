import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.ao.quantization as quant
from torch.ao.quantization import QuantStub, DeQuantStub

# --- Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
FEATURES = 6  # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

# --- Dataset Definition ---
class IMUDataset(Dataset):
    def __init__(self, file_list):
        self.X = []
        self.Y = []
        
        for file in file_list:
            try:
                data = np.load(file)
                self.X.append(data['X'])
                self.Y.append(data['Y'])
            except Exception as e:
                print(f"Skipping {file} due to error: {e}")
                
        if len(self.X) == 0:
            raise ValueError("No valid data found in the provided files.")
            
        self.X = np.concatenate(self.X, axis=0)
        self.Y = np.concatenate(self.Y, axis=0)
        
        # PyTorch Conv1d expects shape: (Batch, Channels, SequenceLength)
        # Currently X is (Batch, Sequence, Channels), so we transpose it
        self.X = np.transpose(self.X, (0, 2, 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y

# --- QAT-Ready Model Architecture ---
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        # Padding to keep the sequence length mathematically aligned
        padding = (kernel_size - 1) * dilation // 2 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
        
    def fuse_model(self):
        quant.fuse_modules(self, [['conv', 'bn', 'relu']], inplace=True)


class TCNVelocityEstimator(nn.Module):
    def __init__(self, in_features):
        super(TCNVelocityEstimator, self).__init__()
        self.quant = QuantStub()
        self.block1 = TCNBlock(in_features, 32, kernel_size=3, dilation=1)
        self.block2 = TCNBlock(32, 64, kernel_size=3, dilation=2)
        self.block3 = TCNBlock(64, 128, kernel_size=3, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(128, 64)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        self.block1.fuse_model()
        self.block2.fuse_model()
        self.block3.fuse_model()
        quant.fuse_modules(self, [['fc1', 'relu_fc']], inplace=True)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'results', 'processed_data')
    model_save_dir = os.path.join(base_dir, 'results', 'saved_models')
    os.makedirs(model_save_dir, exist_ok=True)
    
    all_files = glob.glob(os.path.join(data_dir, '*.npz'))
    
    train_files = [f for f in all_files if '-V' in os.path.basename(f)]
    test_files = [f for f in all_files if '-V' not in os.path.basename(f)]
    
    print(f"Strict Split: Training on {len(train_files)} files (Driver E).")
    print(f"Strict Split: Validating on {len(test_files)} unseen files (Drivers A, B, D).")
    
    train_dataset = IMUDataset(train_files)
    test_dataset = IMUDataset(test_files)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cpu")
    print(f"Executing Quantization-Aware Training on: {device}")
    
    model = TCNVelocityEstimator(in_features=FEATURES).to(device)
    
    # --- CORRECT QAT SEQUENCE ---
    model.eval() # 1. MUST be in eval mode for physical fusion
    model.fuse_model() 
    
    model.train() # 2. MUST be in train mode for QAT prep
    model.qconfig = quant.get_default_qat_qconfig('qnnpack')
    quant.prepare_qat(model, inplace=True)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    for epoch in range(EPOCHS):
        # 1. Training Phase
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 2. Validation Phase (Unseen Drivers)
        model.eval() # Freeze weights
        total_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output.squeeze(), batch_y)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(test_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Unseen Val Loss: {avg_val_loss:.4f}")
        
    # CONVERT TO INT8
    print("\nTraining finished. Converting model to INT8...")
    model.eval()
    quant.convert(model, inplace=True)
    
    save_path = os.path.join(model_save_dir, 'tiny_tcn_qat_int8.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Success! Highly compressed INT8 model saved to {save_path}")
