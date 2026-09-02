import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') # Force headless backend to bypass broken Tkinter on Windows
import matplotlib.pyplot as plt
import torch.ao.quantization as quant

# Ensure we can import from train_tcn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_tcn import TCNVelocityEstimator, IMUDataset, FEATURES

def evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'results', 'processed_data')
    model_path = os.path.join(base_dir, 'results', 'saved_models', 'tiny_tcn_qat_int8.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return

    # 1. Load the exact same unseen files (Drivers A, B, D)
    all_files = glob.glob(os.path.join(data_dir, '*.npz'))
    test_files = [f for f in all_files if '-V' not in os.path.basename(f)]
    
    if not test_files:
        print("No unseen test files found!")
        return
        
    print(f"Loading {len(test_files)} unseen driver files for evaluation...")
    test_dataset = IMUDataset(test_files)
    
    # 2. Reconstruct the INT8 Model Architecture
    print("Loading INT8 Quantized Model...")
    model = TCNVelocityEstimator(in_features=FEATURES)
    
    # PyTorch QAT strict sequence for reconstruction:
    model.eval() # 1. MUST be in eval mode for physical fusion
    model.fuse_model()
    
    model.train() # 2. MUST be in train mode for prepare_qat
    model.qconfig = quant.get_default_qat_qconfig('qnnpack')
    quant.prepare_qat(model, inplace=True)
    
    model.eval() # 3. MUST be in eval mode for convert
    quant.convert(model, inplace=True) # Snap architecture to INT8
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    
    predictions = []
    truths = []
    variances = []
    
    # 3. Run Inference on Unseen Data
    print("Running AI inference on unseen engine vibrations...")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            x, y = test_dataset[i]
            x = x.unsqueeze(0) # Add batch dimension: (1, Channels, Sequence)
            
            output = model(x)
            
            pred_speed = output[0, 0].item()
            
            predictions.append(pred_speed)
            truths.append(y.item())

    predictions = np.array(predictions)
    truths = np.array(truths)
    
    # 4. Calculate Mathematical Metrics
    rmse = np.sqrt(np.mean((predictions - truths)**2))
    mae = np.mean(np.abs(predictions - truths))
    
    print("\n" + "="*40)
    print("🏆 UNSEEN DRIVER EVALUATION METRICS (ITERATION 2) 🏆")
    print("="*40)
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print("="*40)
    
    # 5. Generate Visual Proof for ISRO
    print("Generating performance graph...")
    # Plot a small snippet (e.g., 300 sequence windows) so the graph is readable
    snippet_len = min(300, len(truths))
    
    plt.figure(figsize=(14, 6))
    plt.plot(truths[:snippet_len], label='True Speed (Vehicle Ground Truth)', color='blue', linewidth=2)
    plt.plot(predictions[:snippet_len], label='AI Predicted Speed (From Phone IMU)', color='red', linestyle='dashed', linewidth=2)
                     
    plt.title('AI Evaluation (MSE Loss): True Speed vs AI Predicted Speed')
    plt.xlabel('Time Steps (Sliding Windows)')
    plt.ylabel('Velocity (Normalized / Standard Units)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(base_dir, 'results', 'speed_evaluation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Graph successfully saved to: {plot_path}")

if __name__ == "__main__":
    evaluate()
