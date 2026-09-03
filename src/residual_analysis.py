import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') # Force headless backend
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch.ao.quantization as quant

# Ensure we can import from train_tcn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_tcn import TCNVelocityEstimator, IMUDataset, FEATURES

def calculate_acf(series, max_lag=50):
    """Custom Autocorrelation Function (ACF) to avoid statsmodels dependency."""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    acf = []
    for lag in range(max_lag + 1):
        c0 = np.sum((series[:n-lag] - mean) * (series[lag:] - mean)) / n
        acf.append(c0 / var)
    return np.array(acf)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'results', 'processed_data')
    model_path = os.path.join(base_dir, 'results', 'saved_models', 'tiny_tcn_qat_int8.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find model at {model_path}")
        return

    # 1. Load unseen test files
    all_files = glob.glob(os.path.join(data_dir, '*.npz'))
    test_files = [f for f in all_files if '-V' not in os.path.basename(f)]
    
    if not test_files:
        print("No unseen test files found!")
        return
        
    print(f"Loading {len(test_files)} unseen test files for Residual Analysis...")
    test_dataset = IMUDataset(test_files)
    
    # 2. Reconstruct the INT8 Model Architecture (Iteration 6)
    print("Loading INT8 Quantized Model...")
    model = TCNVelocityEstimator(in_features=FEATURES)
    model.eval()
    model.fuse_model()
    model.train()
    model.qconfig = quant.get_default_qat_qconfig('qnnpack')
    quant.prepare_qat(model, inplace=True)
    model.eval()
    quant.convert(model, inplace=True)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    predictions, truths = [], []
    
    # 3. Run Inference
    print("Running inference to collect residuals...")
    with torch.no_grad():
        for i in range(len(test_dataset)):
            x, y = test_dataset[i]
            x = x.unsqueeze(0)
            output = model(x)
            predictions.append(output[0, 0].item())
            truths.append(y.item())

    predictions = np.array(predictions)
    truths = np.array(truths)
    
    # CALCULATE RESIDUALS: y_true - y_pred
    residuals = truths - predictions

    # 4. Generate the 4-Panel Residual Analysis Dashboard
    print("Generating 4-panel residual plots...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Iteration 6 TCN Model: Comprehensive Residual Analysis', fontsize=18, fontweight='bold', y=0.96)

    # --- PLOT 1: Residual Distribution (Hist + KDE) ---
    ax1 = axes[0, 0]
    ax1.hist(residuals, bins=60, density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Histogram')
    kde = gaussian_kde(residuals)
    x_vals = np.linspace(min(residuals), max(residuals), 1000)
    ax1.plot(x_vals, kde(x_vals), color='crimson', lw=2.5, label='KDE')
    ax1.axvline(0, color='black', linestyle='--', lw=2, label='Zero Error')
    ax1.axvline(np.mean(residuals), color='limegreen', linestyle='-', lw=2, label=f'Mean ({np.mean(residuals):.2f})')
    ax1.set_title('1. Residual Distribution Plot', fontsize=14)
    ax1.set_xlabel('Residual Error (True - Predicted)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- PLOT 2: Autocorrelation (ACF) ---
    ax2 = axes[0, 1]
    lags = 60
    acf_vals = calculate_acf(residuals, max_lag=lags)
    ax2.bar(range(len(acf_vals)), acf_vals, width=0.4, color='darkorange')
    ax2.axhline(0, color='black', lw=1)
    
    # 95% confidence interval (~ 1.96 / sqrt(N))
    conf_int = 1.96 / np.sqrt(len(residuals))
    ax2.axhline(conf_int, color='red', linestyle='--', alpha=0.6, label='95% Confidence Interval')
    ax2.axhline(-conf_int, color='red', linestyle='--', alpha=0.6)
    
    ax2.set_title('2. Autocorrelation of Residuals (ACF)', fontsize=14)
    ax2.set_xlabel('Lag (Time Steps)', fontsize=12)
    ax2.set_ylabel('Autocorrelation', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- PLOT 3: Residuals vs. Fitted Plot ---
    ax3 = axes[1, 0]
    ax3.scatter(predictions, residuals, alpha=0.3, color='purple', s=8)
    ax3.axhline(0, color='black', linestyle='--', lw=2)
    
    # Fit a lowess/poly trendline to easily spot heteroscedasticity or curve bias
    z = np.polyfit(predictions, residuals, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(predictions), max(predictions), 100)
    ax3.plot(x_trend, p(x_trend), color='crimson', lw=2.5, label='Trend (2nd deg poly)')
    
    ax3.set_title('3. Residuals vs. Fitted Plot', fontsize=14)
    ax3.set_xlabel('Fitted Values (Predicted Speed km/h)', fontsize=12)
    ax3.set_ylabel('Residuals (True - Predicted)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- PLOT 4: Time-Series Error Overlay ---
    ax4 = axes[1, 1]
    # Plot a 500-step snippet for visual clarity
    snippet_len = min(500, len(truths))
    t_steps = np.arange(snippet_len)
    
    ax4.plot(t_steps, truths[:snippet_len], label='True Speed', color='blue', lw=2)
    ax4.plot(t_steps, predictions[:snippet_len], label='Predicted Speed', color='orange', lw=2, alpha=0.8)
    ax4.set_xlabel('Time Step (Sliding Windows)', fontsize=12)
    ax4.set_ylabel('Speed (km/h)', fontsize=12)
    
    # Overlay residuals on a secondary Y-axis
    ax4_twin = ax4.twinx()
    ax4_twin.bar(t_steps, residuals[:snippet_len], alpha=0.25, color='red', label='Residual Error Magnitude')
    ax4_twin.set_ylabel('Residual Magnitude', color='red', fontsize=12)
    ax4_twin.tick_params(axis='y', labelcolor='red')
    
    ax4.set_title(f'4. Time-Series Error Overlay (First {snippet_len} Steps)', fontsize=14)
    
    # Combine legends from both axes
    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc='upper left')
    
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = os.path.join(base_dir, 'results', 'residual_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Residual Analysis Dashboard successfully saved to: {plot_path}")

if __name__ == "__main__":
    main()
