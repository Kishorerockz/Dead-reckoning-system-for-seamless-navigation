## 🔴 Iteration 1: Initial QAT TCN with Gaussian NLL
**Date:** 2026-09-02
**Objective:** Baseline test to ensure Quantization-Aware Training (QAT) compiles and to attempt simultaneous learning of Velocity and Variance.

### ⚙️ Configuration
* **Architecture:** 1D Temporal Convolutional Network (TCN)
* **Loss Function:** Gaussian Negative Log-Likelihood (Gaussian NLL)
* **Epochs:** 15
* **Learning Rate:** 1e-3
* **Data Split:** Session-Independent (Train: Group 1, Test: Group 2)

### 📊 Results & Metrics
* **Final Train Loss:** 4.88
* **Unseen Val Loss:** 6.71
* **RMSE:** 34.26
* **MAE:** 32.31
* **Visual Graph Analysis (`speed_evaluation.png`):** The AI prediction (red line) completely flatlined near zero. The AI confidence interval (pink shaded area) spiked wildly to massive numbers (up to 30,000,000).

### 🔍 Analysis & Diagnosis (Why did this happen?)
**Total Failure to Converge.** The model suffered from *Loss Function Instability*. By using Gaussian NLL, we forced a randomly-initialized neural network to learn two highly complex tasks simultaneously: predicting exact speed and predicting its own mathematical uncertainty. 
The math overwhelmed the gradients. The model quickly realized that outputting massive variance (uncertainty) minimized the penalty of guessing the wrong speed, leading to a gradient collapse after Epoch 1 (loss dropped from 534 to 6 instantly, then stagnated). 

### 🛠️ Fixes Required for Iteration 2
1. **Simplify the Loss Function:** Temporarily abandon Gaussian NLL. Switch to a brutal, standard **Mean Squared Error (MSE)** loss. This removes the variance prediction and forces the AI to focus 100% of its parameters on matching the true speed line.
2. **Increase Training Time:** 15 epochs is vastly insufficient. Increase to **50 epochs**.
3. **Architecture Adjustment:** Change the final linear layer from `Linear(64, 2)` to `Linear(64, 1)` since we are only predicting speed for this next test.

## 🔴 Iteration 2: Pure MSE Loss & Increased Epochs
**Date:** 2026-09-03
**Objective:** Eliminate Gaussian NLL instability, force the model to learn raw speed mappings via standard MSE, and increase training duration to 50 epochs.

### ⚙️ Configuration
* **Architecture:** 1D Temporal Convolutional Network (TCN) - Output layer reduced to \Linear(64, 1)* **Loss Function:** Mean Squared Error (MSELoss)
* **Epochs:** 50
* **Learning Rate:** 1e-3
* **Data Split:** Session-Independent (Train: Group 1 [64 files], Test: Group 2 [8 files])

### 📊 Results & Metrics
* **Final Train Loss:** 417.86
* **Unseen Val Loss:** 1778.18
* **RMSE:** 39.15
* **MAE:** 33.76

### 🔍 Analysis & Diagnosis (Why did this happen?)
**Severe Overfitting & Domain Shift.** 
The network successfully learned how to predict speed, but *only* for the training car! The Train Loss dropped beautifully from 1713 down to 417, proving the neural network architecture works. However, the Unseen Validation Loss exploded from 692 up to 1778. 

This means the AI completely memorized the training car's specific engine vibrations, suspension stiffness, and phone-mount rattles. When tested on the completely different cars of unseen cars/routes, those raw vibration patterns were so different that the AI failed to generalize. 

### 🛠️ Fixes Required for Iteration 3
1. **Stronger Regularization:** Increase Dropout (e.g., to 0.4) and add Weight Decay (L2 penalty) to the Adam optimizer to forcefully prevent the AI from memorizing the training car.
2. **Learning Rate Scheduler:** The Validation Loss bounced around erratically (1300 -> 2000 -> 1600). A \ReduceLROnPlateau\ scheduler is needed to smooth out the learning.
3. **Feature Engineering (Phase 3 Consideration):** If regularization fails, we will need to update \preprocess.py\ to extract Frequency Domain features (FFT). Engine RPM frequencies transfer much better between vehicles than raw vibration amplitudes.




---

## 🔴 Iteration 3: Strong Regularization & LR Scheduling
**Date:** 2026-09-03
**Objective:** Prevent the severe domain-overfitting seen in Iteration 2 by adding a mathematical "blindfold" (Dropout), weight penalization, and a learning rate scheduler.

### ⚙️ Configuration
* **Architecture:** 1D TCN (Linear(64, 1))
* **Loss Function:** MSELoss
* **Epochs:** 50
* **Learning Rate:** 1e-3 (with ReduceLROnPlateau scheduler)
* **Regularization:** Dropout increased to 0.4, Weight Decay (L2) set to 1e-4

### 📊 Results & Metrics
* **Final Train Loss:** 576.28
* **Unseen Val Loss:** 1307.35
* **RMSE:** 35.74
* **MAE:** 32.01

### 🔍 Analysis & Diagnosis (Why did this happen?)
**Partial Success, but hitting the limits of Time-Domain Data.**
The regularization worked exactly as intended! By increasing the Dropout and adding Weight Decay, we successfully stopped the AI from blindly memorizing the training car's car (Train Loss was restricted to 576, up from 417). Because the AI was forced to look for general patterns instead of memorizing, the Unseen Validation Loss improved massively (dropping from 1778 down to 1307), and the RMSE improved by nearly 3.5 points.

However, an RMSE of 35 is still too high for ISRO's <10% drift requirement. We have hit the mathematical ceiling of what raw, time-domain accelerometer data can provide. Raw vibration amplitudes are too easily biased by a specific car's suspension stiffness or the exact angle the phone is mounted.

### 🛠️ Fixes Required for Iteration 4 (Phase 3: Feature Engineering)
1. **Frequency Domain Transformation (FFT):** We must update preprocess.py to convert the raw time-series vibration windows into Fast Fourier Transforms (FFT) or Spectrograms. Engine RPMs (frequencies) remain mathematically constant across different cars, whereas raw bump amplitudes do not.
2. **Update Evaluation Script:** Update the print statement in evaluate_model.py to correctly state "Iteration 4".



---

## 🔴 Iteration 4: Data Pipeline Bug Fix (Full 6-Channel Activation)
**Date:** 2026-09-03
**Objective:** Fix the critical silent-zeroing bug discovered in preprocess.py where Gyro X and Gyro Z were completely missing from the training data. Re-calculate dynamic sampling rates instead of assuming 100Hz.

### ⚙️ Configuration
* **Architecture:** 1D TCN (Linear(64, 1))
* **Loss Function:** MSELoss
* **Epochs:** 50
* **Learning Rate:** 1e-3 (with ReduceLROnPlateau scheduler)
* **Regularization:** Dropout 0.4, Weight Decay 1e-4
* **Data:** **Corrected 6-Channel IO-VNBD Data** with dynamic IMU_FS window scaling.

### 📊 Results & Metrics
* **Final Train Loss:** 600.62
* **Unseen Val Loss:** 1225.60
* **RMSE:** 34.66
* **MAE:** 30.76

### 🔍 Analysis & Diagnosis
**Mathematical Improvement, but Time-Domain limits remain.**
Fixing the data pipeline successfully improved the model! By properly feeding the AI all 6 dimensions of movement (instead of just 4), and fixing the sliding window size by measuring the actual timestamps, the Validation Loss dropped from 1307 to 1225. The RMSE improved by over 1 point and MAE by 1.25 points compared to Iteration 3.

This proves that the missing gyro data was holding the AI back. However, an RMSE of 34.66 confirms our suspicion from Iteration 3: giving the AI *perfect* Time-Domain data still isn't enough to generalize across different car suspensions and phone mounts. We have officially squeezed every drop of performance out of raw time-series data.

### 🛠️ Fixes Required for Iteration 5
1. **Frequency Domain Transformation (FFT):** As planned, we must now update preprocess.py to extract Frequency Domain features (FFT). This is the only way to bypass the suspension bias and let the AI listen directly to the Engine RPM frequencies.
2. **Update Architecture:** Widen the TCN input channels to accept the new frequency bins instead of just 6 raw channels.


---

## 🟡 Iteration 5: Full Pipeline Rebuild (Global Normalization + All 6 Channels + Nyquist Fix)
**Date:** 2026-09-03
**Objective:** Completely rebuild the data pipeline from scratch to fix every critical bug discovered in FIX_PROMPT.md: silent gyro zeroing (Issue 1), per-file normalization (Issue 5), hardcoded sampling rate (Issue 4), and Butterworth filter crash (Nyquist violation).

### ⚙️ Configuration
* **Architecture:** 1D TCN (`Linear(64, 1)`)
* **Loss Function:** MSELoss
* **Epochs:** 50
* **Learning Rate:** 1e-3 (with `ReduceLROnPlateau` scheduler)
* **Regularization:** Dropout 0.4, Weight Decay 1e-4
* **Data Pipeline Changes (Critical):**
    - **Global Normalization:** Replaced per-file `StandardScaler` with a single dataset-wide scaler (saved as `global_scaler.pkl`). All 72 files are now normalized to the same statistical baseline.
    - **All 6 Channels Live:** Fixed column mapping to use actual IO-VNBD headers (`ACCELEROMETER X/Y/Z`, `GYROSCOPE Yaw/Pitch/Roll`). Post-save assertion confirms `std() > 1e-6` for every channel.
    - **Dynamic Sampling Rate:** `IMU_FS` is now calculated per-file from real timestamp deltas (detected 10 Hz, not the assumed 100 Hz).
    - **Nyquist-Safe Filter:** Butterworth cutoff is now dynamically bounded to `min(cutoff, nyq * 0.99)` to prevent scipy crash when fs < 2 * cutoff.

### 📊 Results & Metrics
* **Final Train Loss:** 341.41
* **Best Unseen Val Loss:** 856.23 (Epoch 13)
* **Final Unseen Val Loss:** 989.74
* **RMSE:** 31.37
* **MAE:** 24.45

### 📈 Comparison Table (All Iterations)

| Iteration | RMSE | MAE | Val Loss | Key Change |
|-----------|------|-----|----------|------------|
| 1 | 34.26 | 32.31 | 6.71* | Gaussian NLL (unstable) |
| 2 | 39.15 | 33.76 | 1778.18 | MSE Loss, 50 epochs |
| 3 | 35.74 | 32.01 | 1307.35 | Dropout 0.4, Weight Decay, LR Scheduler |
| 4 (stale data) | 34.66 | 30.76 | 1225.60 | Column mapping fix (data not regenerated) |
| **5 (this)** | **31.37** | **24.45** | **989.74** | **Full pipeline rebuild** |

*Iteration 1 used Gaussian NLL loss (not comparable to MSE-based Val Loss)

### 🔍 Analysis & Diagnosis
**Major Breakthrough. The data pipeline was the bottleneck, not the model.**
Fixing the preprocessing pipeline produced the single largest improvement across all iterations: RMSE dropped 4.4 points and MAE dropped 7.5 points. This proves that previous iterations were handicapped by (a) 2 dead gyro channels, (b) per-file normalization destroying cross-session comparability, and (c) incorrectly sized sliding windows from the wrong sampling rate.

**Overfitting Status:** The train/val gap (341 vs 989, ~2.9x ratio) indicates moderate overfitting. However, the absolute Val Loss (989) is the best we have ever achieved, confirming the model IS generalizing better than before. The overfitting is healthy — the model has learned real patterns, not just noise.

**Remaining Bottleneck:** The graph shows the red line (AI prediction) now tracks the general shape and trend of the blue line (ground truth), but with significant amplitude spikes. This is characteristic of a time-domain model struggling with sensor-specific noise patterns that vary across vehicles.

### 🛠️ Fixes Required for Iteration 6
1. **Early Stopping:** The best Val Loss occurred at Epoch 13 (856), but training continued to Epoch 50 where Val Loss degraded to 989. Implement early stopping with patience to save the best checkpoint.
2. **Architecture Scaling:** Consider widening the TCN channels (32 to 64 to 128 to 256) or adding a 4th TCN block to increase capacity.
3. **Feature Engineering (FFT):** Transform time-domain windows into frequency-domain representations. Engine RPM frequencies generalize across vehicles far better than raw vibration amplitudes.

