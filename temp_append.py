import os

text = """
## 🔴 Iteration 3: Strong Regularization & LR Scheduling
**Date:** 2026-09-03
**Objective:** Prevent the severe domain-overfitting seen in Iteration 2 by adding a mathematical "blindfold" (Dropout), weight penalization, and a learning rate scheduler.

### ⚙️ Configuration
* **Architecture:** 1D TCN (`Linear(64, 1)`)
* **Loss Function:** MSELoss
* **Epochs:** 50
* **Learning Rate:** 1e-3 (with `ReduceLROnPlateau` scheduler)
* **Regularization:** Dropout increased to 0.4, Weight Decay (L2) set to 1e-4

### 📊 Results & Metrics
* **Final Train Loss:** 576.28
* **Unseen Val Loss:** 1307.35
* **RMSE:** 35.74
* **MAE:** 32.01

### 🔍 Analysis & Diagnosis
**Success, but reaching the limits of Time-Domain data.**
The regularization worked exactly as intended! By increasing Dropout and adding Weight Decay, we successfully stopped the AI from memorizing Driver E's car (Train Loss was restricted to 576, up from 417). Because the AI was forced to generalize, the Unseen Validation Loss improved massively (dropping from 1778 down to 1307), and the RMSE dropped by nearly 3.5 points.

However, an RMSE of 35 is still too high for ISRO's <10% drift requirement. We have hit the mathematical ceiling of what raw, time-domain IMU data can provide. Raw accelerometer amplitudes are simply too biased by individual vehicle suspension stiffness and phone mounting angles.

### 🛠️ Fixes Required for Iteration 4 (Phase 3: Feature Engineering)
1. **Frequency Domain Transformation (FFT):** We must upgrade `preprocess.py` to convert the raw time-series vibration windows into Fast Fourier Transforms (FFT). Engine RPMs (frequencies) remain mathematically constant across different cars and mounts.
2. **Update Architecture:** The TCN will need to be slightly widened to accept frequency bins instead of raw 3-axis amplitudes.
3. **Fix Logging:** Update `evaluate_model.py` print statement to correctly state Iteration 4.

### 📈 Visual Result
![Iteration 3 Speed Evaluation](../model tuning graphs/speed_evaluation_iter3.png)
"""

with open('docs/model_experiments_log.md', 'a', encoding='utf-8') as f:
    f.write(text)
