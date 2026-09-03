## 🔴 Iteration 1: Initial QAT TCN with Gaussian NLL
**Date:** 2026-09-02
**Objective:** Baseline test to ensure Quantization-Aware Training (QAT) compiles and to attempt simultaneous learning of Velocity and Variance.

### ⚙️ Configuration
* **Architecture:** 1D Temporal Convolutional Network (TCN)
* **Loss Function:** Gaussian Negative Log-Likelihood (Gaussian NLL)
* **Epochs:** 15
* **Learning Rate:** 1e-3
* **Data Split:** Driver-Independent (Train: Driver E, Test: Drivers A, B, D)

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
* **Data Split:** Driver-Independent (Train: Driver E [64 files], Test: Drivers A, B, D [8 files])

### 📊 Results & Metrics
* **Final Train Loss:** 417.86
* **Unseen Val Loss:** 1778.18
* **RMSE:** 39.15
* **MAE:** 33.76

### 🔍 Analysis & Diagnosis (Why did this happen?)
**Severe Overfitting & Domain Shift.** 
The network successfully learned how to predict speed, but *only* for the training car! The Train Loss dropped beautifully from 1713 down to 417, proving the neural network architecture works. However, the Unseen Validation Loss exploded from 692 up to 1778. 

This means the AI completely memorized Driver E's specific engine vibrations, suspension stiffness, and phone-mount rattles. When tested on the completely different cars of Drivers A, B, and D, those raw vibration patterns were so different that the AI failed to generalize. 

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
The regularization worked exactly as intended! By increasing the Dropout and adding Weight Decay, we successfully stopped the AI from blindly memorizing Driver E's car (Train Loss was restricted to 576, up from 417). Because the AI was forced to look for general patterns instead of memorizing, the Unseen Validation Loss improved massively (dropping from 1778 down to 1307), and the RMSE improved by nearly 3.5 points.

However, an RMSE of 35 is still too high for ISRO's <10% drift requirement. We have hit the mathematical ceiling of what raw, time-domain accelerometer data can provide. Raw vibration amplitudes are too easily biased by a specific car's suspension stiffness or the exact angle the phone is mounted.

### 🛠️ Fixes Required for Iteration 4 (Phase 3: Feature Engineering)
1. **Frequency Domain Transformation (FFT):** We must update preprocess.py to convert the raw time-series vibration windows into Fast Fourier Transforms (FFT) or Spectrograms. Engine RPMs (frequencies) remain mathematically constant across different cars, whereas raw bump amplitudes do not.
2. **Update Evaluation Script:** Update the print statement in evaluate_model.py to correctly state "Iteration 4".


