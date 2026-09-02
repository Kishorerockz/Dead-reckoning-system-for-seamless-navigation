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

### 📈 Visual Result

![[Pasted image 20260903002155.png]]
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

### 📈 Visual Result

![[Pasted image 20260903004157.png]]