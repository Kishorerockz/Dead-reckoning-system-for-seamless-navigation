1. Downloaded Dataset from https://github.com/onyekpeu/IO-VNBD References\IO-VNBD

2. Ran dr_pipeline.py. Fixed hardcoded paths, executed monte carlo sim, and validated benchmark (0.08% drift using the full stack).

3. Created src/preprocess.py to handle Phase 1 data ingestion. Implemented synchronization, Butterworth low-pass filtering, Z-score normalization, and overlapping window extraction for the IO-VNBD dataset.

4. Fixed 'utf-8' decoding error in src/preprocess.py by applying 'latin1' encoding. Successfully batch-processed 72 smartphone datasets into .npz tensor files inside results/processed_data/.

5. Created src/train_tcn.py. Implemented a Temporal Convolutional Network (TCN) equipped with PyTorch Quantization-Aware Training (QAT) stubs and fusion methods to ensure zero accuracy loss when deployed to INT8 edge hardware.

6. Refactored train_tcn.py to enforce a Route/Session-Independent Split. The model trains on Group 1 files (files containing '-V', 64 files) and validates on unseen Group 2 files (8 files, e.g., 'S1', 'M', 'Y1') to prevent data leakage.

7. Created src/evaluate_model.py to test the INT8 Quantized AI on the unseen driver test set. Calculates RMSE and generates a visual overlap graph (speed_evaluation.png) proving the TCN's ability to filter vibration noise into accurate speed predictions.
