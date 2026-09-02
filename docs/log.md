1. Downloaded Dataset from https://github.com/onyekpeu/IO-VNBD References\IO-VNBD

2. Ran dr_pipeline.py. Fixed hardcoded paths, executed monte carlo sim, and validated benchmark (0.08% drift using the full stack).

3. Created src/preprocess.py to handle Phase 1 data ingestion. Implemented synchronization, Butterworth low-pass filtering, Z-score normalization, and overlapping window extraction for the IO-VNBD dataset.

4. Fixed 'utf-8' decoding error in src/preprocess.py by applying 'latin1' encoding. Successfully batch-processed 72 smartphone datasets into .npz tensor files inside results/processed_data/.
