# Fix Request — Dead Reckoning IMU Preprocessing & Training Pipeline

## Context
This is the `Dead-reckoning-system-for-seamless-navigation` repo (ISRO PS 26168,
Intelligent Dead Reckoning). The EKF/filter side (`dr_pipeline.py`) is verified
working and reproducible — **do not touch it**. The issues below are in the
ML data pipeline (`preprocess.py` → `train_tcn.py` → `evaluate_model.py`).
Fix them in that order; each depends on the previous one.

---

## Issue 1 (CRITICAL): Two of six IMU channels are silently zeroed in every processed file

### What's wrong
In `src/preprocess.py`, `synchronize_and_interpolate()` matches source CSV
columns to canonical feature names using this heuristic:

```python
feature_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
for fc in feature_cols:
    matched = [c for c in s_df.columns if fc.split('_')[1] in c.lower() and fc.split('_')[0] in c.lower()]
    s_mapped[fc] = s_df[matched[0]] if matched else np.zeros(len(s_df))
```

For `gyro_x`, this looks for any column containing **both** `'x'` and `'gyro'`
(substring match, case-insensitive). If no column matches, it **silently**
defaults to an all-zero array instead of raising an error.

### Proof this is currently happening
I inspected the already-generated `results/processed_data/*.npz` files.
Every single one of the 72 files has **channel index 3 (`gyro_x`) and channel
index 5 (`gyro_z`) with `std() == 0.0`** — i.e., dead/constant-zero input
across the entire dataset. Only `acc_x`, `acc_y`, `acc_z`, `gyro_y` contain
real signal. This means every model trained so far (Iteration 1 and 2 in the
experiment log) was trained on 4 real channels instead of 6, which is a
plausible contributor to the poor generalization seen in both iterations.

### What to do
1. **Diagnose first, don't guess.** Load one real `S-*.csv` from
   `data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/`
   and print `df.columns.tolist()`. Do the same for a `V-*.csv`. Paste the
   actual column names before writing the fix — the IO-VNBD schema may use
   naming like `GyroX`, `wx`, `gyro_x (rad/s)`, `Gyroscope_X`, etc., and the
   current heuristic may simply not be matching whatever pattern is actually
   used for the X and Z gyro axes (it apparently *does* match Y correctly and
   *does* match all three accel axes, so the bug is specific to how X/Z gyro
   columns happen to be named in this dataset).
2. Replace the fragile double-substring match with an explicit,
   case-insensitive lookup table built from the real column names you find
   in step 1 — no silent fallback to zeros. If a required column genuinely
   isn't found for a given file, **raise an exception and skip/log that file
   loudly** (the existing `try/except` in the `__main__` block of
   `preprocess.py` already logs per-file errors — let it do its job instead
   of masking the problem with zeros).
3. Add an automated post-processing sanity check: after writing each `.npz`,
   assert that `X[..., i].std() > 1e-6` for every channel `i` before saving,
   and print a loud warning (with the filename) if any channel is dead. This
   prevents this exact bug from silently recurring for any future dataset or
   schema drift.
4. **Re-run preprocessing on all 72 files** with the fix in place, and
   **re-run `train_tcn.py` from scratch** (the current
   `results/saved_models/tiny_tcn_qat_int8.pth` was trained on broken data —
   do not just fine-tune it, retrain fresh). Then re-run `evaluate_model.py`
   and regenerate `speed_evaluation.png`, and update
   `docs/#L01f9ea AI Model Experiment & Tuning Log.md` with a new
   "Iteration 3" entry documenting the RMSE/MAE before vs. after this fix —
   this is a genuinely useful before/after data point for the writeup.

---

## Issue 2: "Driver-independent split" claim is not verified against what the code actually does

### What's wrong
`docs/log.md` and `docs/#L01f9ea AI Model Experiment & Tuning Log.md` both
describe the train/test split as being by **driver identity**: "Train:
Driver E [64 files], Test: Drivers A, B, D [8 files]." But `train_tcn.py`'s
actual split logic is:

```python
train_files = [f for f in all_files if '-V' in os.path.basename(f)]
test_files = [f for f in all_files if '-V' not in os.path.basename(f)]
```

This splits by whether `-V` appears in the filename, not by any driver
label. The real filenames in `results/processed_data/` are things like
`Processed-Vta10.npz`, `Processed-Vw14b.npz`, `Processed-Vfa02.npz`,
`Processed-M.npz`, `Processed-S1.npz`, `Processed-Y1.npz` — these read like
route/scenario/session codes, not driver letters A/B/D/E.

### What to do
1. Check the IO-VNBD dataset's actual README/documentation
   (github.com/onyekpeu/IO-VNBD) for what the `S-*`/`V-*` filename prefixes
   and suffixes (`ta`, `tb`, `w`, `fa`, `M`, `S`, `Y`, etc.) actually encode —
   driver, route, weather condition, or something else.
2. If they encode drivers, rename the split logic to filter on the actual
   driver-identifying substring/field (not the coincidental `-V` presence)
   and correct the doc language to name the real driver IDs.
3. If they do **not** encode drivers (most likely, given the naming
   pattern), correct `log.md` and the experiment log to describe the split
   accurately (e.g., "session/route-independent split," not "driver-
   independent"). This isn't just a documentation nitpick — an ISRO reviewer
   who checks the dataset schema and finds the claim doesn't match will
   discount the whole generalization argument.

---

## Issue 3: Real-dataset validation is still outstanding

### What's wrong
`dr_pipeline.py`'s headline result (0.08% single-seed / 0.193%±0.135%
Monte Carlo drift, "MEETS ISRO benchmark") is computed against a physics-
simulated synthetic trajectory + published MEMS IMU error model, **not**
real IO-VNBD sensor data. This is honestly disclosed in the code and docs
already — good — but it's still the single biggest open risk before
submission, since real accelerometer/gyro data has non-idealities (clipping,
temperature-dependent bias drift, mount rattle, GPS multipath in the ground
truth itself) that a hand-tuned Gaussian error model won't fully capture.

### What to do
1. Once Issue 1 is fixed and real IO-VNBD data can be pulled without LFS
   rate-limiting, run `load_real_dataset()` (already stubbed in
   `dr_pipeline.py`) against real `S-*.csv`/`V-*.csv` pairs and re-run the
   same Monte Carlo benchmark harness on real data.
2. Replace the synthetic numbers in `core_mechanism.md` §4 with the real-
   data numbers, keeping the synthetic results as a clearly labeled
   secondary "architecture sanity check" table rather than removing them —
   the synthetic ablation is still a legitimate and useful result, it just
   shouldn't be the only evidence in the final submission.

---

## Issue 4 (minor, lower priority)

1. `preprocess.py` hardcodes `IMU_FS = 100.0` with a comment admitting
   it's an assumption ("We will resample/verify this"). Some docs state the
   real sampling rate may be up to 200Hz depending on phone. Verify the
   actual sample rate per-file (e.g., from the real timestamp column deltas
   found in Issue 1's diagnosis step) rather than assuming a constant, since
   a wrong `IMU_FS` silently mis-sizes the Butterworth cutoff and the
   windowing (`WINDOW_SIZE_SEC * IMU_FS`) for any file recorded at a
   different rate.
2. Add a `requirements.txt` pinning exact versions of `torch`, `numpy`,
   `scipy`, `pandas`, `scikit-learn` — the QAT fuse/prepare/convert sequence
   in `train_tcn.py`/`evaluate_model.py` is fragile across PyTorch versions
   and this should be locked down before other people (or CI) run it.

---

## Acceptance criteria (how to know this is actually fixed)

- [ ] Printed real column names from at least one real `S-*.csv` and
      `V-*.csv`, pasted into the PR/commit description.
- [ ] `preprocess.py` no longer silently defaults missing columns to zeros
      — it raises/logs loudly instead.
- [ ] Re-generated all `results/processed_data/*.npz` files; verified via
      script that `X[...,i].std() > 1e-6` for all 6 channels in all files.
- [ ] Retrained `tiny_tcn_qat_int8.pth` from scratch on the corrected data;
      new RMSE/MAE recorded as "Iteration 3" in the experiment log with a
      before/after comparison against Iteration 2's numbers.
- [ ] Regenerated `speed_evaluation.png` from the retrained model.
- [ ] Confirmed (or corrected) the "driver-independent" split claim against
      the real IO-VNBD schema, and updated `log.md` /
      `#L01f9ea AI Model Experiment & Tuning Log.md` wording to match reality.
- [ ] `dr_pipeline.py` is untouched and still reproduces the existing
      synthetic Monte Carlo numbers (regression check — run it before and
      after all other changes to confirm nothing broke).
