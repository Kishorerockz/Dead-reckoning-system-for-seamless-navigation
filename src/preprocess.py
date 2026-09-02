import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

# Constants based on typical IMU/GNSS datasets
IMU_FS = 100.0  # Assumed IMU sampling frequency (Hz). We will resample/verify this.
WINDOW_SIZE_SEC = 1.0
STRIDE_SEC = 0.5
CUTOFF_HZ = 10.0  # Low-pass filter cutoff frequency

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Applies a Butterworth low-pass filter to remove engine/road vibration noise."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # filtfilt applies the filter forward and backward to prevent phase shifting
    return filtfilt(b, a, data, axis=0)

def synchronize_and_interpolate(s_df, v_df):
    """
    Matches the high-frequency Smartphone (S) data with the lower-frequency 
    Vehicle (V) ground truth data.
    """
    # Standardize column names assuming standard CSV headers
    # Note: Actual header names might need tweaking based on exact IO-VNBD schema
    s_cols = [c for c in s_df.columns if 'time' in c.lower()]
    v_cols = [c for c in v_df.columns if 'time' in c.lower()]
    
    if not s_cols or not v_cols:
        raise ValueError("Could not find timestamp columns in CSVs.")
        
    t_s = s_cols[0]
    t_v = v_cols[0]

    # Convert timestamps to numeric relative seconds
    s_df['rel_time'] = s_df[t_s] - s_df[t_s].iloc[0]
    v_df['rel_time'] = v_df[t_v] - v_df[t_v].iloc[0]

    # Target features: Accel X/Y/Z, Gyro X/Y/Z
    feature_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    # We map columns defensively
    s_mapped = {}
    for fc in feature_cols:
        matched = [c for c in s_df.columns if fc.split('_')[1] in c.lower() and fc.split('_')[0] in c.lower()]
        s_mapped[fc] = s_df[matched[0]] if matched else np.zeros(len(s_df))
    
    # Target label: Speed
    speed_col = [c for c in v_df.columns if 'speed' in c.lower() or 'vel' in c.lower()]
    v_speed = v_df[speed_col[0]] if speed_col else np.zeros(len(v_df))

    # Interpolate Vehicle speed to match exactly with IMU timestamps
    interpolated_speed = np.interp(s_df['rel_time'], v_df['rel_time'], v_speed)
    
    # Construct unified dataframe
    unified_df = pd.DataFrame(s_mapped)
    unified_df['speed'] = interpolated_speed
    unified_df['rel_time'] = s_df['rel_time']
    
    return unified_df

def create_windows(features, labels, window_size, stride):
    """Slices continuous time-series into fixed-size overlapping blocks."""
    X, Y = [], []
    for i in range(0, len(features) - window_size, stride):
        X.append(features[i : i + window_size])
        # The label is the speed at the *end* of the window
        Y.append(labels[i + window_size - 1])
    return np.array(X), np.array(Y)

def preprocess_dataset(s_csv_path, v_csv_path, output_dir):
    """Main pipeline for a single paired dataset."""
    print(f"Processing: {os.path.basename(s_csv_path)}")
    s_df = pd.read_csv(s_csv_path, encoding='latin1')
    v_df = pd.read_csv(v_csv_path, encoding='latin1')

    # 1. Synchronization
    unified_df = synchronize_and_interpolate(s_df, v_df)
    
    # 2. Low-Pass Filtering
    feature_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    raw_features = unified_df[feature_cols].values
    filtered_features = butter_lowpass_filter(raw_features, CUTOFF_HZ, IMU_FS)
    
    # 3. Normalization
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(filtered_features)
    
    # 4. Windowing
    samples_per_window = int(WINDOW_SIZE_SEC * IMU_FS)
    stride_samples = int(STRIDE_SEC * IMU_FS)
    labels = unified_df['speed'].values
    
    X, Y = create_windows(normalized_features, labels, samples_per_window, stride_samples)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the preprocessed tensors
    out_name = os.path.basename(s_csv_path).replace('S-', 'Processed-').replace('.csv', '.npz')
    out_path = os.path.join(output_dir, out_name)
    np.savez_compressed(out_path, X=X, Y=Y)
    print(f"Saved tensor X shape {X.shape}, Y shape {Y.shape} to {out_path}")

if __name__ == "__main__":
    # Batch processing execution block
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Path to the Categorised IOVNB Dataset
    data_dir = os.path.join(base_dir, 'data', 'IO-VNBD', 'Synchronised V abd S datasets', 'Categorised IOVNB Dataset')
    output_dir = os.path.join(base_dir, 'results', 'processed_data')
    
    # Find all S-*.csv files recursively
    search_pattern = os.path.join(data_dir, '**', 'S-*.csv')
    s_files = glob.glob(search_pattern, recursive=True)
    
    if not s_files:
        print(f"No S-*.csv files found in {data_dir}. Please check the path.")
    else:
        print(f"Found {len(s_files)} smartphone datasets. Beginning batch preprocessing...")
        for s_csv in s_files:
            # The corresponding vehicle file replaces the 'S-' prefix with 'V-'
            s_basename = os.path.basename(s_csv)
            v_basename = s_basename.replace('S-', 'V-', 1)
            v_csv = os.path.join(os.path.dirname(s_csv), v_basename)
            
            if os.path.exists(v_csv):
                try:
                    preprocess_dataset(s_csv, v_csv, output_dir)
                except Exception as e:
                    print(f"Error processing {s_basename}: {e}")
            else:
                print(f"Warning: Could not find matching vehicle data '{v_basename}' for '{s_basename}'")
                
        print(f"\nPre-processing complete. All .npz tensor files saved to: {output_dir}")
