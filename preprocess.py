"""
preprocess.py — Data preprocessing and window extraction pipeline for NPPAD.

========================================================================================
DECISIONS FROM DATA EXPLORATION (STEP 2):
1. NORMAL DATA:
   - 'Operation_csv_data/Normal/1.csv' provides 302 steps of 100% steady-state operation.
   - For streaming accident scenarios, a realistic 60-step normal baseline (t=0..600s)
     is prepended to each accident transient. The fault starts at onset step = 60.
   - Normal windows (label=0) are extracted strictly from pre-accident segments of
     training and validation scenarios.
2. ONSET TIME:
   - Onset step is step 60 for accident scenarios (label switches from 0 to 1).
   - For pure normal scenarios, onset step is None / -1 (label remains 0 throughout).
3. SEVERITY ENCODING & BINNING:
   - Severity is encoded by the filename (e.g., 50.csv -> 50% physical break or variation).
   - Single-scenario severe accidents (ATWS, LOF, LACP, TT, SP) are assigned 100% severity.
   - Binned into tertiles:
     * Low:    severity <= 33%
     * Medium: 33% < severity <= 66%
     * High:   severity > 66%
4. FEATURE PREPROCESSING:
   - 'TIME' column is dropped.
   - Constant sensors (std == 0 in training normal data, 37 sensors) are dropped.
   - MinMaxScaler is fitted strictly on training normal data.
   - Sliding windows of length 30, stride 1 are generated.
========================================================================================
"""

import os
import glob
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import config

def extract_severity(filename, accident_type):
    # Single severe accidents default to 100.0
    if accident_type in ['ATWS', 'LOF', 'LACP', 'TT', 'SP']:
        return 100.0
    m = re.findall(r'[-+]?\d+', filename)
    if m:
        return abs(float(m[0]))
    return 50.0  # fallback default

def get_severity_bin(severity):
    if severity <= 33.0:
        return 'Low'
    elif severity <= 66.0:
        return 'Medium'
    else:
        return 'High'

def downsample_dataframe(df, max_steps):
    if len(df) <= max_steps:
        return df.copy()
    indices = np.linspace(0, len(df) - 1, max_steps, dtype=int)
    return df.iloc[indices].copy().reset_index(drop=True)

def create_windows(data, labels, window_size=30, stride=1):
    """
    Creates sliding windows from 2D array of shape (T, F) and 1D labels of shape (T,).
    Returns windows of shape (N, window_size, F) and window labels of shape (N,).
    A window is normal (0) only if all its time steps are normal.
    """
    num_steps, num_features = data.shape
    if num_steps < window_size:
        return np.empty((0, window_size, num_features)), np.empty((0,))
    
    starts = np.arange(0, num_steps - window_size + 1, stride)
    windows = np.array([data[s:s+window_size] for s in starts])
    # Window label: 1 if any step in window is fault, 0 if all normal
    win_labels = np.array([labels[s:s+window_size].max() for s in starts])
    return windows, win_labels

def preprocess():
    config.set_seed()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    log("="*75)
    log("NPPAD PREPROCESSING PIPELINE")
    log("="*75)
    
    op_dir = os.path.join(config.DATA_DIR, 'Operation_csv_data')
    if not os.path.exists(op_dir):
        raise FileNotFoundError(f"Directory {op_dir} does not exist.")

    # 1. Load pure normal baseline
    norm_path = os.path.join(op_dir, 'Normal', '1.csv')
    df_norm_full = pd.read_csv(norm_path)
    if 'TIME' in df_norm_full.columns:
        df_norm_features = df_norm_full.drop(columns=['TIME'])
    else:
        df_norm_features = df_norm_full.copy()
    
    # 60 steps baseline (10 mins at 10s intervals)
    normal_baseline = df_norm_features.iloc[:60].copy().reset_index(drop=True)
    log(f"Normal baseline lead-in segment shape: {normal_baseline.shape}")

    # 2. Collect all scenario files grouped by accident type
    accident_types = sorted([d for d in os.listdir(op_dir) if os.path.isdir(os.path.join(op_dir, d)) and d != 'Normal'])
    log(f"Found {len(accident_types)} accident types: {accident_types}")
    
    scenarios_by_type = {}
    for acc in accident_types:
        acc_path = os.path.join(op_dir, acc)
        files = sorted([f for f in os.listdir(acc_path) if f.endswith('.csv')])
        if config.QUICK_MODE:
            files = files[:config.QUICK_MAX_SCENARIOS]
        scenarios_by_type[acc] = [(os.path.join(acc_path, f), extract_severity(f, acc)) for f in files]
        log(f"  {acc:<8}: {len(scenarios_by_type[acc])} scenarios")

    # Pure normal scenarios from Normal/1.csv (multiple slices)
    normal_scenarios = [(norm_path, 0.0)]
    scenarios_by_type['Normal'] = normal_scenarios
    log(f"  {'Normal':<8}: {len(scenarios_by_type['Normal'])} scenarios")

    # 3. Stratified Split by scenario (70% train, 15% val, 15% test)
    rng = np.random.RandomState(config.SEED)
    train_files, val_files, test_files = [], [], []

    for acc, items in scenarios_by_type.items():
        shuffled = items.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * config.TRAIN_RATIO))
        n_val = max(1, int(n * config.VAL_RATIO)) if n >= 3 else 0
        if n == 1:
            # If only 1 scenario (e.g. Normal or single rare accident), put in train and copy reference to val/test
            train_files.extend([(f, sev, acc) for f, sev in shuffled])
            val_files.extend([(f, sev, acc) for f, sev in shuffled])
            test_files.extend([(f, sev, acc) for f, sev in shuffled])
        elif n == 2:
            train_files.append((shuffled[0][0], shuffled[0][1], acc))
            val_files.append((shuffled[1][0], shuffled[1][1], acc))
            test_files.append((shuffled[1][0], shuffled[1][1], acc))
        else:
            train_part = shuffled[:n_train]
            val_part = shuffled[n_train:n_train + n_val]
            test_part = shuffled[n_train + n_val:]
            if len(test_part) == 0 and len(val_part) > 1:
                test_part = [val_part.pop()]
            train_files.extend([(f, sev, acc) for f, sev in train_part])
            val_files.extend([(f, sev, acc) for f, sev in val_part])
            test_files.extend([(f, sev, acc) for f, sev in test_part])

    log(f"\nScenario split counts:")
    log(f"  Train: {len(train_files)} scenarios")
    log(f"  Val:   {len(val_files)} scenarios")
    log(f"  Test:  {len(test_files)} scenarios")

    # 4. Feature Selection: drop constant and non-numeric columns
    # We inspect standard deviation on the normal baseline
    stds = normal_baseline.std()
    non_constant_sensors = stds[stds > 1e-6].index.tolist()
    constant_sensors = stds[stds <= 1e-6].index.tolist()
    log(f"\nFeature filtering:")
    log(f"  Original sensor features: {len(df_norm_features.columns)}")
    log(f"  Constant sensors dropped: {len(constant_sensors)}")
    log(f"  Active sensors kept:      {len(non_constant_sensors)}")

    # 5. Build Training Normal Data to fit MinMaxScaler
    # Extract normal-only steps from all training scenarios
    train_normal_steps = []
    normal_baseline_active = normal_baseline[non_constant_sensors]
    
    # Each training scenario contributes its 60-step normal lead-in
    for _ in train_files:
        train_normal_steps.append(normal_baseline_active.values)
    train_normal_steps_arr = np.concatenate(train_normal_steps, axis=0)

    # Fit MinMaxScaler strictly on training normal data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_normal_steps_arr)
    log(f"Fitted MinMaxScaler on {len(train_normal_steps_arr)} normal time steps.")

    # 6. Process scenarios into sequences and sliding windows
    def process_scenario_list(file_list, is_train_or_val_normal_extraction=False):
        processed_scenarios = []
        normal_windows_list = []

        for fpath, sev, acc in file_list:
            try:
                # Delimiter check
                sep = ',' if fpath.endswith('.csv') else r'\s+'
                df = pd.read_csv(fpath, sep=sep, engine='python')
                if 'TIME' in df.columns:
                    df = df.drop(columns=['TIME'])
                # Select active sensors
                df = df[[c for c in non_constant_sensors if c in df.columns]]
                
                # Check if scenario is pure normal
                is_pure_normal = (acc == 'Normal')
                
                if is_pure_normal:
                    # Pure normal scenario
                    df_down = downsample_dataframe(df, config.MAX_STEPS)
                    labels = np.zeros(len(df_down), dtype=int)
                    onset_step = -1
                    data_arr = scaler.transform(df_down.values)
                else:
                    # Prepend 60 steps normal baseline
                    # Downsample transient so total <= MAX_STEPS
                    max_transient_steps = config.MAX_STEPS - len(normal_baseline_active)
                    df_transient = downsample_dataframe(df, max_transient_steps)
                    
                    df_combined = pd.concat([normal_baseline_active, df_transient], ignore_index=True)
                    data_arr = scaler.transform(df_combined.values)
                    
                    onset_step = len(normal_baseline_active)  # step 60
                    labels = np.zeros(len(df_combined), dtype=int)
                    labels[onset_step:] = 1  # fault steps
                
                # Create sliding windows
                windows, win_labels = create_windows(data_arr, labels, window_size=config.WINDOW, stride=config.STRIDE)
                
                # If we need normal-only windows (for training / val baseline)
                if is_train_or_val_normal_extraction and len(windows) > 0:
                    norm_mask = (win_labels == 0)
                    if np.any(norm_mask):
                        normal_windows_list.append(windows[norm_mask])

                sev_bin = get_severity_bin(sev)
                scen_dict = {
                    'filename': os.path.basename(fpath),
                    'filepath': fpath,
                    'accident_type': acc,
                    'severity': sev,
                    'severity_bin': sev_bin,
                    'onset_step': onset_step,
                    'data': data_arr,            # scaled (T, F)
                    'labels': labels,            # (T,)
                    'windows': windows,          # (N_win, W, F)
                    'window_labels': win_labels  # (N_win,)
                }
                processed_scenarios.append(scen_dict)
            except Exception as e:
                log(f"Error processing {fpath}: {e}")

        norm_windows_arr = np.concatenate(normal_windows_list, axis=0) if normal_windows_list else np.empty((0, config.WINDOW, len(non_constant_sensors)))
        return processed_scenarios, norm_windows_arr

    log("\nProcessing train scenarios...")
    train_scenarios, train_normal_windows = process_scenario_list(train_files, is_train_or_val_normal_extraction=True)
    
    log("Processing val scenarios...")
    val_scenarios, val_normal_windows = process_scenario_list(val_files, is_train_or_val_normal_extraction=True)
    
    log("Processing test scenarios...")
    test_scenarios, _ = process_scenario_list(test_files, is_train_or_val_normal_extraction=False)

    log(f"\nWindow Extraction Summary:")
    log(f"  Training normal windows:   {train_normal_windows.shape}")
    log(f"  Validation normal windows: {val_normal_windows.shape}")
    log(f"  Total test scenarios:      {len(test_scenarios)}")
    test_win_count = sum(len(s['windows']) for s in test_scenarios)
    log(f"  Total test windows:        {test_win_count}")

    # 7. Package and save preprocessed data
    preprocessed_data = {
        'sensor_names': non_constant_sensors,
        'scaler': scaler,
        'train_normal_windows': train_normal_windows,
        'val_normal_windows': val_normal_windows,
        'train_scenarios': train_scenarios,
        'val_scenarios': val_scenarios,
        'test_scenarios': test_scenarios
    }

    out_pkl = os.path.join(config.DATA_DIR, 'preprocessed.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(preprocessed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"\nSaved preprocessed dataset to: {out_pkl}")

    # 8. Save summary text
    sum_file = os.path.join(config.RESULTS_DIR, 'preprocessing_summary.txt')
    with open(sum_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_lines))
    log(f"Saved preprocessing summary to: {sum_file}")
    log("="*75)

if __name__ == '__main__':
    preprocess()
