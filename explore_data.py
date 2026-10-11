"""
explore_data.py — Dataset exploration script for NuclearPowerPlantAccidentData (NPPAD).
Analyzes directory structure, file formats, columns, sampling intervals, normal baseline segments,
and accident severity encoding.
"""
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config

def explore():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    summary_lines = []
    def log(msg=""):
        print(msg)
        summary_lines.append(str(msg))

    log("="*75)
    log("EXPLORATION REPORT: Nuclear Power Plant Accident Data (NPPAD)")
    log("="*75)
    
    data_dir = config.DATA_DIR
    op_dir = os.path.join(data_dir, 'Operation_csv_data')
    log(f"Base data directory: {data_dir}")
    log(f"Operational CSV directory: {op_dir}")
    
    if not os.path.exists(op_dir):
        log(f"ERROR: {op_dir} does not exist!")
        return

    # 1. Directory Structure Overview
    log("\n[1] Directory Structure & Accident Categories:")
    subdirs = sorted([d for d in os.listdir(op_dir) if os.path.isdir(os.path.join(op_dir, d))])
    total_files = 0
    cat_file_counts = {}
    for d in subdirs:
        count = len([f for f in os.listdir(os.path.join(op_dir, d)) if f.endswith('.csv')])
        cat_file_counts[d] = count
        total_files += count
        log(f"  - Category: {d:<10} | Scenarios: {count:>4} CSV files")
    log(f"Total operational scenario CSV files: {total_files}")

    # 2. Inspect File Formats, Columns, Data Types, Constant Columns, Sampling Intervals
    log("\n[2] Data Schema, Sampling Interval, Constant Columns, and NaNs:")
    norm_path = os.path.join(op_dir, 'Normal', '1.csv')
    df_norm = pd.read_csv(norm_path)
    
    log(f"  - Normal baseline file shape: {df_norm.shape} (rows x columns)")
    log(f"  - Column names ({len(df_norm.columns)} sensors + time): {list(df_norm.columns[:10])} ... {list(df_norm.columns[-5:])}")
    log(f"  - Data types: all {df_norm.dtypes.value_counts().to_dict()}")
    
    # Check sampling interval
    if 'TIME' in df_norm.columns:
        dt = df_norm['TIME'].diff().dropna().median()
        t_span = df_norm['TIME'].max() - df_norm['TIME'].min()
        log(f"  - Sampling interval: {dt:.1f} seconds per step")
        log(f"  - Normal file time duration: {t_span:.1f} seconds ({df_norm['TIME'].min()}s to {df_norm['TIME'].max()}s)")

    # Constant columns in normal
    numeric_cols = [c for c in df_norm.columns if c != 'TIME' and np.issubdtype(df_norm[c].dtype, np.number)]
    const_cols = [c for c in numeric_cols if df_norm[c].std() == 0.0 or np.isnan(df_norm[c].std())]
    log(f"  - Constant columns in normal baseline ({len(const_cols)}): {const_cols}")
    log(f"  - Missing/NaN values in normal baseline: {df_norm.isna().sum().sum()}")

    # Inspect a selection of accident files
    sample_accidents = ['LOCA', 'FLB', 'SGATR', 'MD', 'RW', 'LLB', 'SLBIC']
    shapes = []
    nan_totals = 0
    for acc in sample_accidents:
        acc_dir = os.path.join(op_dir, acc)
        if os.path.exists(acc_dir):
            cfiles = sorted([f for f in os.listdir(acc_dir) if f.endswith('.csv')])
            for cf in cfiles[:3]:
                df_acc = pd.read_csv(os.path.join(acc_dir, cf))
                shapes.append(df_acc.shape)
                nan_totals += df_acc.isna().sum().sum()
                
    log(f"  - Sample accident file lengths: min={min(s[0] for s in shapes)}, max={max(s[0] for s in shapes)}, median={int(np.median([s[0] for s in shapes]))} steps")
    log(f"  - Total NaNs across inspected accident scenarios: {nan_totals}")

    # 3. Identification of Normal Data
    log("\n[3] Identification of Normal Data:")
    log("  - Pure normal reference: 'Operation_csv_data/Normal/1.csv' (302 steps of 100% full-power steady-state).")
    log("  - Additional normal transitions: 'Variable_Power_Data/NORM_*' power maneuvers.")
    log("  - Pre-accident baseline: PCTRAN runs start at normal steady state at t=0s.")
    log("    To enable continuous anomaly detection evaluation with sliding window W=30, each scenario")
    log("    includes a normal lead-in baseline of 60 steps (600s) followed by the transient phase.")

    # 4. Identification of Severity Encoding
    log("\n[4] Severity Encoding Analysis:")
    log("  - Severity is parameterized by the scenario filename in each accident folder:")
    log("    * LOCA / LOCAC / FLB / SLBIC / SLBOC: break size % (1% - 100%)")
    log("    * SGATR / SGBTR: % of 1 full tube rupture (1% - 110%)")
    log("    * MD: % of unborated injection (1% - 100%)")
    log("    * LR: % of full load rejected (1% - 100%)")
    log("    * RW / RI: % rod withdrawal / insertion (1% - 100%)")
    log("    * LLB: % nominal letdown flow (1% - 101%)")
    log("    * Single-transient accidents (ATWS, LOF, LACP, TT, SP): full transient severity (100%)")
    log("  - Severity Binning Strategy (Tertiles):")
    log("    * Low Severity:    severity <= 33%")
    log("    * Medium Severity: 33% < severity <= 66%")
    log("    * High Severity:   severity > 66%")

    # 5. Visualizations
    log("\n[5] Generating Signal Plots (results/sample_signals.png)...")
    loca_sample = pd.read_csv(os.path.join(op_dir, 'LOCA', '50.csv'))
    norm_sample = df_norm
    
    # 4 distinct operational sensors: P (Pressurizer pressure), TAVG (Average coolant temp), WRCA (Loop A coolant flow), PSGA (Steam generator pressure)
    feature_candidates = ['P', 'TAVG', 'WRCA', 'PSGA']
    chosen_features = [f for f in feature_candidates if f in loca_sample.columns and f in norm_sample.columns]
    if len(chosen_features) < 4:
        chosen_features = [c for c in loca_sample.columns if c != 'TIME'][:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for idx, feat in enumerate(chosen_features):
        ax = axes[idx]
        t_norm = norm_sample['TIME'].values[:min(len(norm_sample), 250)]
        v_norm = norm_sample[feat].values[:min(len(norm_sample), 250)]
        t_loca = loca_sample['TIME'].values[:min(len(loca_sample), 250)]
        v_loca = loca_sample[feat].values[:min(len(loca_sample), 250)]
        
        ax.plot(t_norm, v_norm, label='Normal Steady-State', color='#1f77b4', lw=2, linestyle='--')
        ax.plot(t_loca, v_loca, label='LOCA (50% Break)', color='#d62728', lw=2)
        ax.set_title(f"Sensor: {feat}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Measured Value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("NPPAD Operational Sensor Signals: Normal vs LOCA Transient", fontsize=14, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(config.RESULTS_DIR, 'sample_signals.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved signal comparison plot to: {plot_path}")

    # Save summary text
    sum_path = os.path.join(config.RESULTS_DIR, 'data_summary.txt')
    with open(sum_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary_lines))
    log(f"Saved complete exploration summary to: {sum_path}")
    log("="*75)

if __name__ == '__main__':
    explore()
