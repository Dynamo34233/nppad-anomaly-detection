"""
demo_app.py — Interactive Streamlit Web Application for NPPAD Anomaly Detection.
Allows selecting test scenarios, visualizing signals, and comparing Baseline vs Proposed P2 models.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch

import config
from models_baseline import Conv1DAutoencoder
from models_proposed import TCNAE, adaptive_alarm

st.set_page_config(page_title="NPPAD Anomaly Detection", layout="wide")

st.title("☢️ Nuclear Power Plant Accident Anomaly Detection")
st.markdown("Proof-of-Concept benchmark evaluating baseline and proposed causal TCN models on the **NPPAD** dataset.")

@st.cache_resource
def load_resources():
    prep_path = os.path.join(config.DATA_DIR, 'preprocessed.pkl')
    sample_path = os.path.join(config.MODELS_DIR, 'sample_scenarios.pkl')
    if os.path.exists(prep_path):
        with open(prep_path, 'rb') as f:
            prep = pickle.load(f)
    elif os.path.exists(sample_path):
        with open(sample_path, 'rb') as f:
            prep = pickle.load(f)
    else:
        raise FileNotFoundError("Neither preprocessed.pkl nor sample_scenarios.pkl found.")
    with open(os.path.join(config.MODELS_DIR, 'baseline_info.json'), 'r') as f:
        base_info = json.load(f)
    with open(os.path.join(config.MODELS_DIR, 'proposed_info.json'), 'r') as f:
        prop_info = json.load(f)
    selected_sensors = np.load(os.path.join(config.MODELS_DIR, 'selected_sensors.npy'))

    device = torch.device('cpu')
    n_features_all = len(prep['sensor_names'])
    n_features_sel = len(selected_sensors)
    w_size = config.WINDOW

    # Baseline Conv1D
    b3 = Conv1DAutoencoder(n_features_all, w_size, latent_dim=config.LATENT_DIM)
    b3.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'b3_cnnae.pth'), map_location=device))
    b3.eval()

    # Proposed P2
    p2 = TCNAE(n_features_sel, w_size, channels=config.TCN_CHANNELS, kernel_size=config.TCN_KERNEL, dilations=config.TCN_DILATIONS, latent_dim=config.LATENT_DIM)
    p2.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'p1_p2_tcnae.pth'), map_location=device))
    p2.eval()

    return prep, base_info, prop_info, selected_sensors, b3, p2

try:
    prep, base_info, prop_info, selected_sensors, b3, p2 = load_resources()
    test_scenarios = prep['test_scenarios']

    st.sidebar.header("Scenario Controls")
    scen_labels = [f"[{s['accident_type']}] {s['filename']} (Sev: {s['severity']:.1f}% - {s['severity_bin']})" for s in test_scenarios]
    selected_label = st.sidebar.selectbox("Select Test Scenario:", scen_labels)
    selected_idx = scen_labels.index(selected_label)
    scen = test_scenarios[selected_idx]

    wins = scen['windows']
    w_labels = scen['window_labels']
    onset_step = scen['onset_step']
    fault_win = max(0, onset_step - config.WINDOW + 1) if onset_step >= 0 else -1

    # Inference B3
    with torch.no_grad():
        b_in = torch.tensor(wins, dtype=torch.float32)
        r_b3 = b3(b_in)
        mse_b3 = torch.mean((b_in - r_b3)**2, dim=(1,2)).numpy()

    # Inference P2
    with torch.no_grad():
        p_in = torch.tensor(wins[:, :, selected_sensors], dtype=torch.float32)
        r_p2 = p2(p_in)
        mse_p2 = torch.mean((p_in - r_p2)**2, dim=(1,2)).numpy()

    thresh_b3 = base_info['thresholds']['B3_CNNAE']
    alarms_b3 = (mse_b3 > thresh_b3)

    k_val = prop_info['P2']['calibrated_k']
    alarms_p2, smoothed_p2, dyn_thresh_p2 = adaptive_alarm(
        mse_p2, alpha=prop_info['P2']['alpha'], window=prop_info['P2']['window'], k=k_val, persist=prop_info['P2']['persistence']
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accident Type", scen['accident_type'])
    col2.metric("Physical Severity", f"{scen['severity']:.1f}% ({scen['severity_bin']})")
    first_b3 = np.where(alarms_b3)[0]
    first_b3_alarm = int(first_b3[0]) if len(first_b3) > 0 else "None"
    first_p2 = np.where(alarms_p2)[0]
    first_p2_alarm = int(first_p2[0]) if len(first_p2) > 0 else "None"
    col3.metric("Baseline Alarm Window", str(first_b3_alarm))
    col4.metric("Proposed P2 Alarm Window", str(first_p2_alarm))

    st.subheader("Residual & Threshold Trajectories")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = np.arange(len(wins))

    # Baseline Plot
    ax1.plot(t, mse_b3, label="Conv1D-AE Residual", color='#1f77b4', lw=1.5)
    ax1.axhline(y=thresh_b3, color='#d62728', linestyle='--', label=f"Fixed Threshold ({thresh_b3:.4f})")
    if fault_win >= 0:
        ax1.axvline(x=fault_win, color='black', linestyle=':', lw=2, label="Fault Onset")
    if len(first_b3) > 0:
        ax1.scatter([first_b3[0]], [mse_b3[first_b3[0]]], color='red', s=80, zorder=5, label="Alarm Raised")
    ax1.set_title("Baseline (Conv1D-AE) Residuals & Fixed Alarm", fontweight='bold')
    ax1.set_ylabel("MSE")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Proposed Plot
    ax2.plot(t, smoothed_p2, label="P2 EWMA Residual", color='#2ca02c', lw=1.5)
    ax2.plot(t, dyn_thresh_p2, color='#e377c2', linestyle='--', label="Adaptive MAD Threshold")
    if fault_win >= 0:
        ax2.axvline(x=fault_win, color='black', linestyle=':', lw=2, label="Fault Onset")
    if len(first_p2) > 0:
        ax2.scatter([first_p2[0]], [smoothed_p2[first_p2[0]]], color='crimson', marker='X', s=100, zorder=5, label="Alarm Raised")
    ax2.set_title("Proposed P2 (Causal TCN-AE + Adaptive Threshold + Persistence)", fontweight='bold')
    ax2.set_xlabel("Window Index")
    ax2.set_ylabel("Residual / Threshold")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig)

except Exception as e:
    st.info("Run `python main.py` first to train models and generate artifacts before viewing the demo.")
    st.error(f"Details: {e}")
