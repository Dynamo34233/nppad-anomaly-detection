"""
evaluate.py — Evaluation pipeline for NPPAD benchmark.
Evaluates baselines (B1-B4) and proposed variants (P0-P2) on all test scenarios.
Computes Detection Rate, False Alarm Rate (per 1000 steps), Mean Detection Delay, ROC-AUC,
overall and broken down by severity bin (Low, Medium, High).
Generates comparison tables and publication-quality plots.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch
import config
from models_baseline import FCAutoencoder, Conv1DAutoencoder, LSTMAutoencoder
from models_proposed import TCNAE, adaptive_alarm

def evaluate_all():
    config.set_seed()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("="*75)
    print("EVALUATION OF ANOMALY DETECTION MODELS ON NPPAD TEST SET")
    print("="*75)

    # 1. Load preprocessed data and models metadata
    with open(os.path.join(config.DATA_DIR, 'preprocessed.pkl'), 'rb') as f:
        prep = pickle.load(f)
    with open(os.path.join(config.MODELS_DIR, 'baseline_info.json'), 'r') as f:
        base_info = json.load(f)
    with open(os.path.join(config.MODELS_DIR, 'proposed_info.json'), 'r') as f:
        prop_info = json.load(f)
    selected_sensors = np.load(os.path.join(config.MODELS_DIR, 'selected_sensors.npy'))

    test_scenarios = prep['test_scenarios']
    sensor_names = prep['sensor_names']
    n_features_all = len(sensor_names)
    n_features_sel = len(selected_sensors)
    window_size = config.WINDOW
    print(f"Loaded {len(test_scenarios)} test scenarios.")

    # 2. Load models into memory
    device = torch.device('cpu')

    # B1: PCA
    with open(os.path.join(config.MODELS_DIR, 'b1_pca.pkl'), 'rb') as f:
        b1_pca = pickle.load(f)

    # B2: FC-AE
    b2_model = FCAutoencoder(n_features_all, window_size, fc_hidden=config.FC_HIDDEN, latent_dim=config.LATENT_DIM)
    b2_model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'b2_fcae.pth'), map_location=device))
    b2_model.eval()

    # B3: Conv1D-AE
    b3_model = Conv1DAutoencoder(n_features_all, window_size, latent_dim=config.LATENT_DIM)
    b3_model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'b3_cnnae.pth'), map_location=device))
    b3_model.eval()

    # B4: LSTM-AE
    b4_model = LSTMAutoencoder(n_features_all, window_size, hidden_dim=config.LSTM_HIDDEN, num_layers=config.LSTM_LAYERS)
    b4_model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'b4_lstmae.pth'), map_location=device))
    b4_model.eval()

    # P0: TCN-AE (All Sensors)
    p0_model = TCNAE(n_features_all, window_size, channels=config.TCN_CHANNELS, kernel_size=config.TCN_KERNEL, dilations=config.TCN_DILATIONS, latent_dim=config.LATENT_DIM)
    p0_model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'p0_tcnae.pth'), map_location=device))
    p0_model.eval()

    # P1 & P2: TCN-AE (Selected Sensors)
    p1_model = TCNAE(n_features_sel, window_size, channels=config.TCN_CHANNELS, kernel_size=config.TCN_KERNEL, dilations=config.TCN_DILATIONS, latent_dim=config.LATENT_DIM)
    p1_model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, 'p1_p2_tcnae.pth'), map_location=device))
    p1_model.eval()

    model_list = [
        ('B1: PCA', 'B1', base_info['param_counts']['B1_PCA']),
        ('B2: FC-AE', 'B2', base_info['param_counts']['B2_FCAE']),
        ('B3: Conv1D-AE', 'B3', base_info['param_counts']['B3_CNNAE']),
        ('B4: LSTM-AE', 'B4', base_info['param_counts']['B4_LSTMAE']),
        ('P0: TCN-AE (All)', 'P0', prop_info['P0']['params']),
        ('P1: TCN-AE (Selected)', 'P1', prop_info['P1']['params']),
        ('P2: Proposed (TCN+Adaptive)', 'P2', prop_info['P2']['params']),
    ]

    # Helper function to compute residuals on a batch of windows
    def get_residuals(model_id, windows):
        if len(windows) == 0:
            return np.empty((0,))
        if model_id == 'B1':
            flat = windows.reshape(len(windows), -1)
            recon = b1_pca.inverse_transform(b1_pca.transform(flat))
            return np.mean((flat - recon) ** 2, axis=1)
        
        # PyTorch models
        if model_id == 'B2':
            mod = b2_model
            data_in = windows
        elif model_id == 'B3':
            mod = b3_model
            data_in = windows
        elif model_id == 'B4':
            mod = b4_model
            data_in = windows
        elif model_id == 'P0':
            mod = p0_model
            data_in = windows
        elif model_id in ['P1', 'P2']:
            mod = p1_model
            data_in = windows[:, :, selected_sensors]
        
        res_list = []
        bs = 256
        with torch.no_grad():
            for i in range(0, len(data_in), bs):
                bx = torch.tensor(data_in[i:i+bs], dtype=torch.float32)
                r = mod(bx)
                mse = torch.mean((bx - r) ** 2, dim=(1, 2))
                res_list.append(mse.numpy())
        return np.concatenate(res_list, axis=0)

    # 3. Evaluate each model across all scenarios
    results_records = []
    scenario_details = {m_tag: [] for _, m_tag, _ in model_list}

    for name, m_tag, p_count in model_list:
        print(f"\nEvaluating {name}...")
        all_window_labels = []
        all_residuals = []
        total_normal_windows = 0
        total_false_alarms = 0
        
        scen_metrics = []

        for scen in test_scenarios:
            wins = scen['windows']
            w_labels = scen['window_labels']
            onset_step = scen['onset_step']
            sev = scen['severity']
            sev_bin = scen['severity_bin']
            is_fault_scen = (onset_step is not None and onset_step >= 0)

            res = get_residuals(m_tag, wins)
            all_residuals.extend(res)
            all_window_labels.extend(w_labels)

            # Apply alarm logic
            if m_tag == 'P2':
                k_val = prop_info['P2']['calibrated_k']
                alpha_val = prop_info['P2']['alpha']
                win_val = prop_info['P2']['window']
                pers_val = prop_info['P2']['persistence']
                alarms, smoothed, dyn_thresh = adaptive_alarm(res, alpha=alpha_val, window=win_val, k=k_val, persist=pers_val)
            else:
                if m_tag in ['B1', 'B2', 'B3', 'B4']:
                    tag_map = {'B1': 'B1_PCA', 'B2': 'B2_FCAE', 'B3': 'B3_CNNAE', 'B4': 'B4_LSTMAE'}
                    thresh = base_info['thresholds'][tag_map[m_tag]]
                elif m_tag == 'P0':
                    thresh = prop_info['P0']['threshold']
                elif m_tag == 'P1':
                    thresh = prop_info['P1']['threshold']
                alarms = (res > thresh)
                dyn_thresh = np.full_like(res, thresh)
                smoothed = res

            # False alarms on normal segment
            norm_indices = np.where(w_labels == 0)[0]
            n_norm = len(norm_indices)
            n_fa = np.sum(alarms[norm_indices]) if n_norm > 0 else 0
            total_normal_windows += n_norm
            total_false_alarms += n_fa

            # Detection delay and detection success
            detected = False
            delay = np.nan
            first_alarm_idx = -1

            if is_fault_scen:
                # First window touching fault
                fault_start_win = max(0, onset_step - window_size + 1)
                alarm_indices = np.where(alarms)[0]
                # Alarm after or at onset
                post_onset_alarms = [a for a in alarm_indices if a >= fault_start_win]
                if len(post_onset_alarms) > 0:
                    first_alarm_idx = post_onset_alarms[0]
                    detected = True
                    delay = float(first_alarm_idx - fault_start_win)

            scen_res = {
                'accident_type': scen['accident_type'],
                'severity': sev,
                'severity_bin': sev_bin,
                'is_fault': is_fault_scen,
                'detected': detected,
                'delay': delay,
                'false_alarms': n_fa,
                'normal_steps': n_norm,
                'residuals': res,
                'smoothed': smoothed,
                'thresholds': dyn_thresh,
                'alarms': alarms,
                'onset_win': max(0, onset_step - window_size + 1) if is_fault_scen else -1,
                'first_alarm_idx': first_alarm_idx
            }
            scen_metrics.append(scen_res)

        scenario_details[m_tag] = scen_metrics

        # Compute overall ROC-AUC
        all_window_labels_arr = np.array(all_window_labels)
        all_residuals_arr = np.array(all_residuals)
        try:
            auc = roc_auc_score(all_window_labels_arr, all_residuals_arr)
        except Exception:
            auc = np.nan

        far_overall = (total_false_alarms / total_normal_windows * 1000.0) if total_normal_windows > 0 else 0.0

        # Aggregations: Overall, and by Severity Bin
        for s_bin in ['Overall', 'Low', 'Medium', 'High']:
            if s_bin == 'Overall':
                sub_scens = [s for s in scen_metrics if s['is_fault']]
            else:
                sub_scens = [s for s in scen_metrics if s['is_fault'] and s['severity_bin'] == s_bin]
            
            n_scen = len(sub_scens)
            if n_scen > 0:
                det_count = sum(1 for s in sub_scens if s['detected'])
                det_rate = (det_count / n_scen) * 100.0
                detected_delays = [s['delay'] for s in sub_scens if s['detected'] and not np.isnan(s['delay'])]
                mean_delay = np.mean(detected_delays) if len(detected_delays) > 0 else np.nan
            else:
                det_rate = np.nan
                mean_delay = np.nan

            results_records.append({
                'Model': name,
                'Model_ID': m_tag,
                'Params': p_count,
                'Bin': s_bin,
                'Scenarios': n_scen,
                'Detection_Rate_%': round(det_rate, 2),
                'False_Alarm_Rate_per_1k': round(far_overall, 2),
                'Mean_Delay_steps': round(mean_delay, 2) if not np.isnan(mean_delay) else np.nan,
                'ROC_AUC': round(auc, 4)
            })

    df_results = pd.DataFrame(results_records)

    # 4. Save results_table.csv and results_table.md
    csv_path = os.path.join(config.RESULTS_DIR, 'results_table.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Build Markdown Table (Overall summary view)
    df_overall = df_results[df_results['Bin'] == 'Overall'].copy()
    md_lines = [
        "# Benchmark Results: NPPAD Anomaly Detection",
        "",
        "### Overall Model Comparison",
        "| Model | Params | ROC-AUC | Detection Rate (%) | False Alarm Rate (/1k) | Mean Delay (steps) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for _, row in df_overall.iterrows():
        md_lines.append(f"| **{row['Model']}** | {row['Params']:,} | {row['ROC_AUC']:.4f} | {row['Detection_Rate_%']:.2f}% | {row['False_Alarm_Rate_per_1k']:.2f} | {row['Mean_Delay_steps']:.2f} |")

    md_lines.extend([
        "",
        "### Performance Breakdown by Severity Bin",
        "| Model | Low Det Rate (%) | Med Det Rate (%) | High Det Rate (%) | Low Delay | Med Delay | High Delay |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for _, m_tag, _ in model_list:
        sub = df_results[df_results['Model_ID'] == m_tag]
        m_name = sub.iloc[0]['Model']
        low_dr = sub[sub['Bin'] == 'Low']['Detection_Rate_%'].values[0]
        med_dr = sub[sub['Bin'] == 'Medium']['Detection_Rate_%'].values[0]
        hi_dr = sub[sub['Bin'] == 'High']['Detection_Rate_%'].values[0]
        
        low_del = sub[sub['Bin'] == 'Low']['Mean_Delay_steps'].values[0]
        med_del = sub[sub['Bin'] == 'Medium']['Mean_Delay_steps'].values[0]
        hi_del = sub[sub['Bin'] == 'High']['Mean_Delay_steps'].values[0]

        md_lines.append(f"| {m_name} | {low_dr:.1f}% | {med_dr:.1f}% | {hi_dr:.1f}% | {low_del:.1f} | {med_del:.1f} | {hi_del:.1f} |")

    md_path = os.path.join(config.RESULTS_DIR, 'results_table.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    print(f"Saved markdown summary to {md_path}")

    # ---------------------------------------------------------
    # Generate Plots
    # ---------------------------------------------------------
    print("\nGenerating evaluation plots...")

    # Find best baseline based on overall detection rate and ROC-AUC
    base_rows = df_overall[df_overall['Model_ID'].isin(['B1', 'B2', 'B3', 'B4'])]
    best_base_tag = base_rows.sort_values(by=['Detection_Rate_%', 'ROC_AUC'], ascending=False).iloc[0]['Model_ID']
    best_base_name = [name for name, tag, _ in model_list if tag == best_base_tag][0]

    # PLOT 1: Best baseline vs P2 on a medium-severity test scenario
    # Find a representative Medium severity scenario (e.g., LOCA ~50% or SGATR ~50%)
    med_candidates = [i for i, s in enumerate(scenario_details['P2']) if s['is_fault'] and s['severity_bin'] == 'Medium']
    scen_idx = med_candidates[0] if len(med_candidates) > 0 else 0
    scen_info = scenario_details['P2'][scen_idx]
    base_scen_info = scenario_details[best_base_tag][scen_idx]

    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Best Baseline
    plt.subplot(2, 1, 1)
    t_steps = np.arange(len(base_scen_info['residuals']))
    plt.plot(t_steps, base_scen_info['residuals'], label='Reconstruction Residual (MSE)', color='#1f77b4', lw=1.5)
    plt.plot(t_steps, base_scen_info['thresholds'], label='Fixed Threshold (99th %tile)', color='#d62728', linestyle='--', lw=1.5)
    plt.axvline(x=base_scen_info['onset_win'], color='black', linestyle=':', lw=2, label=f"Onset (Window {base_scen_info['onset_win']})")
    if base_scen_info['first_alarm_idx'] >= 0:
        plt.scatter([base_scen_info['first_alarm_idx']], [base_scen_info['residuals'][base_scen_info['first_alarm_idx']]],
                    color='red', marker='o', s=80, zorder=5, label=f"Alarm Raised (Win {base_scen_info['first_alarm_idx']})")
    plt.title(f"Baseline ({best_base_name}) — Scenario: {scen_info['accident_type']} ({scen_info['severity']:.1f}% Severity)", fontsize=11, fontweight='bold')
    plt.ylabel("Residual MSE")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=9)

    # Subplot 2: Proposed P2
    plt.subplot(2, 1, 2)
    plt.plot(t_steps, scen_info['smoothed'], label='EWMA Smoothed Residual', color='#2ca02c', lw=1.5)
    plt.plot(t_steps, scen_info['thresholds'], label='Adaptive Rolling MAD Threshold', color='#e377c2', linestyle='--', lw=1.5)
    plt.axvline(x=scen_info['onset_win'], color='black', linestyle=':', lw=2, label=f"Onset (Window {scen_info['onset_win']})")
    if scen_info['first_alarm_idx'] >= 0:
        plt.scatter([scen_info['first_alarm_idx']], [scen_info['smoothed'][scen_info['first_alarm_idx']]],
                    color='crimson', marker='X', s=100, zorder=5, label=f"Alarm Raised (Win {scen_info['first_alarm_idx']})")
    plt.title(f"Proposed P2 (TCN-AE + Adaptive Threshold + Persistence)", fontsize=11, fontweight='bold')
    plt.xlabel("Window Index")
    plt.ylabel("Residual / Threshold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plot1_path = os.path.join(config.RESULTS_DIR, 'plot_residuals.png')
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"Saved Plot 1 to {plot1_path}")

    # PLOT 2: Bar chart comparing False Alarm Rate and Detection Delay across all models
    fig, ax1 = plt.subplots(figsize=(11, 5))
    x_pos = np.arange(len(model_list))
    width = 0.35

    fars = [df_overall[df_overall['Model_ID'] == tag]['False_Alarm_Rate_per_1k'].values[0] for _, tag, _ in model_list]
    delays = [df_overall[df_overall['Model_ID'] == tag]['Mean_Delay_steps'].values[0] for _, tag, _ in model_list]

    bars1 = ax1.bar(x_pos - width/2, fars, width, label='False Alarm Rate (per 1k)', color='#d62728', alpha=0.85, edgecolor='black')
    ax1.set_ylabel('False Alarms / 1000 Steps', color='#d62728', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#d62728')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_pos + width/2, delays, width, label='Mean Detection Delay (steps)', color='#1f77b4', alpha=0.85, edgecolor='black')
    ax2.set_ylabel('Mean Delay (Windows)', color='#1f77b4', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name for name, _, _ in model_list], rotation=25, ha='right', fontsize=9, fontweight='bold')
    ax1.set_title("Operational Trade-off: False Alarm Rate vs Detection Delay Across Models", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot2_path = os.path.join(config.RESULTS_DIR, 'plot_bar_metrics.png')
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"Saved Plot 2 to {plot2_path}")

    # PLOT 3: Detection rate by severity bin for all models
    plt.figure(figsize=(12, 5))
    bins = ['Low', 'Medium', 'High', 'Overall']
    bar_w = 0.12
    x_base = np.arange(len(bins))

    palette = ['#aec7e8', '#ffbb78', '#98df8a', '#c5b0d5', '#ff9896', '#17becf', '#2ca02c']
    for idx, (name, tag, _) in enumerate(model_list):
        sub = df_results[df_results['Model_ID'] == tag]
        rates = [sub[sub['Bin'] == b]['Detection_Rate_%'].values[0] for b in bins]
        offset = (idx - len(model_list)/2 + 0.5) * bar_w
        plt.bar(x_base + offset, rates, bar_w, label=name, color=palette[idx % len(palette)], edgecolor='black')

    plt.xticks(x_base, [f"{b} Severity" if b != 'Overall' else 'Overall' for b in bins], fontsize=11, fontweight='bold')
    plt.ylabel("Detection Rate (%)", fontsize=11, fontweight='bold')
    plt.ylim(0, 105)
    plt.title("Detection Rate by Accident Severity Bin Across Models", fontsize=13, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plot3_path = os.path.join(config.RESULTS_DIR, 'plot_detection_rate.png')
    plt.savefig(plot3_path, dpi=150)
    plt.close()
    print(f"Saved Plot 3 to {plot3_path}")
    print("\nAll evaluation artifacts and plots created successfully!")

if __name__ == '__main__':
    evaluate_all()
