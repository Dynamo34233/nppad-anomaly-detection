"""
models_proposed.py — Proposed TCN-AE and sensor selection pipeline.
Includes:
a) Sensor selection: variance filtering, correlation pruning, AE reconstruction noise ranking.
b) Dilated Causal TCN-AE architecture.
c) Three variants:
   - P0: TCN-AE with ALL sensors + Fixed Threshold
   - P1: TCN-AE with SELECTED sensors + Fixed Threshold
   - P2: TCN-AE with SELECTED sensors + EWMA Smoothing + Adaptive MAD Threshold + Persistence
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import config

def get_device():
    return torch.device('cpu')

# -------------------------------------------------------------
# Causal TCN Architecture
# -------------------------------------------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.out_relu = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.chomp1(self.conv1(x)))
        out = self.relu2(self.chomp2(self.conv2(out)))
        return self.out_relu(out + self.res(x))


class TCNAE(nn.Module):
    def __init__(self, n_features, window_size, channels=[16, 8], kernel_size=3, dilations=[1, 2], latent_dim=16):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim

        # Encoder blocks
        enc_blocks = []
        in_ch = n_features
        for d in dilations:
            enc_blocks.append(TCNBlock(in_ch, channels[0], kernel_size=kernel_size, dilation=d))
            in_ch = channels[0]
        enc_blocks.append(TCNBlock(channels[0], channels[1], kernel_size=kernel_size, dilation=1))
        self.encoder = nn.Sequential(*enc_blocks)
        
        self.enc_fc = nn.Linear(channels[1] * window_size, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, channels[1] * window_size)

        # Decoder blocks
        dec_blocks = []
        in_ch = channels[1]
        for d in reversed(dilations):
            dec_blocks.append(TCNBlock(in_ch, channels[0], kernel_size=kernel_size, dilation=d))
            in_ch = channels[0]
        self.decoder = nn.Sequential(*dec_blocks)
        self.out_conv = nn.Conv1d(channels[0], n_features, kernel_size=1)

    def forward(self, x):
        # x: (B, W, F) -> permute to (B, F, W)
        b, w, f = x.shape
        x_perm = x.permute(0, 2, 1)
        feat = self.encoder(x_perm)
        flat = feat.view(b, -1)
        z = self.enc_fc(flat)

        dec_flat = self.dec_fc(z)
        dec_feat = dec_flat.view(b, -1, w)
        dec_out = self.decoder(dec_feat)
        out_perm = self.out_conv(dec_out)
        return out_perm.permute(0, 2, 1)  # (B, W, F)


# -------------------------------------------------------------
# Sensor Selection Logic (Training Normal Data Only)
# -------------------------------------------------------------

def select_sensors(train_windows, val_windows, sensor_names):
    """
    1. Drop near-constant sensors (std < 1e-4 after scaling)
    2. Drop redundant sensors (|corr| > 0.98)
    3. Quick FC AE to rank per-sensor reconstruction noise, drop noisiest 50%
    """
    print("\n--- Running Sensor Selection (Training Normal Data Only) ---")
    n_total = len(sensor_names)
    X_train_steps = train_windows.reshape(-1, n_total)
    
    # 1. Near-constant filtering
    stds = np.std(X_train_steps, axis=0)
    var_mask = stds >= 1e-4
    var_indices = np.where(var_mask)[0]
    print(f"Step 1 (Variance): Retained {len(var_indices)}/{n_total} sensors (dropped {n_total - len(var_indices)} near-constant)")

    # 2. Correlation pruning
    corr_data = X_train_steps[:, var_indices]
    corr_matrix = np.corrcoef(corr_data, rowvar=False)
    corr_thresh = config.CORR_THRESH
    to_drop_rel = set()

    for i in range(len(var_indices)):
        if i in to_drop_rel:
            continue
        for j in range(i + 1, len(var_indices)):
            if j in to_drop_rel:
                continue
            if abs(corr_matrix[i, j]) > corr_thresh:
                to_drop_rel.add(j)

    uncorr_indices = [var_indices[i] for i in range(len(var_indices)) if i not in to_drop_rel]
    print(f"Step 2 (Correlation |r| <= {corr_thresh}): Retained {len(uncorr_indices)} sensors (dropped {len(to_drop_rel)} redundant)")

    # 3. Quick FC-AE for noise ranking
    X_train_uncorr = train_windows[:, :, uncorr_indices]
    X_val_uncorr = val_windows[:, :, uncorr_indices]
    
    class QuickAE(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, 12))
            self.dec = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, d_in))
        def forward(self, x):
            return self.dec(self.enc(x))

    d_in = len(uncorr_indices) * config.WINDOW
    q_model = QuickAE(d_in).to(get_device())
    opt = torch.optim.Adam(q_model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    sub_n = min(2000, len(X_train_uncorr))
    q_train = torch.tensor(X_train_uncorr[:sub_n].reshape(sub_n, -1), dtype=torch.float32).to(get_device())
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn(q_model(q_train), q_train)
        loss.backward()
        opt.step()

    # Per-sensor reconstruction variance on val normal data
    q_model.eval()
    with torch.no_grad():
        v_sub = min(1000, len(X_val_uncorr))
        q_val = torch.tensor(X_val_uncorr[:v_sub].reshape(v_sub, -1), dtype=torch.float32).to(get_device())
        recon = q_model(q_val).cpu().numpy().reshape(v_sub, config.WINDOW, len(uncorr_indices))
        orig = X_val_uncorr[:v_sub]
        # Per sensor MSE / variance
        sensor_noise = np.mean((orig - recon) ** 2, axis=(0, 1))

    # Drop noisiest 50%
    median_noise = np.percentile(sensor_noise, config.NOISY_PERCENTILE)
    final_selected_rel = np.where(sensor_noise <= median_noise)[0]
    final_indices = [uncorr_indices[i] for i in final_selected_rel]
    print(f"Step 3 (Reconstruction Noise): Retained top 50% least noisy -> {len(final_indices)} sensors final")

    # Save selected indices
    np.save(os.path.join(config.MODELS_DIR, 'selected_sensors.npy'), np.array(final_indices))
    
    # Plot kept vs dropped
    plt.figure(figsize=(10, 5))
    all_sensor_status = np.zeros(n_total)
    all_sensor_status[var_indices] = 1
    all_sensor_status[uncorr_indices] = 2
    all_sensor_status[final_indices] = 3
    
    colors = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c']
    labels = ['Dropped: Constant', 'Dropped: High Corr', 'Dropped: High Noise', 'Kept: Final Selected']
    counts = [np.sum(all_sensor_status == i) for i in range(4)]
    
    plt.bar(labels, counts, color=colors, edgecolor='black', alpha=0.85)
    plt.ylabel("Number of Sensors", fontsize=11)
    plt.title(f"Sensor Selection Funnel (Initial {n_total} -> Final {len(final_indices)})", fontsize=13, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    for i, c in enumerate(counts):
        plt.text(i, c + 0.5, str(c), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'sensor_selection.png'), dpi=150)
    plt.close()
    print(f"Saved sensor selection plot to results/sensor_selection.png")

    return final_indices


# -------------------------------------------------------------
# Adaptive Threshold Logic & Calibration
# -------------------------------------------------------------

def ewma(series, alpha=0.3):
    s = np.zeros_like(series)
    s[0] = series[0]
    for t in range(1, len(series)):
        s[t] = alpha * series[t] + (1 - alpha) * s[t-1]
    return s

def adaptive_alarm(residuals, alpha=0.3, window=100, k=3.0, persist=3):
    smoothed = ewma(residuals, alpha)
    N = len(residuals)
    alarm_flags = np.zeros(N, dtype=bool)
    thresholds = np.zeros(N, dtype=float)
    clean_history = []
    consecutive_above = 0
    
    for t in range(N):
        if len(clean_history) >= 10:
            hist_win = np.array(clean_history[-window:])
            med = float(np.median(hist_win))
            mad = float(np.median(np.abs(hist_win - med)))
            scale = 1.4826 * max(mad, 1e-4)
            thresh = med + k * scale
        else:
            thresh = 1e9  # burn-in protection
            
        thresholds[t] = thresh
        is_above = (smoothed[t] > thresh)
        
        if is_above:
            consecutive_above += 1
        else:
            consecutive_above = 0
            
        if consecutive_above >= persist:
            alarm_flags[t] = True
        else:
            alarm_flags[t] = False
            clean_history.append(float(smoothed[t]))
            
    return alarm_flags, smoothed, thresholds

def calibrate_k(val_residuals, alpha=0.3, window=100, persist=3):
    print("\n--- Calibrating Adaptive Threshold k on Val Normal Residuals ---")
    best_k = 6.0
    test_range = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]
    for test_k in test_range:
        alarms, _, _ = adaptive_alarm(val_residuals, alpha=alpha, window=window, k=test_k, persist=persist)
        false_alarms = np.sum(alarms)
        far_per_1000 = (false_alarms / len(val_residuals)) * 1000.0
        print(f"  k = {test_k:4.1f} -> False Alarms: {false_alarms:3d}/{len(val_residuals)} ({far_per_1000:.2f} per 1000 steps)")
        if far_per_1000 < 1.0:
            best_k = test_k
            break
        best_k = test_k
    print(f"Calibrated k: {best_k} (FAR < 1.0 per 1000 steps)")
    return best_k


# -------------------------------------------------------------
# Main Training & Proposed Variants Workflow
# -------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_torch_model(model, train_loader, val_loader, epochs, lr, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = get_device()
    model.to(device)

    best_val_loss = float('inf')
    best_weights = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                val_loss += loss.item() * len(batch_x)
        val_loss /= len(val_loader.dataset)

        print(f"  Epoch {epoch:2d}/{epochs:2d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping triggered at epoch {epoch}")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model

def compute_residuals(model, data_windows):
    model.eval()
    device = get_device()
    residuals = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(data_windows), bs):
            bx = torch.tensor(data_windows[i:i+bs], dtype=torch.float32).to(device)
            recon = model(bx)
            mse = torch.mean((bx - recon) ** 2, dim=(1, 2))
            residuals.append(mse.cpu().numpy())
    if len(residuals) == 0:
        return np.empty((0,))
    return np.concatenate(residuals, axis=0)


def train_proposed():
    config.set_seed()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    with open(os.path.join(config.DATA_DIR, 'preprocessed.pkl'), 'rb') as f:
        prep = pickle.load(f)

    X_train_all = prep['train_normal_windows']
    X_val_all = prep['val_normal_windows']
    sensor_names = prep['sensor_names']
    n_features_all = len(sensor_names)
    window_size = config.WINDOW

    print("="*75)
    print("PROPOSED TCN-AE & ABLATION VARIANTS (P0, P1, P2)")
    print("="*75)

    # 1. Run Sensor Selection
    selected_indices = select_sensors(X_train_all, X_val_all, sensor_names)
    X_train_sel = X_train_all[:, :, selected_indices]
    X_val_sel = X_val_all[:, :, selected_indices]
    n_features_sel = len(selected_indices)

    epochs = config.EPOCHS_QUICK if config.QUICK_MODE else config.EPOCHS_FULL
    max_train_samples = 4000 if not config.QUICK_MODE else 1500
    if len(X_train_all) > max_train_samples:
        rng = np.random.RandomState(config.SEED)
        sub_idx = rng.choice(len(X_train_all), max_train_samples, replace=False)
        X_train_all_sub = X_train_all[sub_idx]
        X_train_sel_sub = X_train_sel[sub_idx]
    else:
        X_train_all_sub = X_train_all
        X_train_sel_sub = X_train_sel

    # DataLoaders for all sensors
    loader_train_all = DataLoader(TensorDataset(torch.tensor(X_train_all_sub, dtype=torch.float32)), batch_size=config.BATCH_SIZE, shuffle=True)
    loader_val_all = DataLoader(TensorDataset(torch.tensor(X_val_all, dtype=torch.float32)), batch_size=config.BATCH_SIZE, shuffle=False)

    # DataLoaders for selected sensors
    loader_train_sel = DataLoader(TensorDataset(torch.tensor(X_train_sel_sub, dtype=torch.float32)), batch_size=config.BATCH_SIZE, shuffle=True)
    loader_val_sel = DataLoader(TensorDataset(torch.tensor(X_val_sel, dtype=torch.float32)), batch_size=config.BATCH_SIZE, shuffle=False)

    proposed_info = {}

    # ---------------------------------------------------------
    # P0: TCN-AE (All Sensors, Fixed Threshold)
    # ---------------------------------------------------------
    print("\n--- Training P0: TCN-AE (All Sensors) ---")
    p0_model = TCNAE(n_features_all, window_size, channels=config.TCN_CHANNELS, kernel_size=config.TCN_KERNEL, dilations=config.TCN_DILATIONS, latent_dim=config.LATENT_DIM)
    p0_params = count_parameters(p0_model)
    print(f"P0 TCN-AE Parameter count: {p0_params:,}")

    p0_model = train_torch_model(p0_model, loader_train_all, loader_val_all, epochs, config.LR, patience=config.EARLY_STOP_PATIENCE)
    p0_val_res = compute_residuals(p0_model, X_val_all)
    p0_thresh = float(np.percentile(p0_val_res, config.FIXED_THRESH_PERCENTILE))
    torch.save(p0_model.state_dict(), os.path.join(config.MODELS_DIR, 'p0_tcnae.pth'))

    proposed_info['P0'] = {
        'threshold': p0_thresh,
        'params': int(p0_params),
        'n_features': int(n_features_all)
    }

    # ---------------------------------------------------------
    # P1 & P2: TCN-AE (Selected Sensors)
    # ---------------------------------------------------------
    print(f"\n--- Training P1/P2: TCN-AE (Selected {n_features_sel} Sensors) ---")
    p1_model = TCNAE(n_features_sel, window_size, channels=config.TCN_CHANNELS, kernel_size=config.TCN_KERNEL, dilations=config.TCN_DILATIONS, latent_dim=config.LATENT_DIM)
    p1_params = count_parameters(p1_model)
    print(f"P1/P2 TCN-AE Parameter count: {p1_params:,}")

    p1_model = train_torch_model(p1_model, loader_train_sel, loader_val_sel, epochs, config.LR, patience=config.EARLY_STOP_PATIENCE)
    p1_val_res = compute_residuals(p1_model, X_val_sel)
    p1_thresh = float(np.percentile(p1_val_res, config.FIXED_THRESH_PERCENTILE))
    torch.save(p1_model.state_dict(), os.path.join(config.MODELS_DIR, 'p1_p2_tcnae.pth'))

    # Calibrate adaptive threshold parameter k on val normal residuals for P2
    calibrated_k = calibrate_k(p1_val_res, alpha=config.EWMA_ALPHA, window=config.ROLLING_WINDOW, persist=config.PERSISTENCE)

    proposed_info['P1'] = {
        'threshold': p1_thresh,
        'params': int(p1_params),
        'n_features': int(n_features_sel)
    }
    proposed_info['P2'] = {
        'threshold': 'adaptive',
        'calibrated_k': float(calibrated_k),
        'alpha': float(config.EWMA_ALPHA),
        'window': int(config.ROLLING_WINDOW),
        'persistence': int(config.PERSISTENCE),
        'params': int(p1_params),
        'n_features': int(n_features_sel)
    }

    with open(os.path.join(config.MODELS_DIR, 'proposed_info.json'), 'w') as f:
        json.dump(proposed_info, f, indent=2)

    print("\nProposed models training and calibration complete! Saved to models/.")

if __name__ == '__main__':
    train_proposed()
