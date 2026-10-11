"""
models_baseline.py — Implementation and training of baseline anomaly detection models (B1-B4).
All baselines utilize all active plant operational sensors.
B1: PCA (95% variance retained)
B2: Fully-Connected Autoencoder (FC-AE)
B3: 1D-CNN Autoencoder (Conv1D-AE)
B4: LSTM Autoencoder (LSTM-AE)
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
import config

def get_device():
    return torch.device('cpu')

# -------------------------------------------------------------
# PyTorch Baseline Architectures
# -------------------------------------------------------------

class FCAutoencoder(nn.Module):
    def __init__(self, n_features, window_size, fc_hidden=64, latent_dim=16):
        super().__init__()
        in_dim = n_features * window_size
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, in_dim)
        )

    def forward(self, x):
        # x: (B, W, F)
        b, w, f = x.shape
        flat = x.view(b, -1)
        z = self.encoder(flat)
        recon_flat = self.decoder(z)
        return recon_flat.view(b, w, f)


class Conv1DAutoencoder(nn.Module):
    def __init__(self, n_features, window_size, latent_dim=16):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim

        # Input to Conv1d: (B, in_channels=n_features, length=window_size)
        self.enc_conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.enc_relu1 = nn.ReLU()
        self.enc_conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.enc_relu2 = nn.ReLU()
        self.enc_fc = nn.Linear(16 * window_size, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 16 * window_size)
        self.dec_conv1 = nn.ConvTranspose1d(16, 32, kernel_size=3, padding=1)
        self.dec_relu1 = nn.ReLU()
        self.dec_conv2 = nn.ConvTranspose1d(32, n_features, kernel_size=3, padding=1)

    def forward(self, x):
        # x: (B, W, F) -> permute to (B, F, W)
        b, w, f = x.shape
        x_perm = x.permute(0, 2, 1)
        
        c1 = self.enc_relu1(self.enc_conv1(x_perm))
        c2 = self.enc_relu2(self.enc_conv2(c1))
        flat = c2.view(b, -1)
        z = self.enc_fc(flat)

        dec_flat = self.dec_fc(z)
        dec_unflat = dec_flat.view(b, 16, w)
        t1 = self.dec_relu1(self.dec_conv1(dec_unflat))
        out_perm = self.dec_conv2(t1)  # (B, F, W)
        
        return out_perm.permute(0, 2, 1)  # (B, W, F)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, window_size, hidden_dim=32, num_layers=1):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: (B, W, F)
        b, w, f = x.shape
        _, (h_n, _) = self.encoder_lstm(x)
        # Take last layer's hidden state: (B, hidden_dim)
        z = h_n[-1]
        # Repeat hidden state for each time step in window: (B, W, hidden_dim)
        rep = z.unsqueeze(1).repeat(1, w, 1)
        dec_out, _ = self.decoder_lstm(rep)
        recon = self.fc_out(dec_out)  # (B, W, F)
        return recon


# -------------------------------------------------------------
# Training & Evaluation Helpers
# -------------------------------------------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_torch_ae(model, train_loader, val_loader, epochs, lr, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    device = get_device()
    model.to(device)

    best_val_loss = float('inf')
    best_weights = None
    no_improve_epochs = 0

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

        print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"  Early stopping triggered at epoch {epoch} (best val loss: {best_val_loss:.6f})")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model

def compute_torch_residuals(model, data_windows):
    """
    Computes per-window MSE residual for PyTorch model.
    data_windows: (N, W, F) numpy array
    returns: (N,) numpy array of residuals
    """
    model.eval()
    device = get_device()
    residuals = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(data_windows), batch_size):
            batch = torch.tensor(data_windows[i:i+batch_size], dtype=torch.float32).to(device)
            recon = model(batch)
            mse = torch.mean((batch - recon) ** 2, dim=(1, 2))
            residuals.append(mse.cpu().numpy())
    if len(residuals) == 0:
        return np.empty((0,))
    return np.concatenate(residuals, axis=0)


def train_baselines():
    config.set_seed()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1. Load preprocessed data
    pkl_path = os.path.join(config.DATA_DIR, 'preprocessed.pkl')
    with open(pkl_path, 'rb') as f:
        prep = pickle.load(f)

    X_train = prep['train_normal_windows']  # (N_train, 30, F)
    X_val = prep['val_normal_windows']      # (N_val, 30, F)
    n_features = X_train.shape[2]
    window_size = config.WINDOW

    print("="*75)
    print("TRAINING BASELINE ANOMALY DETECTION MODELS (B1 - B4)")
    print(f"Number of sensors: {n_features} | Window size: {window_size}")
    print(f"Train windows: {len(X_train)} | Val windows: {len(X_val)}")
    print("="*75)

    # Subsample windows for training if needed to maintain fast execution
    max_train_samples = 4000 if not config.QUICK_MODE else 1500
    if len(X_train) > max_train_samples:
        rng = np.random.RandomState(config.SEED)
        idx_train = rng.choice(len(X_train), max_train_samples, replace=False)
        X_train_sub = X_train[idx_train]
    else:
        X_train_sub = X_train

    max_val_samples = 1500 if not config.QUICK_MODE else 800
    if len(X_val) > max_val_samples:
        rng = np.random.RandomState(config.SEED)
        idx_val = rng.choice(len(X_val), max_val_samples, replace=False)
        X_val_sub = X_val[idx_val]
    else:
        X_val_sub = X_val

    epochs = config.EPOCHS_QUICK if config.QUICK_MODE else config.EPOCHS_FULL
    thresholds = {}
    param_counts = {}

    # ---------------------------------------------------------
    # Baseline 1: PCA (95% variance)
    # ---------------------------------------------------------
    print("\n--- Training B1: PCA (95% variance) ---")
    X_train_flat = X_train_sub.reshape(len(X_train_sub), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)

    pca = PCA(n_components=0.95, random_state=config.SEED)
    pca.fit(X_train_flat)
    n_comp = pca.n_components_
    print(f"PCA components selected: {n_comp} (out of {X_train_flat.shape[1]})")

    # Reconstruct val data and compute threshold
    val_recon_flat = pca.inverse_transform(pca.transform(X_val_flat))
    b1_val_res = np.mean((X_val_flat - val_recon_flat) ** 2, axis=1)
    b1_thresh = float(np.percentile(b1_val_res, config.FIXED_THRESH_PERCENTILE))
    thresholds['B1_PCA'] = float(b1_thresh)
    param_counts['B1_PCA'] = int(n_comp * X_train_flat.shape[1])

    with open(os.path.join(config.MODELS_DIR, 'b1_pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)
    print(f"B1 PCA 99th percentile val threshold: {b1_thresh:.6f}")

    # Prepare PyTorch DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train_sub, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_sub, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # ---------------------------------------------------------
    # Baseline 2: Fully-Connected Autoencoder
    # ---------------------------------------------------------
    print("\n--- Training B2: Fully-Connected Autoencoder (FC-AE) ---")
    b2_model = FCAutoencoder(n_features, window_size, fc_hidden=config.FC_HIDDEN, latent_dim=config.LATENT_DIM)
    b2_params = count_parameters(b2_model)
    param_counts['B2_FCAE'] = b2_params
    print(f"B2 FC-AE Parameter count: {b2_params:,}")

    b2_model = train_torch_ae(b2_model, train_loader, val_loader, epochs, config.LR, patience=config.EARLY_STOP_PATIENCE)
    b2_val_res = compute_torch_residuals(b2_model, X_val)
    b2_thresh = float(np.percentile(b2_val_res, config.FIXED_THRESH_PERCENTILE))
    thresholds['B2_FCAE'] = b2_thresh
    torch.save(b2_model.state_dict(), os.path.join(config.MODELS_DIR, 'b2_fcae.pth'))
    print(f"B2 FC-AE 99th percentile val threshold: {b2_thresh:.6f}")

    # ---------------------------------------------------------
    # Baseline 3: 1D-CNN Autoencoder
    # ---------------------------------------------------------
    print("\n--- Training B3: 1D-CNN Autoencoder (Conv1D-AE) ---")
    b3_model = Conv1DAutoencoder(n_features, window_size, latent_dim=config.LATENT_DIM)
    b3_params = count_parameters(b3_model)
    param_counts['B3_CNNAE'] = b3_params
    print(f"B3 Conv1D-AE Parameter count: {b3_params:,}")

    b3_model = train_torch_ae(b3_model, train_loader, val_loader, epochs, config.LR, patience=config.EARLY_STOP_PATIENCE)
    b3_val_res = compute_torch_residuals(b3_model, X_val)
    b3_thresh = float(np.percentile(b3_val_res, config.FIXED_THRESH_PERCENTILE))
    thresholds['B3_CNNAE'] = b3_thresh
    torch.save(b3_model.state_dict(), os.path.join(config.MODELS_DIR, 'b3_cnnae.pth'))
    print(f"B3 Conv1D-AE 99th percentile val threshold: {b3_thresh:.6f}")

    # ---------------------------------------------------------
    # Baseline 4: LSTM Autoencoder
    # ---------------------------------------------------------
    print("\n--- Training B4: LSTM Autoencoder (LSTM-AE) ---")
    b4_model = LSTMAutoencoder(n_features, window_size, hidden_dim=config.LSTM_HIDDEN, num_layers=config.LSTM_LAYERS)
    b4_params = count_parameters(b4_model)
    param_counts['B4_LSTMAE'] = b4_params
    print(f"B4 LSTM-AE Parameter count: {b4_params:,}")

    b4_model = train_torch_ae(b4_model, train_loader, val_loader, epochs, config.LR, patience=config.EARLY_STOP_PATIENCE)
    b4_val_res = compute_torch_residuals(b4_model, X_val)
    b4_thresh = float(np.percentile(b4_val_res, config.FIXED_THRESH_PERCENTILE))
    thresholds['B4_LSTMAE'] = b4_thresh
    torch.save(b4_model.state_dict(), os.path.join(config.MODELS_DIR, 'b4_lstmae.pth'))
    print(f"B4 LSTM-AE 99th percentile val threshold: {b4_thresh:.6f}")

    # Save metadata & thresholds
    baseline_info = {
        'thresholds': {k: float(v) for k, v in thresholds.items()},
        'param_counts': {k: int(v) for k, v in param_counts.items()},
        'n_features': int(n_features),
        'window_size': int(window_size)
    }
    with open(os.path.join(config.MODELS_DIR, 'baseline_info.json'), 'w') as f:
        json.dump(baseline_info, f, indent=2)

    print("\nBaseline training completed successfully! Models and thresholds saved to models/.")

if __name__ == '__main__':
    train_baselines()
