# config.py — central hyperparameter store
import os, random, numpy as np, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')
MODELS_DIR = os.path.join(ROOT, 'models')

SEED = 42
WINDOW = 30
STRIDE = 1
MAX_STEPS = 600  # downsample target per scenario
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# TEST_RATIO = 0.15

# Model hyperparameters
LATENT_DIM = 16
FC_HIDDEN = 64
LSTM_HIDDEN = 32
LSTM_LAYERS = 1
CNN_CHANNELS = [32, 16]
TCN_CHANNELS = [16, 8]  # per block
TCN_KERNEL = 3
TCN_DILATIONS = [1, 2]

EPOCHS_FULL = 30
EPOCHS_QUICK = 5
BATCH_SIZE = 64
LR = 1e-3
EARLY_STOP_PATIENCE = 5

# Alarm logic
FIXED_THRESH_PERCENTILE = 99  # percentile of val normal residuals
EWMA_ALPHA = 0.3
ROLLING_WINDOW = 100
MAD_K = 3.0  # will be calibrated
PERSISTENCE = 3  # consecutive steps above threshold

# Sensor selection
CORR_THRESH = 0.98
NOISY_PERCENTILE = 50  # drop noisiest 50%

# Quick mode flags
QUICK_MODE = os.environ.get('NPPAD_QUICK', '0') == '1'
QUICK_MAX_SCENARIOS = 10
if QUICK_MODE:
    EPOCHS_FULL = EPOCHS_QUICK

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
