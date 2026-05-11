
#######
# Configuration
#######

import torch


DATA_PATH = "./extrassign/ander/data/"
PLOTS_PATH = "./extrassign/ander/docs/plots/"
BEST_MODEL_PATH = "./extrassign/ander/models/best_vae.pt"
OPTUNA_DB_URL = "sqlite:///extrassign/ander/vae_mnist.db"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

N_EPOCHS = 1
INPUT_DIM = 28 * 28
TRIALS = 1
N_STARTUP_TRIALS = 15
WARMUP_EPOCHS = 5
VISUALIZE_PLOTS = True