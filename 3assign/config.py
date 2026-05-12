
import torch

ORIG_CSV: str = "./3assign/data/tweet_emotions.csv"
DATA_DIR: str = "./3assign/data/"
PLOTS_DIR: str = "./3assign/plots/"
RESULTS_DIR: str = "./3assign/results/"

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
MODEL: str = "roberta-large-mnli"
ZERO_SHOT_BATCH_SIZE: int = 256
SEED: int = 42
N_CLASSES: int = 12

#OPTUNA
optimize_hyperparams: bool = True
n_trials: int = 1
n_epochs: int = 1
