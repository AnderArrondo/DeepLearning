
import torch

ORIG_CSV: str = "./3assign/data/tweet_emotions.csv"
DATA_DIR: str = "./3assign/data/"
PLOTS_DIR: str = "./3assign/plots/"

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
MODEL: str = "roberta-large-mnli"
ZERO_SHOT_BATCH_SIZE: int = 256
