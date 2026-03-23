
import torch
import numpy as np

from models import *


class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.csv_path: str  = "./data/insurance.csv"
        self.show_plots: bool = False
        self.train_model: bool = True
        self.optimize_hyperparams: bool = True
        self.batch_size: int  = 32
        self.epochs: int  = 120
        self.lr: float= 6e-3
        self.random_seed: int  = 42
        self.val_trials: int = 50

        self.device: str = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )

        self.models = {
            "model1": InsuranceModel1,
            "model2": InsuranceModel2,
            "model3": InsuranceModel3,
            "model4": InsuranceModel4,
            "model5": InsuranceModel5,
            "model6": InsuranceModel6
        }
        self.best_model = "model4"

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
