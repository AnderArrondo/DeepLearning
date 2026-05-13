from sklearn.model_selection import train_test_split

#RADEON
#import os
#os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.1.0"

import pandas as pd
import torch
import optuna

import config
import utils

config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(config.DEVICE)

#OPTUNA
study_name=input("Enter study name:")
config.STUDY_NAME=study_name
if config.optimize_hyperparams:
    sampler=optuna.samplers.TPESampler(seed=config.SEED)
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)
    study=optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name, pruner=pruner)
    study.optimize(utils.objective_function,n_trials=config.n_trials,show_progress_bar=True,n_jobs=-1)

    print(
    f"""\n--- Optimization Finished ---\n
    {config.n_trials} trials\n
    Best Score: {study.best_value:.4f}\n
    Best Hyperparameters:""")

    for key, value in study.best_params.items():
        print(f"  {key}: {value}")