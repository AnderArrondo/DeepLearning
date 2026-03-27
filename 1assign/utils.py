
from config import Config

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import optuna

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import NormalizedRootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# DATA VISUALIZATION
def visualize_distributions(
        df: pd.DataFrame
        ) -> None:
    fig, ax = plt.subplots(3, 2, figsize=(12, 8))

    sns.histplot(
        data=df, x="age",
        kde=True, 
        ax=ax[0, 0]
    )
    ax[0, 0].set_title("Age Distribution")

    sns.histplot(
        data=df, x="bmi",
        kde=True, ax=ax[0, 1]
    )
    ax[0, 1].set_title("BMI Distribution")

    sns.barplot(df["children"].value_counts(), ax=ax[1, 0], color="royalblue")
    ax[1, 0].set_title("Children Distribution")

    sns.barplot(df["smoker"].value_counts(), ax=ax[1, 1], color="royalblue")
    ax[1, 1].set_title("Smoker Distribution")

    sns.barplot(df["region"].value_counts(), ax=ax[2, 0], color="royalblue")
    ax[2, 0].set_title("Region Distribution")

    sns.histplot(
        data=df, x="charges", kde=True, ax=ax[2, 1],
        color="royalblue"
    )
    ax[2, 1].set_title("Charges Distribution")
    
    sns.set_theme(style="ticks")

    g = sns.pairplot(
        df, 
        vars=["age", "bmi", "charges"], 
        hue="smoker", 
        markers=["o", "s"],
        plot_kws={'alpha': 0.25}, 
        corner=True
    )

    g.figure.suptitle("Insurance Data: Numerical Correlations", y=1.02)

    plt.tight_layout()
    plt.show()

# DATA PREPARATION
def split_data(
        df: pd.DataFrame,
        batch_size: int
        ) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:

    def create_loader(
            X: np.ndarray,
            y: np.ndarray,
            batch_size: int,
            shuffle: bool = False
            ) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    X = df.drop("charges", axis=1)
    y = np.log(df["charges"].values)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.15, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    numerical_cols =["age", "bmi", "children"]
    categorical_cols = ["sex", "smoker", "region"]

    x_scaler = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )

    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    X_val = x_scaler.transform(X_val)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))
    y_val = y_scaler.transform(y_val.reshape(-1, 1))

    train_loader = create_loader(X_train, y_train, batch_size, shuffle=True)
    test_loader = create_loader(X_test, y_test, batch_size, shuffle=True)
    val_loader = create_loader(X_val, y_val, batch_size, shuffle=True)

    return train_loader, test_loader, val_loader, x_scaler, y_scaler

# TRAINING
def train(
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module, 
        optimizer: torch.optim.Optimizer,
        device: str,
        writer: SummaryWriter,
        epoch: int,
        #scheduler: StepLR = None
        ) -> None:
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        global_step = current = batch * dataloader.batch_size + len(X)#Se supone que asi es mejor

        writer.add_scalar(f"Train/Loss", loss.item(), global_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch %20 == 0 and epoch%100==0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    
    # if scheduler:
    #     scheduler.step()


# TESTING
def test(
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        y_scaler: StandardScaler,
        device: str,
        writer: SummaryWriter,
        epoch: int
        ) -> None:
    model.eval()
    config = Config()

    test_loss = 0
    with torch.no_grad():
        nrmse = NormalizedRootMeanSquaredError().to(device)
        mae = MeanAbsoluteError().to(device)
        mape = MeanAbsolutePercentageError().to(device)

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X) 
            test_loss += loss_fn(pred, y).item()

            pred_original = torch.tensor(
                np.exp(y_scaler.inverse_transform(pred.detach().cpu().numpy())),
                device=device)
            y_original = torch.tensor(
                np.exp(y_scaler.inverse_transform(y.cpu().numpy())),
                device=device)
            
            nrmse.update(pred_original, y_original)
            mae.update(pred_original, y_original)
            mape.update(pred_original, y_original)
    test_loss /= len(dataloader)

    epoch_nrmse = nrmse.compute()
    epoch_mae = mae.compute()
    epoch_mape = mape.compute()

    writer.add_scalar("Test/Loss", test_loss, 0)
    writer.add_scalar("Test/NRMSE", epoch_nrmse, 0)
    writer.add_scalar("Test/MAE", epoch_mae, 0)
    writer.add_scalar("Test/MAPE", epoch_mape, 0)

    print(f"Test Error: \n Avg loss: {test_loss:8f}\n Avg MAPE: {epoch_mape:8f}\n")

# VALIDATION
def validate(
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        device:str, 
        writer: SummaryWriter,
        epoch: int
        ) -> None:
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X) 
            val_loss += loss_fn(pred, y).item()
    val_loss /= len(dataloader)

    writer.add_scalar(f"Validation/Loss", val_loss, epoch)
    return val_loss

# HYPERPARAM OPTIMIZATION
def make_objective(train_loader, val_loader, loss_fn):
    def objective(trial):
        config = Config()
        model_name = trial.suggest_categorical("model", list(config.models.keys()))
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        trial_writer = SummaryWriter(f"runs/insurance/trial_{trial.number}_{model_name}")
        model = config.models[model_name]().to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        n_epochs = config.epochs
        val_loss = float("inf")

        for t in range(n_epochs):
            train(train_loader, model, loss_fn, optimizer, config.device, trial_writer,epoch=t)
            val_loss = validate(val_loader, model, loss_fn, config.device, trial_writer, t)
            if t%100==0:
                print(f"Trial: {trial.number}")
                print(f"Validation Loss: {val_loss:8f}\n")


            trial.report(val_loss, t)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        trial_writer.close()
        return val_loss
    
    return objective

def weighted_stats(group):
    d={}
    d["model"] = group.name
    d["avg_lr"]=np.average(group["trial_lr"], weights=group["loss_inverse"])
    d["avg_loss"]=np.average(group["trial_val"], weights=group["loss_inverse"])

    return pd.Series(d)

def load_model(path):
    check=torch.load(path,weights_only=False)#SINO FALA
    model_key=check["model_key"]
    model=check["model"]
    lr=check["lr"]
    return model_key,model,lr