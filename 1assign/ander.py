
# LIBRARIES
import torch
import torch.nn as nn
from torchmetrics import NormalizedRootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import StandardScaler


# CONFIGS
csv_path: str = "./data/insurance.csv"
device: str = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
show_plots: bool = False
train_model: bool = True
batch_size: int = 32
epochs: int = 20
lr: float = 1e-3
writer = SummaryWriter("runs/insurance")

random_seed: int = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

print("# DATA COLLECTION")
raw_df: pd.DataFrame = pd.read_csv(csv_path)
print(raw_df.head())

print("\n")

print("# PREPROCESSING")
raw_df.info()
print()
print(raw_df.nunique())
print()

categorical_cols = ["sex", "smoker", "region"]
for category in categorical_cols:
    raw_df[category] = pd.Categorical(raw_df[category])

df: pd.DataFrame = pd.get_dummies(raw_df, drop_first=False, dtype=int)
print(df.head())
print()
df.info()
print()

def visualize_distributions(
        df: pd.DataFrame,
        raw_df: pd.DataFrame
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

    sns.barplot(df["smoker_yes"].value_counts(), ax=ax[1, 1], color="royalblue")
    ax[1, 1].set_title("Smoker Distribution")

    sns.barplot(raw_df["region"].value_counts(), ax=ax[2, 0], color="royalblue")
    ax[2, 0].set_title("Region Distribution")

    sns.histplot(
        data=df, x="charges", kde=True, ax=ax[2, 1],
        color="royalblue"
    )
    ax[2, 1].set_title("Charges Distribution")

    plt.tight_layout()
    plt.show()

if show_plots:
    print("# EXPLORATORY DATA ANALYSIS")
    visualize_distributions(df, raw_df)

    fig = px.scatter_matrix(
        raw_df,
        dimensions=["age", "bmi", "charges"],
        color="smoker",
        symbol="sex",
        opacity=0.25,
        title="Insurance Data: Numerical Correlations",
        labels={col: col.replace('_', ' ').title() for col in raw_df.columns}
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    # se abre en localhost
    # fig.show()

# DATA PREPARATION
def split_data(
        df: pd.DataFrame,
        batch_size: int
        ) -> np.ndarray[DataLoader]:

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


    X = df.drop("charges", axis=1).values
    y = df["charges"].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    X_val = x_scaler.transform(X_val)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = y_scaler.transform(y_test.reshape(-1, 1))
    y_val = y_scaler.transform(y_val.reshape(-1, 1))

    train_loader = create_loader(X_train, y_train, batch_size, shuffle=True)
    test_loader = create_loader(X_test, y_test, batch_size)
    val_loader = create_loader(X_val, y_val, batch_size)

    return train_loader, test_loader, val_loader, x_scaler, y_scaler

train_loader, test_loader, val_loader, x_scaler, y_scaler = split_data(df, batch_size=batch_size)
for X, y in train_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# MODEL DEFINITION
class InsuranceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),

            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

model = InsuranceModel().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# TRAINING
def train(
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module, 
        optimizer: torch.optim.Optimizer,
        writer: SummaryWriter,
        epoch: int
        ) -> None:
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        global_step = epoch * len(dataloader) + batch

        writer.add_scalar("Train/Loss", loss.item(), global_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# TESTING
def test(
        dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        y_scaler: StandardScaler,
        writer: SummaryWriter,
        epoch: int
        ) -> None:
    model.eval()
    
    nrmse = NormalizedRootMeanSquaredError().to(device)
    mae = MeanAbsoluteError().to(device)
    mape = MeanAbsolutePercentageError().to(device)

    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X) 
            test_loss += loss_fn(pred, y).item()

            pred_original = torch.tensor(y_scaler.inverse_transform(pred.cpu().numpy()), device=device)
            y_original = torch.tensor(y_scaler.inverse_transform(y.cpu().numpy()), device=device)
            nrmse.update(pred_original, y_original)
            mae.update(pred_original, y_original)
            mape.update(pred_original, y_original)
    test_loss /= len(dataloader)

    epoch_nrmse = nrmse.compute()
    epoch_mae = mae.compute()
    epoch_mape = mape.compute()

    writer.add_scalar("Test/Loss", test_loss, epoch)
    writer.add_scalar("Test/NRMSE", epoch_nrmse, epoch)
    writer.add_scalar("Test/MAE", epoch_mae, epoch)
    writer.add_scalar("Test/MAPE", epoch_mape, epoch)

    print(f"Test Error: \n Avg loss: {test_loss:8f}\n Avg MAPE: {epoch_mape:8f}\n")
    mape.reset()

if train_model:
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, writer, t)
        test(test_loader, model, loss_fn, y_scaler, writer, t)
    print("Done!")
