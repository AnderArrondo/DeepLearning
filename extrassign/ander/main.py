
#######
# Libraries
#######
from schemas import VAE
from optim import objective, vae_loss
from img_viz import view_image, view_images, view_reconstructions
from config import (
    SEED, DEVICE, INPUT_DIM,
    DATA_PATH, BEST_MODEL_PATH, OPTUNA_DB_URL,
    TRIALS, N_EPOCHS,
    N_STARTUP_TRIALS, WARMUP_EPOCHS,
    VISUALIZE_PLOTS
)
import optuna

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

#######
# Data collection
#######
train_dataset = datasets.MNIST(
    root=DATA_PATH,
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)
train_size = int(0.9 * len(train_dataset))
validation_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(
    train_dataset, 
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(SEED)
)
print("Train dataset size: " + str(len(train_dataset)))
print("Validation dataset size: " + str(len(val_dataset)))
test_dataset = datasets.MNIST(
    root=DATA_PATH,
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)
print("Test dataset size: " + str(len(test_dataset)))

# data example
image, label = train_dataset[0]
print("# Image example:")
print(image.shape)
print(image)
print("Label: " + str(label))
if VISUALIZE_PLOTS:
    view_image(image, label)

example_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# batch example
images, labels = next(iter(example_loader))
images = images.view(images.size(0), -1)
print(images.shape)
if VISUALIZE_PLOTS:
    view_images(images, labels, 8, 4)

#######
# Optuna
#######
sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=N_STARTUP_TRIALS)
pruner = optuna.pruners.MedianPruner(n_warmup_steps=WARMUP_EPOCHS)
study = optuna.create_study(
    study_name="vae_mnist",
    storage=OPTUNA_DB_URL,
    load_if_exists=True,
    direction="minimize",
    sampler=sampler,
    pruner=pruner
)
study.optimize(
    lambda trial: objective(trial, train_dataset, val_dataset),
    n_trials=TRIALS
)

# best model
best_params = study.best_params
batch_size = best_params["batch_size"]
best_hidden_dims = []
for i in range(best_params["n_layers"]):
    n_units_power = best_params[f"n_units_power_l{i}"]
    best_hidden_dims.append(2 ** n_units_power)
best_model = VAE(
    INPUT_DIM,
    best_hidden_dims,
    best_params["dropout_rate"],
    best_params["latent_dim"]
).to(DEVICE)

if best_params["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(
        best_model.parameters(),
        lr=best_params["lr"],
        betas=(best_params["beta1"], best_params["beta2"]),
        weight_decay=best_params["weight_decay"]
    )
else: 
    optimizer = torch.optim.AdamW(
        best_model.parameters(),
        lr=best_params["lr"],
        betas=(best_params["beta1"], best_params["beta2"]),
        weight_decay=best_params["weight_decay"]
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

writer = SummaryWriter(log_dir=f"extrassign/runs/best_trial")

for epoch in range(N_EPOCHS):
    best_model.train()
    total_loss = 0.0
    recon_loss = 0.0
    kl_loss = 0.0
    for images, _ in train_loader:
        images = images.view(images.size(0), -1).to(DEVICE)
        optimizer.zero_grad()

        x_hat, mu, logvar = best_model(images)

        loss, recon, kl = vae_loss(x_hat, images, mu, logvar)
        loss.backward()     
        optimizer.step()

        total_loss += loss.item()
        recon_loss += recon.item()
        kl_loss += kl.item()

    avg_loss = total_loss / len(train_loader.dataset)
    avg_recon = recon_loss / len(train_loader.dataset)
    avg_kl = kl_loss / len(train_loader.dataset)
    writer.add_scalar(
        "Best/Train/Loss",
        avg_loss,
        epoch
    )
    writer.add_scalar(
        "Best/Train/Recon",
        avg_recon,
        epoch
    )
    writer.add_scalar(
        "Best/Train/KL",
        avg_kl,
        epoch
    )
    print(f"Epoch {epoch+1}/{N_EPOCHS}, Train/Loss: {avg_loss:.4f}")

writer.close()

# testing 
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

best_model.eval()

total_loss = 0.0
recon_loss = 0.0
kl_loss = 0.0

with torch.no_grad():
    for images, _ in test_loader:
        images = images.view(images.size(0), -1).to(DEVICE)

        x_hat, mu, logvar = best_model(images)

        loss, recon, kl = vae_loss(x_hat, images, mu, logvar)

        total_loss += loss.item()
        recon_loss += recon.item()
        kl_loss += kl.item()

avg_loss = total_loss / len(test_loader.dataset)
avg_recon = recon_loss / len(test_loader.dataset)
avg_kl = kl_loss / len(test_loader.dataset)

print("\n===== TEST RESULTS =====")
print(f"Test Loss:  {avg_loss:.4f}")
print(f"Recon Loss: {avg_recon:.4f}")
print(f"KL Loss:    {avg_kl:.4f}")

# save model
torch.save(best_model.state_dict(), BEST_MODEL_PATH)

# reconstruction
images, labels = next(iter(test_loader))
images = images.view(images.size(0), -1).to(DEVICE)

with torch.no_grad():
    reconstructions, _, _ = best_model(images)

images = images.cpu().view(-1, 28, 28)
reconstructions = reconstructions.cpu().view(-1, 28, 28)

view_reconstructions(
    images,
    reconstructions,
    n_cols=8,
    n_rows=4
)
