
#######
# Optuna optimization
#######
from extrassign.ander.schemas import VAE
from extrassign.ander.config import INPUT_DIM, DEVICE, N_EPOCHS

import optuna

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    # reconstruction loss
    recon = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    # kl divergence loss
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl

def objective(trial, train_dataset, val_dataset):
    # hyper params
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    beta1 = trial.suggest_float("beta1", 0.8, 0.95)
    beta2 = trial.suggest_float("beta2", 0.98, 0.9999)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)


    n_layers = trial.suggest_int("n_layers", 1, 6)
    hidden_dims = []
    for i in range(n_layers):
        n_units_power = trial.suggest_int(f"n_units_power_l{i}", 3, 9) 
        hidden_dims.append(2 ** n_units_power)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    latent_dim = trial.suggest_int("latent_dim", 8, 256, log=True)

    # model and optimizer and data
    model = VAE(INPUT_DIM, hidden_dims, dropout_rate, latent_dim).to(DEVICE)

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
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

    writer = SummaryWriter(log_dir=f"extrassign/runs/trial_{trial.number}")

    # training loop
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(DEVICE)
            optimizer.zero_grad()

            x_hat, mu, logvar = model(images)

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
            "Train/Loss",
            avg_loss,
            epoch
        )
        writer.add_scalar(
            "Train/Recon",
            avg_recon,
            epoch
        )
        writer.add_scalar(
            "Train/KL",
            avg_kl,
            epoch
        )
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Train/Loss: {avg_loss:.4f}")
    
        # validation loop
        model.eval()
        total_loss = 0.0
        recon_loss = 0.0
        kl_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.view(images.size(0), -1).to(DEVICE)

                x_hat, mu, logvar = model(images)

                loss, recon, kl = vae_loss(x_hat, images, mu, logvar)
                total_loss += loss.item()
                recon_loss += recon.item()
                kl_loss += kl.item()

        avg_loss = total_loss / len(val_loader.dataset)
        avg_recon = recon_loss / len(val_loader.dataset)
        avg_kl = kl_loss / len(val_loader.dataset)
        writer.add_scalar(
            "Validation/Loss",
            avg_loss,
            epoch
        )
        writer.add_scalar(
            "Validation/Recon",
            avg_recon,
            epoch
        )
        writer.add_scalar(
            "Validation/KL",
            avg_kl,
            epoch
        )
        print(f"Epoch {epoch+1}/{N_EPOCHS}, Validation/Loss: {avg_loss:.4f}")
        trial.report(avg_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    writer.close()
    return avg_loss
