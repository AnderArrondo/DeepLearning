
#######
# VAE class utils
#######

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate, latent_dim):
        super(Encoder, self).__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h

        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, dropout_rate, output_dim):
        super(Decoder, self).__init__()

        layers = []
        prev_dim = latent_dim

        for h in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h

        self.decoder = nn.Sequential(*layers)

        self.fc_out = nn.Linear(prev_dim, output_dim)

    def forward(self, z):
        h = self.decoder(z)
        x = self.fc_out(h)
        return torch.sigmoid(x)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, dropout_rate, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, dropout_rate, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    