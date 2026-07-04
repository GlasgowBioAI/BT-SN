"""Shared model definitions and embedding utilities for BT-SN."""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader


def set_seed(seed):
    """Set random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Encoder(nn.Module):
    """Standard encoder used by AE, BT, SCARF, BYOL, VIME."""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        all_dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                layers.append(nn.BatchNorm1d(all_dims[i + 1]))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VAE_Encoder(nn.Module):
    """VAE encoder with reparameterisation (mu only used for embeddings)."""
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.shared_layers = nn.Sequential()
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            self.shared_layers.add_module(f"linear_{i}", nn.Linear(dims[i], dims[i + 1]))
            self.shared_layers.add_module(f"bn_{i}", nn.BatchNorm1d(dims[i + 1]))
            self.shared_layers.add_module(f"relu_{i}", nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        return self.fc_mu(self.shared_layers(x))


def build_hidden_dims(input_dim, output_dim=4):
    """Build hidden layer dimensions: [1024, 512, 256, ...] until < output_dim."""
    hidden = [1024]
    for _ in range(1, 3):
        n = hidden[-1] // 2
        if n < output_dim:
            break
        hidden.append(n)
    return hidden


def load_encoder(checkpoint_path, input_dim, hidden_dims, output_dim=4, is_vae=False):
    """Load a pre-trained encoder from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = VAE_Encoder if is_vae else Encoder
    enc = cls(input_dim, hidden_dims, output_dim).to(device)
    sd = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    enc.load_state_dict(sd, strict=False)
    enc.eval()
    return enc


def generate_embeddings(data_df, encoder, device=None):
    """Generate 4-dim embeddings for all samples in data_df."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor(data_df.values, dtype=torch.float32)
    loader = DataLoader(tensor, batch_size=256, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            embeddings.append(encoder(batch.to(device)).cpu())
    emb = torch.cat(embeddings, dim=0)
    return pd.DataFrame(emb.numpy(), index=data_df.index)
