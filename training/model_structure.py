import torch
import torch.nn as nn
import torch.nn.functional as F
from common import set_seed
from barlow_config import *

set_seed(42)



def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Encoder(nn.Module):
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

        self.apply(initialize_weights)

    def forward(self, x):
        h = self.shared_layers(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterization trick
        return z, mu, log_var




class Projector(nn.Module):
    def __init__(self, output_dim, hidden_dims, input_dim):
        super().__init__()
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                layers.append(nn.BatchNorm1d(all_dims[i + 1]))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.apply(initialize_weights)

    def forward(self, x):
        return self.layers(x)


# Define the Barlow Twins loss function
def off_diagonal(x):
    # Return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param=5e-3, output_dim=1024):
        super().__init__()
        self.lambda_param = lambda_param
        self.bn = nn.BatchNorm1d(output_dim, affine=False)  # assuming the input features are of size 20000
        self.ln = nn.LayerNorm(output_dim)
        self.output_dim = output_dim

    def forward(self, z1, z2):
        # Apply batch normalization to z1 and z2
        # c = self.bn(z1).T @ self.bn(z2)
        c = self.bn(z1).T @ self.bn(z2)

        # Normalize the cross-correlation matrix by the batch size
        c.div_(z1.size(0))

        c_matrix = c.detach().clone()

        # Calculate the loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()

        lambda_times = (self.output_dim * self.lambda_param)/ (self.output_dim * (self.output_dim -1))
        # lambda_times = z1.size(0)/(self.lambda_param *(self.output_dim-1))
        loss = on_diag + lambda_times * off_diag

        return loss, c_matrix


# SCARF: InfoNCE (NT-Xent) Loss
# ═══════════════════════════════════════════════════════════



class ScarfProjector(nn.Module):
    """SCARF projector: 4 → 64 → 256 MLP."""
    def __init__(self, input_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 256),
        )
        self.apply(initialize_weights)
    def forward(self, x):
        return self.net(x)

class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) loss for contrastive learning.

    Given two augmented views of a batch, treats (z1[i], z2[i]) as positive pairs
    and all other pairs as negatives. After L2-normalization, computes:

    loss = -log( exp(sim(i,i)/τ) / Σ_j exp(sim(i,j)/τ) )

    Args:
        temperature: Temperature scaling parameter τ (default: 0.5)
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1: Projections from first view [B, dim]
            z2: Projections from second view [B, dim]
        Returns:
            loss: Scalar loss
        """
        B = z1.shape[0]

        # L2 normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate: [2*B, dim]
        z = torch.cat([z1, z2], dim=0)

        # Similarity matrix: [2*B, 2*B]
        sim = torch.mm(z, z.T) / self.temperature

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim[~mask].view(2 * B, -1)

        # Labels: for z[i], its positive is z[(i+B) % (2*B)]
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss


# ═══════════════════════════════════════════════════════════
# VIME: Mask Decoder for self-supervised feature + mask prediction
# ═══════════════════════════════════════════════════════════

class VIMEDecoder(nn.Module):
    """
    VIME dual decoder: predicts original features + mask from corrupted input.

    Two heads sharing the same backbone encoder:
    - feature_decoder: reconstructs the original feature values (MSE loss)
    - mask_decoder: predicts which features were masked (BCE loss)

    Args:
        input_dim: Original input dimension (number of genes)
        hidden_dims: List of hidden dimensions
    """
    def __init__(self, latent_dim, hidden_dims, input_dim):
        super().__init__()
        # Feature decoder: reconstruct original expression values
        feat_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        feat_layers = []
        for i in range(len(feat_dims) - 1):
            feat_layers.append(nn.Linear(feat_dims[i], feat_dims[i + 1]))
            if i < len(feat_dims) - 2:
                feat_layers.append(nn.BatchNorm1d(feat_dims[i + 1]))
                feat_layers.append(nn.ReLU())
        self.feature_decoder = nn.Sequential(*feat_layers)

        # Mask decoder: predict which genes were masked (binary)
        mask_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        mask_layers = []
        for i in range(len(mask_dims) - 1):
            mask_layers.append(nn.Linear(mask_dims[i], mask_dims[i + 1]))
            if i < len(mask_dims) - 2:
                mask_layers.append(nn.BatchNorm1d(mask_dims[i + 1]))
                mask_layers.append(nn.ReLU())
        # Final sigmoid is applied in loss function
        self.mask_decoder = nn.Sequential(*mask_layers)

        self.apply(initialize_weights)

    def forward(self, z):
        """
        Args:
            z: Latent representation from encoder [B, latent_dim]
        Returns:
            x_recon: Reconstructed original features [B, input_dim]
            m_logit: Mask logits (before sigmoid) [B, input_dim]
        """
        x_recon = self.feature_decoder(z)
        m_logit = self.mask_decoder(z)
        return x_recon, m_logit


class VIMELoss(nn.Module):
    """
    VIME combined loss: α * MSE(feature_reconstruction) + BCE(mask_prediction)

    The MSE loss is only computed on the masked (corrupted) positions.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x_recon, x_original, m_logit, m_true, mask_indices=None):
        """
        Args:
            x_recon: Reconstructed features [B, G]
            x_original: Original (uncorrupted) features [B, G]
            m_logit: Mask logits [B, G]
            m_true: True binary mask [B, G] (1=corrupted)
            mask_indices: Boolean mask of which positions were corrupted [B, G]
                          If None, all positions contribute equally.
        Returns:
            loss: Combined scalar loss
        """
        if mask_indices is not None:
            # Only compute reconstruction MSE on masked positions
            recon_loss = F.mse_loss(
                x_recon[mask_indices],
                x_original[mask_indices]
            )
        else:
            recon_loss = F.mse_loss(x_recon, x_original)

        mask_loss = self.bce(m_logit, m_true)

        return recon_loss + self.alpha * mask_loss, recon_loss, mask_loss


# ═══════════════════════════════════════════════════════════
# BYOL: Predictor + EMA update helper
# ═══════════════════════════════════════════════════════════

class BYOLPredictor(nn.Module):
    """
    BYOL's predictor head (only on the online branch).
    Architecture mirrors the projector (asymmetric: predictor only exists on online side).
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        all_dims = [input_dim] + hidden_dims[::-1] + [output_dim]
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                layers.append(nn.BatchNorm1d(all_dims[i + 1]))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.apply(initialize_weights)

    def forward(self, x):
        return self.layers(x)


def byol_loss_fn(q, z):
    """
    BYOL loss: MSE between L2-normalized online prediction and target projection.

    Loss = ||l2_norm(q) - l2_norm(z)||² = 2 - 2*cosine_sim(q, z)
    """
    q = F.normalize(q, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (q * z).sum(dim=1).mean()


def update_target_network(online_net, target_net, tau):
    """
    EMA update: ξ ← τ * ξ + (1 - τ) * θ

    Args:
        online_net: Online network (θ)
        target_net: Target network (ξ), updated in-place
        tau: EMA decay rate (close to 1 = slow update)
    """
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        p_target.data.mul_(tau).add_(p_online.data, alpha=1 - tau)
