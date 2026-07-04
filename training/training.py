import torch
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import hashlib
from torch.cuda.amp import autocast
from model_structure import byol_loss_fn, update_target_network

def compute_regularization_loss(encoder, projector, regularization, main_loss, l1_lambda=0.0, l2_lambda=0.0, epsilon=1e-8):
    """
      L1 / L2  ，  l1_lambda   l2_lambda
    :param encoder:  
    :param projector:  （ ）
    :param regularization:   'l1' / 'l2' / 'elastic' / None
    :param main_loss:  
    :param l1_lambda: L1  
    :param l2_lambda: L2  
    :param epsilon:  
    :return:  
    """
    #   L1   L2  
    l1_norm = sum(p.abs().sum() for p in encoder.parameters())
    l2_norm = sum((p ** 2).sum() for p in encoder.parameters())

    #   projector， 
    if projector:
        l1_norm += sum(p.abs().sum() for p in projector.parameters())
        l2_norm += sum((p ** 2).sum() for p in projector.parameters())

    #  ，  l1   l2   epsilon
    l1_norm = l1_norm + epsilon
    l2_norm = l2_norm + epsilon

    #  
    if regularization == 'l1':
        reg_loss = l1_lambda * l1_norm  #   L1   l1_lambda
    elif regularization == 'l2':
        reg_loss = l2_lambda * l2_norm  #   L2   l2_lambda
    elif regularization == 'elastic':
        reg_loss = l1_lambda * l1_norm + l2_lambda * l2_norm  # L1   L2  
    else:
        reg_loss = torch.tensor(0.0, device=main_loss.device)  #  ，  0

    return reg_loss


import numpy as np

import torch

def generate_weighted_normal_tensor(
    normal_bulk: torch.Tensor,  # [N_norm, G]，  device  
    batch_size: int,
    num_samples: int = 20,
) -> torch.Tensor:
    """
      batch   normal bulk，  PyTorch  。
     ：
      - normal_bulk: Tensor，shape = [N_norm, G]，  GPU  
      - batch_size:  int，  bulk
      - num_samples: int，  normal_bulk   num_samples  （ ）
     ：
      - w_batch: Tensor，shape = [batch_size, G]
    """
    N_norm, G = normal_bulk.shape
    device = normal_bulk.device

    # 1.   [batch_size, num_samples]
    idxs = torch.randint(0, N_norm, (batch_size, num_samples), device=device)

    # 2.   sampled：[batch_size, num_samples, G]
    sampled = normal_bulk[idxs]  #  

    # 3.  ：[batch_size, num_samples, 1]
    weights = torch.rand(batch_size, num_samples, device=device)
    weights = weights / weights.sum(dim=1, keepdim=True)
    weights = weights.unsqueeze(-1)

    # 4.   → [batch_size, G]
    w_batch = (weights * sampled).sum(dim=1)

    return w_batch  # dtype   normal_bulk  


def generate_gaussian_noise_tensor(
    bulk_batch: torch.Tensor,  # [batch_size, G]， batch 
    random_seed: int = 42,
) -> torch.Tensor:
    """
     batch Gaussian distortion。
     ：
      - bulk_batch: Tensor，shape = [batch_size, G]， batch 
      - random_seed: int， 
     ：
      - gaussian_noise: Tensor，shape = [batch_size, G]，Gaussian distortion
    """
    device = bulk_batch.device
    batch_size, G = bulk_batch.shape
    
    #  batch 
    batch_mean = torch.mean(bulk_batch, dim=0)  # shape = [G]
    batch_std = torch.std(bulk_batch, dim=0)    # shape = [G]
    
    #  （ ： ）
    torch.manual_seed(random_seed)
    
    #  Gaussian  -  
    gaussian_noise = torch.zeros_like(bulk_batch)
    for i in range(batch_size):
        gaussian_noise[i] = torch.normal(
            mean=batch_mean,
            std=batch_std
        )
    
    return gaussian_noise


def train_barlow_twins(encoder, projector, dataloader, criterion, optimizer, scaler, device, normal_bulk,
                       lambda_pseudo_bulk_rate1,
                       lambda_pseudo_bulk_rate2,
                       regularization=None, l1_regular_lambda=1e-5, l2_regular_lambda=1e-5):
    encoder.train()
    if projector:
        projector.train()
    total_loss = 0

    normal_bulk_tensor = torch.from_numpy(normal_bulk.values).to(device).float()
    eps = 1e-6  #   log2(0)

    for bulk_batch, idx_batch in dataloader:
        bulk_batch = bulk_batch.to(device, non_blocking=True)

        # ----   ----
        w1 = generate_weighted_normal_tensor(normal_bulk_tensor, bulk_batch.size(0), 5)
        w2 = generate_weighted_normal_tensor(normal_bulk_tensor, bulk_batch.size(0), 5)

        d1 = torch.log2(((1 - lambda_pseudo_bulk_rate1) * bulk_batch + lambda_pseudo_bulk_rate1 * w1).clamp(min=eps) + 1)
        d2 = torch.log2(((1 - lambda_pseudo_bulk_rate2) * bulk_batch + lambda_pseudo_bulk_rate2 * w2).clamp(min=eps) + 1)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            # ---- Encoder ----
            z1 = encoder(d1)
            z2 = encoder(d2)

            # ---- Projector (embedding + decoder) ----
            if projector is not None:
                z1_proj = projector(z1)
                z2_proj = projector(z2)
            else:
                #   projector  ，  Barlow Twins  
                z1_proj = z1
                z2_proj = z2

            # ----  : Barlow Twins + MSE ----
            loss, c_matrix = criterion(z1_proj, z2_proj)
            last_c_matrix = c_matrix

            # ----   ----
            if regularization == 'l1':
                l1_norm = sum(p.abs().sum() for p in encoder.parameters())
                if projector:
                    l1_norm += sum(p.abs().sum() for p in projector.parameters())
                loss += l1_regular_lambda * l1_norm

            elif regularization == 'l2':
                l2_norm = sum(p.pow(2).sum() for p in encoder.parameters())
                if projector:
                    l2_norm += sum(p.pow(2).sum() for p in projector.parameters())
                loss += l2_regular_lambda * l2_norm

            elif regularization == 'elastic':
                l1_norm = sum(p.abs().sum() for p in encoder.parameters())
                l2_norm = sum(p.pow(2).sum() for p in encoder.parameters())
                if projector:
                    l1_norm += sum(p.abs().sum() for p in projector.parameters())
                    l2_norm += sum(p.pow(2).sum() for p in projector.parameters())
                loss += l1_regular_lambda * l1_norm + l2_regular_lambda * l2_norm

        # ---- AMP backward & optimizer step ----
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss, last_c_matrix


def train_vae(encoder, decoder, dataloader, criterion, optimizer, scaler, device, beta):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_samples = 0

    for bulk, *_ in dataloader:
        bulk = torch.log2(bulk + 1).to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            z, mu, log_var = encoder(bulk)
            reconstructed = decoder(z)

            recon_loss = criterion(reconstructed, bulk)
            # KL Divergence
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            kl_loss = kl_loss.mean()
            loss = recon_loss + beta * kl_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * bulk.size(0)
        total_samples += bulk.size(0)

    return total_loss / total_samples


def train_auto_encoder(encoder, decoder, dataloader, criterion, optimizer, scaler, device, normal_bulk,
                       lambda_pseudo_bulk_rate1,
                       lambda_pseudo_bulk_rate2,
                       regularization=None, l1_regular_lambda=1e-5, l2_regular_lambda=1e-5):

    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_samples = 0
    eps = 1e-6  #   log2(0)

    normal_bulk_tensor = torch.from_numpy(normal_bulk.values).to(device).float()
    for bulk, *_ in dataloader:

        bulk = bulk.to(device, non_blocking=True)
        bulk_log2 = torch.log2(bulk + 1)

        # ----   ----
        w1 = generate_weighted_normal_tensor(normal_bulk_tensor, bulk.size(0), 5)
        d1 = torch.log2(((1 - lambda_pseudo_bulk_rate1) * bulk + lambda_pseudo_bulk_rate1 * w1).clamp(min=eps) + 1)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            encoded = encoder(d1)
            reconstructed = decoder(encoded)

            loss = criterion(reconstructed, bulk_log2)

            if regularization == 'l1':
                l1_norm_enc = sum(p.abs().sum() for p in encoder.parameters())
                l1_norm_dec = sum(p.abs().sum() for p in decoder.parameters())
                loss += l1_regular_lambda * (l1_norm_enc + l1_norm_dec)

            elif regularization == 'l2':
                l2_norm_enc = sum(p.pow(2).sum() for p in encoder.parameters())
                l2_norm_dec = sum(p.pow(2).sum() for p in decoder.parameters())
                loss += l2_regular_lambda * (l2_norm_enc + l2_norm_dec)

            elif regularization == 'elastic':
                l1_norm_enc = sum(p.abs().sum() for p in encoder.parameters())
                l2_norm_enc = sum(p.pow(2).sum() for p in encoder.parameters())
                l1_norm_dec = sum(p.abs().sum() for p in decoder.parameters())
                l2_norm_dec = sum(p.pow(2).sum() for p in decoder.parameters())
                loss += l1_regular_lambda * (l1_norm_enc + l1_norm_dec) + \
                        l2_regular_lambda * (l2_norm_enc + l2_norm_dec)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * bulk.size(0)
        total_samples += bulk.size(0)

    return total_loss / total_samples



def train_scarf(encoder, projector, dataloader, criterion, optimizer, scaler, device, normal_bulk,
                lambda_pseudo_bulk_rate1, lambda_pseudo_bulk_rate2,
                regularization=None, l1_regular_lambda=1e-5, l2_regular_lambda=1e-5):
    """
    Train one epoch of SCARF contrastive learning with pseudo-bulk augmentation.

    Uses the same augmentation as Barlow Twins (pseudo-bulk mixing with normal samples),
    but the loss is InfoNCE (NT-Xent) instead of cross-correlation redundancy reduction.
    """
    encoder.train()
    if projector:
        projector.train()
    total_loss = 0

    normal_bulk_tensor = torch.from_numpy(normal_bulk.values).to(device).float()
    eps = 1e-6

    for bulk_batch, idx_batch in dataloader:
        bulk_batch = bulk_batch.to(device, non_blocking=True)

        # ---- pseudo-bulk augmentation (same as Barlow Twins) ---- #
        w1 = generate_weighted_normal_tensor(normal_bulk_tensor, bulk_batch.size(0), 5)
        w2 = generate_weighted_normal_tensor(normal_bulk_tensor, bulk_batch.size(0), 5)

        d1 = torch.log2(((1 - lambda_pseudo_bulk_rate1) * bulk_batch + lambda_pseudo_bulk_rate1 * w1).clamp(min=eps) + 1)
        d2 = torch.log2(((1 - lambda_pseudo_bulk_rate2) * bulk_batch + lambda_pseudo_bulk_rate2 * w2).clamp(min=eps) + 1)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            z1 = encoder(d1)
            z2 = encoder(d2)

            if projector is not None:
                z1_proj = projector(z1)
                z2_proj = projector(z2)
            else:
                z1_proj = z1
                z2_proj = z2

            # ---- InfoNCE loss (instead of Barlow Twins cross-correlation) ---- #
            loss = criterion(z1_proj, z2_proj)

            # ---- Regularization (same as Barlow Twins) ---- #
            if regularization == 'l1':
                l1_norm = sum(p.abs().sum() for p in encoder.parameters())
                if projector:
                    l1_norm += sum(p.abs().sum() for p in projector.parameters())
                loss += l1_regular_lambda * l1_norm
            elif regularization == 'l2':
                l2_norm = sum(p.pow(2).sum() for p in encoder.parameters())
                if projector:
                    l2_norm += sum(p.pow(2).sum() for p in projector.parameters())
                loss += l2_regular_lambda * l2_norm
            elif regularization == 'elastic':
                l1_norm = sum(p.abs().sum() for p in encoder.parameters())
                if projector:
                    l1_norm += sum(p.abs().sum() for p in projector.parameters())
                l2_norm = sum(p.pow(2).sum() for p in encoder.parameters())
                if projector:
                    l2_norm += sum(p.pow(2).sum() for p in projector.parameters())
                loss += l1_regular_lambda * l1_norm + l2_regular_lambda * l2_norm

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader), None  # No correlation matrix for SCARF


# ═══════════════════════════════════════════════════════════
# VIME: Generative self-supervision with feature + mask prediction
# Uses random gene masking (NOT pseudo-bulk) as augmentation
# ═══════════════════════════════════════════════════════════

def generate_vime_corruption(bulk_batch, mask_prob=0.3):
    """
    Generate VIME-style corruption: random feature masking with marginal sampling.

    For each feature, randomly permute its values across the batch (marginal sampling),
    then apply a binary mask to select which features to corrupt.

    Args:
        bulk_batch: Original data [B, G]
        mask_prob: Probability of masking each feature (default: 0.3)

    Returns:
        corrupted: Corrupted data [B, G]
        mask: Binary mask [B, G] (1 = corrupted position)
    """
    B, G = bulk_batch.shape
    device = bulk_batch.device

    # Generate binary mask: 1 = corrupt, 0 = keep original
    mask = torch.rand(B, G, device=device) < mask_prob
    mask = mask.float()

    # Marginal sampling: shuffle each column independently across the batch
    # For each gene, randomly permute its values
    corrupted = bulk_batch.clone()
    for j in range(G):
        perm = torch.randperm(B, device=device)
        corrupted[:, j] = bulk_batch[perm, j]

    # Apply mask: corrupted = m * corrupted + (1-m) * original
    result = mask * corrupted + (1 - mask) * bulk_batch

    return result, mask


def train_vime(encoder, vime_decoder, dataloader, criterion, optimizer, scaler, device,
               mask_prob=0.3, alpha=1.0):
    """
    Train one epoch of VIME.

    VIME corrupts input features via random masking + marginal sampling,
    then learns to:
    1. Reconstruct the original feature values (MSE on masked positions)
    2. Predict which features were masked (binary classification)
    """
    encoder.train()
    vime_decoder.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_mask_loss = 0.0
    total_samples = 0

    for bulk, *_ in dataloader:
        bulk = bulk.to(device, non_blocking=True)
        B = bulk.shape[0]

        # ---- Generate corruption ---- #
        corrupted, mask = generate_vime_corruption(bulk, mask_prob)

        # ---- Log2 transform both original and corrupted ---- #
        bulk_log2 = torch.log2(bulk + 1)
        corrupted_log2 = torch.log2(corrupted + 1)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            z = encoder(corrupted_log2)
            x_recon, m_logit = vime_decoder(z)

            # Mask indices: where mask == 1 (corrupted positions)
            mask_bool = mask.bool()

            loss, recon_loss, mask_loss = criterion(
                x_recon, bulk_log2, m_logit, mask, mask_bool
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * B
        total_recon_loss += recon_loss.item() * B
        total_mask_loss += mask_loss.item() * B
        total_samples += B

    return (total_loss / total_samples,
            total_recon_loss / total_samples,
            total_mask_loss / total_samples)


# ═══════════════════════════════════════════════════════════
# BYOL: Bootstrap Your Own Latent
# Uses pseudo-bulk augmentation (same as Barlow Twins/SCARF)
# Online branch sees d1, target branch sees d2
# ═══════════════════════════════════════════════════════════

def train_byol(encoder_online, encoder_target,
               projector_online, projector_target,
               predictor,
               dataloader, optimizer, scaler, device, normal_bulk,
               lambda_pseudo_bulk_rate1, lambda_pseudo_bulk_rate2,
               tau=0.996,  # EMA decay rate
               regularization=None, l1_regular_lambda=1e-5, l2_regular_lambda=1e-5):
    """
    Train one epoch of BYOL with pseudo-bulk augmentation.

    BYOL uses asymmetric networks:
    - Online: encoder → projector → predictor (trained via gradient)
    - Target: encoder → projector (EMA copy of online, stop-gradient)

    Views: online(d1) tries to predict target(d2)'s projection.
    Loss is symmetrized (also online(d2) → target(d1)).
    """
    encoder_online.train()
    projector_online.train()
    predictor.train()
    # Target network is in eval mode (no batchnorm updates) or train?
    # BYOL original: target BN stats are updated via EMA, but for simplicity,
    # keep target in eval mode (frozen BN stats)
    encoder_target.eval()
    projector_target.eval()

    total_loss = 0

    normal_bulk_tensor = torch.from_numpy(normal_bulk.values).to(device).float()
    eps = 1e-6

    for bulk_batch, idx_batch in dataloader:
        bulk_batch = bulk_batch.to(device, non_blocking=True)

        # ---- pseudo-bulk augmentation ---- #
        w1 = generate_weighted_normal_tensor(normal_bulk_tensor, bulk_batch.size(0), 5)
        w2 = generate_weighted_normal_tensor(normal_bulk_tensor, bulk_batch.size(0), 5)

        d1 = torch.log2(((1 - lambda_pseudo_bulk_rate1) * bulk_batch + lambda_pseudo_bulk_rate1 * w1).clamp(min=eps) + 1)
        d2 = torch.log2(((1 - lambda_pseudo_bulk_rate2) * bulk_batch + lambda_pseudo_bulk_rate2 * w2).clamp(min=eps) + 1)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            # ---- Online branch (trainable) ---- #
            z1 = encoder_online(d1)
            p1 = projector_online(z1)
            q1 = predictor(p1)

            z2 = encoder_online(d2)
            p2 = projector_online(z2)
            q2 = predictor(p2)

            # ---- Target branch (EMA, no gradient) ---- #
            with torch.no_grad():
                z1_t = encoder_target(d1)
                p1_t = projector_target(z1_t)

                z2_t = encoder_target(d2)
                p2_t = projector_target(z2_t)

            # ---- Symmetrized loss ---- #
            # online(d1) predicts target(d2): loss1
            # online(d2) predicts target(d1): loss2
            loss = byol_loss_fn(q1, p2_t) + byol_loss_fn(q2, p1_t)

            # ---- Regularization (on online encoder + projector) ---- #
            if regularization == 'l1':
                l1_norm = sum(p.abs().sum() for p in encoder_online.parameters())
                l1_norm += sum(p.abs().sum() for p in projector_online.parameters())
                loss += l1_regular_lambda * l1_norm
            elif regularization == 'l2':
                l2_norm = sum(p.pow(2).sum() for p in encoder_online.parameters())
                l2_norm += sum(p.pow(2).sum() for p in projector_online.parameters())
                loss += l2_regular_lambda * l2_norm
            elif regularization == 'elastic':
                l1_norm = sum(p.abs().sum() for p in encoder_online.parameters())
                l2_norm = sum(p.pow(2).sum() for p in encoder_online.parameters())
                l1_norm += sum(p.abs().sum() for p in projector_online.parameters())
                l2_norm += sum(p.pow(2).sum() for p in projector_online.parameters())
                loss += l1_regular_lambda * l1_norm + l2_regular_lambda * l2_norm

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ---- EMA update target network ---- #
        update_target_network(encoder_online, encoder_target, tau)
        update_target_network(projector_online, projector_target, tau)

        total_loss += loss.item()

    return total_loss / len(dataloader), None  # No correlation matrix for BYOL

