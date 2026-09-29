#!/usr/bin/env python3
"""
TCGA Survival Evaluation
- Load test distortion data at λ=0.0 (no distortion)
- Five-fold CoxPH evaluation within the cohort
- Generate 4-dim embeddings using: PCA, NMF, AE, VAE, BT
- Average seed-specific C-indices within each fold
"""

import os
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA, NMF
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

import sys, os as _os
_bt_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), '..', 'training')
sys.path.insert(0, _bt_path)
from common import Encoder, VAE_Encoder, build_hidden_dims, generate_embeddings

from paths import BASE, p, checkpoint_runs
from common_survival import cross_validated_cindex
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ──────────────────────────────────────────────
# 1. Survival data (TCGA)
# ──────────────────────────────────────────────
survival_path = p('tcga_label_data/TCGA_survival_cancer.csv')
survival = pd.read_csv(survival_path, index_col=0)
survival.index = survival.index.str[:-3]  # remove sample suffix
survival = survival.rename(columns={'OS': 'status', 'OS.time': 'time'})
survival.loc[survival['time'] == 0, 'time'] = 1
survival = survival[['time', 'status']].dropna()
print(f"===== Before any filtering =====")
print(f"  survival shape: {survival.shape}")
print(f"  survival time range: {survival['time'].min():.0f} - {survival['time'].max():.0f}")
print(f"  events: {survival['status'].sum():.0f} / {survival.shape[0]}")

# ──────────────────────────────────────────────
# 2. Load test distortion data (λ=0.0 only)
# ──────────────────────────────────────────────
test_path = p('different_mixture_data/TCGA_evaluation_fine_tune/data_normal_bulk_distort/test/normal_bulk_distortion_test_0.0_fullset.txt')
print(f"\n=== Load distortion data ===")
test_dist = pd.read_csv(test_path, index_col=0, sep=',').sort_index().fillna(0)
print(f"  test_dist shape: {test_dist.shape}")

# ──────────────────────────────────────────────
# 3. Filter: keep only common index
# ──────────────────────────────────────────────
print(f"\n=== Filter samples ===")
common = test_dist.index.intersection(survival.index)
print(f"  common (test intersect survival): {len(common)}")

test_dist = test_dist.loc[common]
survival = survival.loc[common]

print(f"  After filtering:")
print(f"  test_dist shape: {test_dist.shape}")
print(f"  survival shape: {survival.shape}")
print(f"  events: {survival['status'].sum():.0f} / {survival.shape[0]}")
print(f"  survival time range: {survival['time'].min():.0f} - {survival['time'].max():.0f}")

# ──────────────────────────────────────────────
# 4. Define models
# ──────────────────────────────────────────────


checkpoints = {name: checkpoint_runs(name, 'tcga')
               for name in ('AE', 'VAE', 'BT', 'SCARF', 'BYOL', 'VIME', 'BT_noise')}

# ──────────────────────────────────────────────
# 5. Generate embeddings
# ──────────────────────────────────────────────
input_dim = test_dist.shape[1]
hidden_dims = build_hidden_dims(input_dim)
X = test_dist.values
results = {}

# PCA
print(f"\n=== PCA (dim=4) ===")
pca = PCA(n_components=4, random_state=42)
emb_pca = pca.fit_transform(X)
results['PCA'] = [pd.DataFrame(emb_pca, index=test_dist.index)]

# NMF
print(f"NMF (dim=4)")
X_pos = X - X.min(axis=0) + 1e-10
nmf = NMF(n_components=4, random_state=42, max_iter=1000)
emb_nmf = nmf.fit_transform(X_pos)
results['NMF'] = [pd.DataFrame(emb_nmf, index=test_dist.index)]

# AE / VAE / BT / BT_noise
for model_name, run_paths in checkpoints.items():
    if not run_paths: print(f"\n{model_name}: no checkpoints found, skipping"); continue
    results[model_name] = []
    for ckpt_path in run_paths:
        print(f"\n{model_name}: loading {ckpt_path}")
        encoder = VAE_Encoder(input_dim, hidden_dims, 4).to(device) if model_name == 'VAE' else Encoder(input_dim, hidden_dims, 4).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        encoder.load_state_dict(sd); encoder.eval()
        results[model_name].append(generate_embeddings(test_dist, encoder, device))

# ──────────────────────────────────────────────
# 6. Survival evaluation
# ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Method':10s} {'C-index':>10s}")
print(f"{'='*60}")

all_metrics = []
for method_name, run_embeddings in results.items():
    fold_ci = cross_validated_cindex(run_embeddings, survival)
    ci_mean, ci_std = fold_ci.mean(), fold_ci.std(ddof=1)
    print(f"{method_name:10s} {ci_mean:>7.4f}+-{ci_std:.4f}")
    all_metrics.append({
        'method': method_name,
        'c_index_mean': round(ci_mean, 4),
        'c_index_std': round(ci_std, 4),
    })

metrics_df = pd.DataFrame(all_metrics)
out_path = os.path.join(os.path.dirname(__file__), 'tcga_survival_results.csv')
metrics_df.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")
print(metrics_df)
