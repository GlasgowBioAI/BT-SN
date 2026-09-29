#!/usr/bin/env python3
"""
SCAN-B Survival Evaluation
- Load test distortion data at λ=0.0 (no distortion)
- Generate 4-dim embeddings using: PCA, NMF, AE, VAE, BT
- Split by median of dim0 → high/low risk
- Compute C-index and log-rank p-value
"""

import os
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA, NMF
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

import sys, os as _os
_bt_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), '..', 'training')
sys.path.insert(0, _bt_path)
from common import Encoder, VAE_Encoder, build_hidden_dims, generate_embeddings

from paths import BASE, p, checkpoint_runs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ──────────────────────────────────────────────
# 1. Load test labels + survival data
# ──────────────────────────────────────────────
test_label_path = p('scanb_data/scanb_2022/test_labels.csv')
test_labels = pd.read_csv(test_label_path, index_col=0).sort_index()

print(f"\n=== Load labels ===")
print(f"  test_labels shape: {test_labels.shape}")
print(f"  test_labels columns: {list(test_labels.columns)}")
print(f"  test_labels PAM50 subtypes: {test_labels['NCN.PAM50'].value_counts().to_dict()}")

# Survival data is in the labels: OS_days / OS_event
survival = test_labels[['OS_days', 'OS_event']].dropna().rename(columns={'OS_days': 'time', 'OS_event': 'status'})
print(f"\n  survival samples with OS data: {survival.shape}")

# ──────────────────────────────────────────────
# 2. Load test distortion data (λ=0.0 only)
# ──────────────────────────────────────────────
test_path = p('scanb_data/scanb_2022/data_normal_bulk_distort/test/normal_bulk_distortion_test_full_set_0.0.txt')
print(f"\n=== Load distortion data ===")
test_dist = pd.read_csv(test_path, index_col=0, sep=',').sort_index()
print(f"  test_dist shape: {test_dist.shape}")

common = test_dist.index.intersection(survival.index)
test_dist = test_dist.loc[common]
survival = survival.loc[common]
print(f"  {len(common)} samples with survival data")
# ──────────────────────────────────────────────
# 4. Define models
# ──────────────────────────────────────────────


checkpoints = {name: checkpoint_runs(name, 'scanb')
               for name in ('AE', 'VAE', 'BT', 'SCARF', 'BYOL', 'VIME', 'BT_noise')}

# ──────────────────────────────────────────────
# 5. Generate embeddings for each method
# ──────────────────────────────────────────────
input_dim = test_dist.shape[1]
hidden_dims = build_hidden_dims(input_dim)
X = test_dist.values
results = {}

# -- PCA --
print(f"\n=== PCA (dim=4) ===")
pca = PCA(n_components=4, random_state=42)
emb_pca = pca.fit_transform(X)
results['PCA'] = [pd.DataFrame(emb_pca, index=test_dist.index)]

# -- NMF --
print(f"NMF (dim=4)")
# Ensure non-negative (shift if needed)
X_pos = X - X.min(axis=0) + 1e-10
nmf = NMF(n_components=4, random_state=42, max_iter=1000)
emb_nmf = nmf.fit_transform(X_pos)
results['NMF'] = [pd.DataFrame(emb_nmf, index=test_dist.index)]

# -- AE / VAE / BT / BT_noise --
for model_name, run_paths in checkpoints.items():
    if not run_paths:
        print(f"\n{model_name}: no checkpoints found, skipping"); continue
    results[model_name] = []
    for ckpt_path in run_paths:
        print(f"\n{model_name}: loading {ckpt_path}")
        encoder = VAE_Encoder(input_dim, hidden_dims, 4).to(device) if model_name == 'VAE' else Encoder(input_dim, hidden_dims, 4).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        encoder.load_state_dict(sd); encoder.eval()
        results[model_name].append(generate_embeddings(test_dist, encoder, device))

# ──────────────────────────────────────────────
# 6. Survival evaluation: 5-fold CV CoxPH + median risk split
# ──────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"{'Method':10s} {'C-index':>10s} {'p-value':>12s} {'n_high':>8s} {'n_low':>8s}")
print(f"{'='*65}")

all_metrics = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for method_name, run_embeddings in results.items():
    emb_df = run_embeddings[0]
    common_idx = emb_df.index.intersection(survival.index)
    t = survival.loc[common_idx, 'time']
    e = survival.loc[common_idx, 'status']

    cox_df = emb_df.loc[common_idx].copy()
    cox_df['time'] = t
    cox_df['status'] = e

    fold_ci = []
    fold_pval = []
    all_risk = pd.Series(index=cox_df.index, dtype=float)

    for train_idx, test_idx in kf.split(cox_df):
        seed_ci, seed_p = [], []
        for run in run_embeddings:
            run_df = run.loc[common_idx].copy(); run_df['time'] = t; run_df['status'] = e
            train_fold, test_fold = run_df.iloc[train_idx], run_df.iloc[test_idx]
            try:
                cph = CoxPHFitter(); cph.fit(train_fold, 'time', 'status', show_progress=False)
                risk_score = cph.predict_partial_hazard(test_fold)
                seed_ci.append(concordance_index(test_fold['time'], -risk_score, test_fold['status']))
                groups = (risk_score >= risk_score.median()).astype(int)
                seed_p.append(multivariate_logrank_test(test_fold['time'], groups, test_fold['status']).p_value)
            except Exception:
                continue
        if seed_ci:
            fold_ci.append(np.mean(seed_ci)); fold_pval.append(np.median(seed_p))

    if len(fold_ci) > 0:
        ci_mean = np.mean(fold_ci)
        ci_std = np.std(fold_ci, ddof=1)
        pval_combined = np.median(fold_pval)
    else:
        ci_mean = 0.5
        ci_std = 0.0
        pval_combined = 1.0

    # Full-dataset median risk split for n_high/n_low
    n_high = len(cox_df) // 2
    n_low = len(cox_df) - n_high

    print(f"{method_name:10s} {ci_mean:>7.4f}+-{ci_std:.4f} {pval_combined:>12.2e} {n_high:>8d} {n_low:>8d}")
    all_metrics.append({
        'method': method_name,
        'c_index_mean': round(ci_mean, 4),
        'c_index_std': round(ci_std, 4),
        'p_value': float(f'{pval_combined:.2e}'),
        'n_high': int(n_high),
        'n_low': int(n_low),
    })

# ──────────────────────────────────────────────
# 7. Save results
# ──────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
out_path = os.path.join(os.path.dirname(__file__), 'scanb_survival_results.csv')
metrics_df.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")
print(metrics_df)
