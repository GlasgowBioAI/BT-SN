#!/usr/bin/env python3
"""Metabric Grade Classification with Distortion Dilution (lam=0.0~1.0)."""
import os, torch, numpy as np, pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import sys as _sys, os as _os
_bt_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), '..', 'training')
_sys.path.insert(0, _bt_path)
from paths import BASE, CHECKPOINTS, p
from common import Encoder, VAE_Encoder, build_hidden_dims as build_arch, generate_embeddings as gen_emb
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ──
TRAIN_CSV  = p('/train_labels.csv')
TEST_CSV   = p('/test_labels.csv')
NORMAL_1   = p('/pyega3/EGA_metabric_normal_discovery.csv')
NORMAL_2   = p('/pyega3/EGA_metabric_normal_validation.csv')
DISTORT_DIR = p('data_normal_bulk_distort')
TRAIN_DIST  = os.path.join(DISTORT_DIR, 'train/normal_bulk_distortion_train_fullset.txt')
TEST_DIST_TP = os.path.join(DISTORT_DIR, 'test/normal_bulk_distortion_test_{}_fullset.txt')

# ── Labels ──
train_labels = pd.read_csv(TRAIN_CSV, index_col=0).sort_index()
test_labels = pd.read_csv(TEST_CSV, index_col=0).sort_index()
normal_bulk = pd.read_csv(NORMAL_1, index_col=0, sep=',')
normal_test = pd.read_csv(NORMAL_2, index_col=0, sep=',')
Normal_bulk = pd.concat([normal_bulk, normal_test], axis=1)

train_grade = train_labels[['grade']].copy()
train_grade[train_grade.index.isin(Normal_bulk.index)] = 0.0
train_grade = train_grade.dropna()
test_grade = test_labels[['grade']].copy()
test_grade[test_grade.index.isin(Normal_bulk.index)] = 0.0
test_grade = test_grade.dropna()
y_train = train_grade['grade']
print(f"Train grade: {train_grade.shape}, Test: {test_grade.shape}")
print(f"Grade distribution: {y_train.value_counts().sort_index().to_dict()}")

# ── Distortion data ──
lams = [f"{l:.1f}" for l in np.arange(0.0, 1.1, 0.1)]
train_dist = pd.read_csv(TRAIN_DIST, index_col=0, sep=',').sort_index().fillna(0)
train_dist = train_dist.loc[train_grade.index]
test_dfs = {lam: pd.read_csv(TEST_DIST_TP.format(lam), index_col=0, sep=',').sort_index().fillna(0) for lam in lams}

# ── Checkpoints ──
checkpoints = {
    'AE': CHECKPOINTS['AE_metabric'],
    'VAE': CHECKPOINTS['VAE_metabric'],
    'BT': CHECKPOINTS['BT_metabric'],
    'SCARF': CHECKPOINTS['SCARF_metabric'],
    'BYOL': CHECKPOINTS['BYOL_metabric'],
    'VIME': CHECKPOINTS['VIME_metabric'],
    'BT_noise': CHECKPOINTS['BT_noise_metabric'],
}

# ── Embeddings ──
input_dim = train_dist.shape[1]
hidden_dims = build_arch(input_dim)
results = {}

# PCA
pca = PCA(4, random_state=42)
train_pca = pd.DataFrame(pca.fit_transform(train_dist.values), index=train_dist.index)
results['PCA'] = {'train': train_pca}
for lam in lams:
    results['PCA'][lam] = pd.DataFrame(pca.transform(test_dfs[lam].values), index=test_dfs[lam].index)

# NMF
Xp = train_dist.values - train_dist.values.min(axis=0) + 1e-10
nmf = NMF(4, random_state=42, max_iter=1000)
train_nmf = pd.DataFrame(nmf.fit_transform(Xp), index=train_dist.index)
results['NMF'] = {'train': train_nmf}
for lam in lams:
    results['NMF'][lam] = pd.DataFrame(nmf.transform(test_dfs[lam].values - test_dfs[lam].values.min(axis=0) + 1e-10), index=test_dfs[lam].index)

# AE / VAE / BT
for name, ckpt in checkpoints.items():
    if not ckpt or not os.path.exists(ckpt): print(f"{name}: SKIP"); continue
    print(f"{name}: loading")
    enc = VAE_Encoder(input_dim, hidden_dims, 4).to(device) if name == 'VAE' else Encoder(input_dim, hidden_dims, 4).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    enc.load_state_dict(sd); enc.eval()
    results[name] = {'train': gen_emb(train_dist, enc)}
    for lam in lams:
        results[name][lam] = gen_emb(test_dfs[lam], enc)

# ── 5-fold CV LogReg ──
print(f"\n{'='*70}")
print(f"{'Method':10s} {'0.0':>7s} {'0.1':>7s} {'0.2':>7s} {'0.3':>7s} {'0.4':>7s} {'0.5':>7s} {'0.6':>7s} {'0.7':>7s} {'0.8':>7s} {'0.9':>7s} {'1.0':>7s}")
print(f"{'='*70}")

all_results = {}
for method, ed in results.items():
    X_tr = ed['train'].values; y = y_train.values
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = []
    for tidx, _ in skf.split(X_tr, y):
        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        clf.fit(X_tr[tidx], y[tidx]); folds.append(clf)

    f1_mean = {}; f1_std = {}
    for lam in lams:
        te = ed[lam]
        te_gt = test_grade['grade'].reindex(te.index).dropna()
        te_X = te.loc[te_gt.index]
        fold_f1s = [f1_score(te_gt, clf.predict(te_X.values), average='macro') for clf in folds]
        f1_mean[lam] = round(np.mean(fold_f1s), 4)
        f1_std[lam] = round(np.std(fold_f1s, ddof=1), 4)
    print(f"{method:10s} {'  '.join(f'{f1_mean[lam]:.4f}' for lam in lams)}")
    print(f"{'':10s} {'  '.join(f'{f1_std[lam]:.4f}' for lam in lams)}")
    all_results[method] = {lam: (f1_mean[lam], f1_std[lam]) for lam in lams}

# ── Save ──
rows = []
for m, sc in all_results.items():
    rm = {'method': m, 'stat': 'mean'}; rs = {'method': m, 'stat': 'std'}
    for lam in lams: rm[f'lam_{lam}'] = sc[lam][0]; rs[f'lam_{lam}'] = sc[lam][1]
    rows.append(rm); rows.append(rs)
pd.DataFrame(rows).to_csv(os.path.join(SCRIPT_DIR, 'grade_macro_F1_summary.csv'), index=False)
print("Saved")
