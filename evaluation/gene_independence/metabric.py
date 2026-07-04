#!/usr/bin/env python3
"""Metabric Gene Independence: pairwise Jaccard of top500+bottom500 genes per dim."""
import os, torch, numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import NMF, PCA
from itertools import combinations

import sys as _sys, os as _os
_bt_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), '..', 'training')
_sys.path.insert(0, _bt_path)
from paths import BASE, CHECKPOINTS, p
from common import Encoder, VAE_Encoder, generate_embeddings
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_BOTTOM = 500

# ── Load data ──
cancer = pd.read_csv(p('different_mixture_data/Metabric_evaluation_fine_tune/data_normal_bulk_distort/test/normal_bulk_distortion_test_0.0_fullset.txt'), index_col=0, sep=',').sort_index()
cancer = cancer.apply(lambda c: c.fillna(c.mean()), axis=0)
print(f"Cancer data: {cancer.shape}")

# ── Checkpoints ──
hd = [1024, 512, 256]
checkpoints = {
    'BT':       CHECKPOINTS['BT_metabric'],
    'BT_noise': CHECKPOINTS['BT_noise_metabric'],
    'AE':       CHECKPOINTS['AE_metabric'],
    'VAE':      CHECKPOINTS['VAE_metabric'],
    'SCARF':    CHECKPOINTS['SCARF_metabric'],
    'VIME':     CHECKPOINTS['VIME_metabric'],
    'BYOL':     CHECKPOINTS['BYOL_metabric'],
}

# ── Embeddings ──
embs = {}
Xp = cancer.values - cancer.values.min(axis=0) + 1e-10
embs['NMF'] = pd.DataFrame(NMF(4, random_state=42, max_iter=1000).fit_transform(Xp), index=cancer.index, columns=list(range(4)))
embs['PCA'] = pd.DataFrame(PCA(4, random_state=42).fit_transform(cancer.values), index=cancer.index, columns=list(range(4)))

input_dim = cancer.shape[1]
for name, ckpt in checkpoints.items():
    if not ckpt or not os.path.exists(ckpt): print(f"{name}: SKIP"); continue
    print(f"{name}: loading...")
    enc = (VAE_Encoder(input_dim, hd, 4) if name == 'VAE' else Encoder(input_dim, hd, 4)).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    enc.load_state_dict(sd); enc.eval()
    embs[name] = generate_embeddings(cancer, enc)
    del enc; torch.cuda.empty_cache()

# ── Spearman ──
print("\nComputing Spearman...")
corr_results = {}
for mname, emb_df in embs.items():
    print(f"  {mname}")
    corr_results[mname] = {}
    for col in emb_df.columns:
        z = emb_df[col].values
        rd = {}
        for gene in cancer.columns:
            gv = cancer[gene].values
            m = np.isfinite(z) & np.isfinite(gv)
            if m.sum() < 3: continue
            r, _ = stats.spearmanr(z[m], gv[m])
            if np.isfinite(r): rd[gene] = r
        corr_results[mname][col] = pd.Series(rd).sort_values(ascending=False)

# ── Jaccard ──
print(f"\nPairwise Jaccard (top{TOP_BOTTOM} ∪ bottom{TOP_BOTTOM} per dim)...")
jaccard_rows = []
for mname, dc in corr_results.items():
    dims = sorted(dc.keys())
    combined = {}
    for d in dims:
        ranked = dc[d]
        combined[d] = set(ranked.index[:TOP_BOTTOM]) | set(ranked.index[-TOP_BOTTOM:])
    jaccards = {(d1, d2): len(combined[d1] & combined[d2]) / max(len(combined[d1] | combined[d2]), 1) for d1, d2 in combinations(dims, 2)}
    mean_j = np.mean(list(jaccards.values()))
    row = {'method': mname, 'mean_jaccard': round(mean_j, 4)}
    row.update({f'J_{d1}{d2}': round(v, 4) for (d1, d2), v in jaccards.items()})
    jaccard_rows.append(row)
    print(f"  {mname:10s}: mean_J={mean_j:.4f}")

jaccard_df = pd.DataFrame(jaccard_rows).set_index('method').sort_values('mean_jaccard')
jaccard_df.to_csv(os.path.join(SCRIPT_DIR, 'pairwise_jaccard_top_bottom500.csv'))
print(f"\n{jaccard_df.round(4).to_string()}")
