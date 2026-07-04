#!/usr/bin/env python3
"""
Compute pathway independence metrics from raw simplified_gobp_cut06/ CSVs.
For each method: D0-D3 sizes, Zero pairs, MeanJ, 3+dim terms.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

DATA_DIR = BASE / 'BIBM_evaluation' / 'Pathway_independence' / 'simplified_gobp_cut06'
OUT_CSV  = BASE / 'BIBM_evaluation' / 'Pathway_independence' / 'independence_full_metrics.csv'

METHODS = ['BT', 'SCARF', 'AE', 'VAE', 'PCA', 'NMF', 'VIME', 'BT_noise', 'BYOL']

def load_dim_terms(method, dim):
    """Load unique TERM_STRING values for a given method+dimension (pos+neg)."""
    terms = set()
    for sign in ['pos', 'neg']:
        f = DATA_DIR / f'simplified_GO_{method}_dim{dim}_{sign}.csv'
        if f.exists():
            df = pd.read_csv(f)
            if 'TERM_STRING' in df.columns:
                terms |= set(df['TERM_STRING'].dropna().str.strip())
    return terms

records = []
for method in METHODS:
    dims = {d: load_dim_terms(method, d) for d in range(4)}
    sizes = {d: len(v) for d, v in dims.items()}

    # All unique terms across all dims
    all_terms = set().union(*dims.values())
    n_total   = len(all_terms)

    # How many dims does each term appear in?
    term_dim_count = {}
    for t in all_terms:
        c = sum(1 for d in range(4) if t in dims[d])
        term_dim_count[t] = c

    # 3+dim terms
    three_plus = sum(1 for c in term_dim_count.values() if c >= 3)

    # Pairwise Jaccard for all 6 dim pairs
    pairs = list(combinations(range(4), 2))
    jaccards = {}
    for d1, d2 in pairs:
        inter = len(dims[d1] & dims[d2])
        union = len(dims[d1] | dims[d2])
        jaccards[(d1, d2)] = inter / union if union else 0

    zero_pairs = sum(1 for j in jaccards.values() if j == 0)
    mean_j     = np.mean(list(jaccards.values()))

    records.append({
        'method':    method,
        'D0':        sizes[0], 'D1': sizes[1],
        'D2':        sizes[2], 'D3': sizes[3],
        'n_total':   n_total,
        'three_plus':three_plus,
        'three_plus_pct': round(100 * three_plus / n_total, 2) if n_total else 0,
        'zero_pairs':zero_pairs,
        'mean_j':    round(mean_j, 4),
        # individual jaccards
        'J_01': round(jaccards[(0,1)], 4),
        'J_02': round(jaccards[(0,2)], 4),
        'J_03': round(jaccards[(0,3)], 4),
        'J_12': round(jaccards[(1,2)], 4),
        'J_13': round(jaccards[(1,3)], 4),
        'J_23': round(jaccards[(2,3)], 4),
    })

res = pd.DataFrame(records)
res.to_csv(OUT_CSV, index=False)

# Pretty print
print("Method   D0   D1   D2   D3  Total  3+dim  Zero  MeanJ")
print("─" * 65)
for _, r in res.iterrows():
    flag = "✅" if r['three_plus'] == 0 and r['zero_pairs'] >= 3 else ""
    print(f"{r['method']:8s} {r['D0']:4d} {r['D1']:4d} {r['D2']:4d} {r['D3']:4d}"
          f"  {r['n_total']:4d}  {r['three_plus']:3d}/{r['n_total']:3d}"
          f"  {r['zero_pairs']}/6  {r['mean_j']:.4f}  {flag}")
