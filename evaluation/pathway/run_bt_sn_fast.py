#!/usr/bin/env python3
"""SCAN-B BT-SN pathway: Spearman (parallel threads) → GSEA."""
import os, sys, torch, torch.nn as nn, numpy as np, pandas as pd, time, subprocess
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys as _sys, os as _os
_bt_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), '..', 'training')
_sys.path.insert(0, _bt_path)
from paths import BASE, CHECKPOINTS, p
from common import Encoder, VAE_Encoder, build_hidden_dims, generate_embeddings
import torch

from paths import BASE, CHECKPOINTS, p
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT = os.path.join(BASE, 'BIBM_evaluation', 'Pathway_independence')
RNK_DIR = os.path.join(OUT, 'rnk_files')
GSEA_DIR = os.path.join(OUT, 'gsea_output')
for d in [RNK_DIR, GSEA_DIR]: os.makedirs(d, exist_ok=True)

GMT = os.path.join(BASE, 'LaCoGSEA', 'lacogsea', 'data', 'go_bp.gmt')
GSEA_CLI = os.path.join(BASE, 'LaCoGSEA', 'lacogsea', 'vendor', 'GSEA_4.4.0', 'gsea-cli.bat')
GSEA_BIN = os.path.dirname(GSEA_CLI)

def spearman_one(gene, z_arr, gv_arr):
    mask = ~np.isnan(z_arr) & ~np.isnan(gv_arr)
    if mask.sum() < 3: return (gene, np.nan)
    rho, _ = stats.spearmanr(z_arr[mask], gv_arr[mask])
    return (gene, rho)

if __name__ == '__main__':
    # Load expression
    print("Loading SCAN-B test...")
    fn = p('scanb_data/scanb_2022/data_normal_bulk_distort/test/normal_bulk_distortion_test_full_set_0.0.txt')
    expr = pd.read_csv(fn, index_col=0, sep=',')
    expr.replace([np.inf,-np.inf],np.nan,inplace=True)
    expr = expr.apply(lambda c: c.fillna(c.mean()), axis=0)
    print(f"  {expr.shape[0]} x {expr.shape[1]}")

    # Load encoder
    print("Loading BT-SN encoder...")
    ckpt = CHECKPOINTS['BT_scanb']
    idim = expr.shape[1]; hd = [1024]
    for _ in range(1,3):
        n = hd[-1]//2
        if n<4: break
        hd.append(n)
    enc = Encoder(idim, hd, 4).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd = {k.replace('_orig_mod.',''): v for k,v in sd.items()}
    enc.load_state_dict(sd, strict=False); enc.eval()

    emb = generate_embeddings(expr, enc)
    print(f"  Embedding: {emb.shape}")

    # Step 1: Spearman correlation (ThreadPool, not ProcessPool)
    nw = min(os.cpu_count() or 8, 16)
    print(f"Spearman with {nw} thread workers...")
    for dim in emb.columns:
        z = emb[dim].reindex(expr.index).values
        t0 = time.time()
        genes = list(expr.columns)
        rhos = {}; done = 0
        with ThreadPoolExecutor(max_workers=nw) as ex:
            fut = {ex.submit(spearman_one, g, z, expr[g].values): g for g in genes}
            for f in as_completed(fut):
                g, r = f.result()
                if not np.isnan(r): rhos[g] = r
                done += 1
                if done % 3000 == 0:
                    print(f"  dim{dim}: {done}/{len(genes)} ({time.time()-t0:.0f}s)")
        cs = pd.Series(rhos).dropna().sort_values(ascending=False)
        mdir = os.path.join(RNK_DIR, 'SCAN-B', 'BT')
        os.makedirs(mdir, exist_ok=True)
        cs.to_csv(os.path.join(mdir, f"Scanb_BT_{dim}_corr_rank.rnk"), sep='\t', header=False)
        print(f"  dim{dim}: {len(cs)} genes done ({time.time()-t0:.0f}s)")

    # Step 2: Java GSEA (4 dims)
    print("\nRunning Java GSEA...")
    jc = os.path.join(os.path.dirname(GSEA_CLI), 'jdk', 'bin', 'java.exe')
    if not os.path.exists(jc): jc = 'java'
    mp = os.path.join(GSEA_BIN, 'modules')
    lp = os.path.join(GSEA_BIN, 'logging.properties')

    for dim in range(4):
        rnk = os.path.join(RNK_DIR, 'SCAN-B', 'BT', f"Scanb_BT_{dim}_corr_rank.rnk")
        od = os.path.join(GSEA_DIR, 'SCAN-B', 'BT', f"Scanb_BT_{dim}_corr_rank")
        if os.path.exists(os.path.join(od, 'index.html')):
            print(f"  dim{dim}: already done, skip")
            continue
        os.makedirs(od, exist_ok=True)
        print(f"  dim{dim}: GSEA...", end='', flush=True)
        tt = time.time()
        cmd = [jc, f'--module-path={mp}', '-Xmx4g', '-Djava.awt.headless=true',
               f'-Djava.util.logging.config.file={lp}',
               '--module=org.gsea_msigdb.gsea/xapps.gsea.CLI', 'GSEAPreranked',
               '-rnk', rnk, '-gmx', GMT, '-out', od,
               '-rpt_label', f"Scanb_BT_{dim}_corr_rank",
               '-plot_top_x', '0', '-set_max', '500', '-set_min', '15',
               '-nperm', '1000', '-scoring_scheme', 'weighted',
               '-make_sets', 'true', '-rnd_seed', '42', '-zip_report', 'false']
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=GSEA_BIN)
            if r.returncode == 0 and os.path.exists(os.path.join(od, 'index.html')):
                print(f" done ({time.time()-tt:.0f}s)")
            else:
                print(f" FAILED: {(r.stderr or '')[-200:]}")
        except subprocess.TimeoutExpired: print(f" TIMEOUT")
        except Exception as e: print(f" ERROR: {e}")

    print(f"\nDone! Results in {os.path.join(GSEA_DIR, 'SCAN-B', 'BT')}")
