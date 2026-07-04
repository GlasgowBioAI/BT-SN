#!/usr/bin/env python3
"""
Pathway Independence Pipeline 
=====================================
For METABRIC / TCGA / SCAN-B, each with 9 methods:
  PCA, NMF, AE, VAE, BT, BT-noise, SCARF, BYOL, VIME

Steps:
  1. Generate embeddings → Spearman rho per gene per dim → .rnk files
  2. Java GSEA (GO-BP, 1000 perm, 15-500)
  3. Compute weighted pathway independence (FDR<0.05/4)
  4. Figures and tables → BIBM-ready

Output directory: BIBM_evaluation/Pathway_independence/
"""

import os, sys, torch, torch.nn as nn, numpy as np, pandas as pd, warnings, time, shutil, glob, re
from scipy import stats
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA, NMF
from itertools import combinations
import matplotlib.pyplot as plt
import mygene
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════
# GLOBAL PATHS
# ═══════════════════════════════════════════════════════
OUT = os.path.join(BASE, 'BIBM_evaluation', 'Pathway_independence')
os.makedirs(OUT, exist_ok=True)

# Subdirs
RNK_DIR   = os.path.join(OUT, 'rnk_files')
GSEA_DIR  = os.path.join(OUT, 'gsea_output')
FIG_DIR   = os.path.join(OUT, 'figures')

for d in [RNK_DIR, GSEA_DIR, FIG_DIR]: os.makedirs(d, exist_ok=True)

# Colab-like GO-BP gmt
GMT = os.path.join(BASE, 'LaCoGSEA', 'lacogsea', 'data', 'go_bp.gmt')
GSEA_CLI = os.path.join(BASE, 'LaCoGSEA', 'lacogsea', 'vendor', 'GSEA_4.4.0', 'gsea-cli.bat')
GSEA_DIR_BIN = os.path.dirname(GSEA_CLI)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════
# DATA CONFIG
# ═══════════════════════════════════════════════════════
DATASETS = {
    'METABRIC': {
        'expr_file': p('Metabric_evaluation/normal_bulk_distortion_test_0.0_fullset.txt'),
        'rnk_prefix': 'Metabric',
    },
    'TCGA': {
        'expr_file': p('/TCGA_evaluation/normal_bulk_distortion_test_0.0_fullset.txt'),
        'rnk_prefix': 'TCGA',
    },
    'SCAN-B': {
        'expr_file': p('/scanb_data/normal_bulk_distortion_test_full_set_0.0.txt'),
        'rnk_prefix': 'Scanb',
    },
}

# Checkpoints — all 9 methods × 3 datasets
CKPTS = {
    'AE': {
        'METABRIC': CHECKPOINTS['AE_metabric'],
        'TCGA': CHECKPOINTS['AE_tcga'],
        'SCAN-B': CHECKPOINTS['AE_scanb'],
    },
    'VAE': {
        'METABRIC': CHECKPOINTS['VAE_metabric'],
        'TCGA': CHECKPOINTS['VAE_tcga'],
        'SCAN-B': CHECKPOINTS['VAE_scanb'],
    },
    'BT': {
        'METABRIC': CHECKPOINTS['BT_metabric'],
        'TCGA': CHECKPOINTS['BT_tcga'],
        'SCAN-B': CHECKPOINTS['BT_scanb'],
    },
    'BT_noise': {
        'METABRIC': CHECKPOINTS['BT_noise_metabric'],
        'TCGA': CHECKPOINTS['BT_noise_tcga'],
        'SCAN-B': CHECKPOINTS['BT_noise_scanb'],
    },
    'SCARF': {
        'METABRIC': CHECKPOINTS['SCARF_metabric'],
        'TCGA': CHECKPOINTS['SCARF_tcga'],
        'SCAN-B': CHECKPOINTS['SCARF_scanb'],
    },
    'BYOL': {
        'METABRIC': CHECKPOINTS['BYOL_metabric'],
        'TCGA': CHECKPOINTS['BYOL_tcga'],
        'SCAN-B': CHECKPOINTS['BYOL_scanb'],
    },
    'VIME': {
        'METABRIC': CHECKPOINTS['VIME_metabric'],
        'TCGA': CHECKPOINTS['VIME_tcga'],
        'SCAN-B': CHECKPOINTS['VIME_scanb'],
    },
}

ALL_METHODS = ['PCA', 'NMF', 'AE', 'VAE', 'BT', 'BT_noise', 'SCARF', 'BYOL', 'VIME']

MODEL_DISPLAY = {
    'PCA': 'PCA', 'NMF': 'NMF', 'AE': 'AE', 'VAE': 'VAE',
    'BT': 'TP', 'BT_noise': 'BT-noisy',
    'SCARF': 'SCARF', 'BYOL': 'BYOL', 'VIME': 'VIME'
}
COLORS = {
    'PCA': '#1f77b4', 'NMF': '#ff7f0e', 'AE': '#2ca02c', 'VAE': '#d62728',
    'TP': '#9467bd', 'BT-noisy': '#8c564b',
    'SCARF': '#e377c2', 'BYOL': '#7f7f7f', 'VIME': '#bcbd22'
}
MODEL_ORDER = ['PCA', 'NMF', 'AE', 'VAE', 'TP', 'BT-noisy', 'SCARF', 'BYOL', 'VIME']

# ═══════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════
class Encoder(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        ad = [i] + h + [o]; layers = nn.Sequential()
        for j in range(len(ad) - 1):
            layers.add_module(f"l{j}", nn.Linear(ad[j], ad[j+1]))
            if j < len(ad) - 2:
                layers.add_module(f"b{j}", nn.BatchNorm1d(ad[j+1]))
                layers.add_module(f"r{j}", nn.ReLU())
        self.net = layers
    def forward(self, x): return self.net(x)

class VAE_Enc(nn.Module):
    def __init__(self, i, h, l):
        super().__init__()
        dims = [i] + h; shared = nn.Sequential()
        for j in range(len(dims)-1):
            shared.add_module(f"l{j}", nn.Linear(dims[j], dims[j+1]))
            shared.add_module(f"b{j}", nn.BatchNorm1d(dims[j+1]))
            shared.add_module(f"r{j}", nn.ReLU())
        self.shared = shared; self.fc_mu = nn.Linear(h[-1], l)
    def forward(self, x): return self.fc_mu(self.shared(x))

def convert_ensembl_to_symbol(expr):
    """Convert Ensembl ID column names to gene symbols using mygene."""
    ensembl_ids = [c.split('.')[0] if '.' in str(c) else str(c) for c in expr.columns]
    mg = mygene.MyGeneInfo()
    query = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human', returnall=True)
    id_to_symbol = {q['query']: q['symbol'] for q in query['out'] if 'symbol' in q and q['symbol']}
    new_cols = []
    keep = []
    for i, eid in enumerate(ensembl_ids):
        if eid in id_to_symbol:
            new_cols.append(id_to_symbol[eid])
            keep.append(i)
        else:
            new_cols.append(eid)
            keep.append(i)
    expr = expr.iloc[:, keep]
    expr.columns = new_cols
    expr = expr.loc[:, ~expr.columns.duplicated(keep='first')]
    return expr

def build_hidden(idim, odim=4):
    h = [1024]
    for _ in range(1, 3):
        n = h[-1] // 2
        if n < odim: break
        h.append(n)
    return h

def gen_emb(df, model, bs=256):
    t = torch.tensor(df.values, dtype=torch.float32)
    loader = DataLoader(t, batch_size=bs, shuffle=False)
    embs = []
    with torch.no_grad():
        for b in loader: embs.append(model(b.to(DEVICE)).cpu())
    return pd.DataFrame(torch.cat(embs, dim=0).numpy(), index=df.index)

def load_ckpt(ckpt_path, model_class, idim, hd, odim=4):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model = model_class(idim, hd, odim).to(DEVICE)
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model

# ═══════════════════════════════════════════════════════
# STEP 1: GENERATE .rnk FILES
# ═══════════════════════════════════════════════════════
def step1_generate_rnk():
    """For each dataset x method, compute embedding -> Spearman rho -> .rnk"""
    print("="*60)
    print("  STEP 1: Generate .rnk files")
    print("="*60)

    for dset_name, dcfg in DATASETS.items():
        print(f"\n  [{dset_name}]")
        expr_file = dcfg['expr_file']
        if not os.path.exists(expr_file):
            print(f"    SKIP: {expr_file} not found")
            continue

        expr = pd.read_csv(expr_file, index_col=0, sep=',').sort_index()
        expr.replace([np.inf, -np.inf], np.nan, inplace=True)
        expr = expr.apply(lambda c: c.fillna(c.mean()), axis=0)
        expr = convert_ensembl_to_symbol(expr)
        print(f"    Expression: {expr.shape[0]} samples x {expr.shape[1]} genes")

        prefix = dcfg['rnk_prefix']
        idim = expr.shape[1]
        hd = build_hidden(idim)

        for method in ALL_METHODS:
            print(f"    {method:12s}: ", end='', flush=True)

            if method == 'PCA':
                pca = PCA(4, random_state=42)
                emb = pd.DataFrame(pca.fit_transform(expr.values), index=expr.index)
            elif method == 'NMF':
                Xp = expr.values - expr.values.min(axis=0) + 1e-10
                nmf = NMF(4, random_state=42, max_iter=1000)
                emb = pd.DataFrame(nmf.fit_transform(Xp), index=expr.index)
            else:
                ckpt = CKPTS[method].get(dset_name, '')
                if not ckpt or not os.path.exists(ckpt):
                    print(f"ckpt not found, skip")
                    continue
                enc_class = VAE_Enc if method == 'VAE' else Encoder
                try:
                    enc = load_ckpt(ckpt, enc_class, idim, hd)
                    emb = gen_emb(expr, enc)
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

            # Compute Spearman per dim
            mdir = os.path.join(RNK_DIR, dset_name, method)
            os.makedirs(mdir, exist_ok=True)

            for dim in emb.columns:
                z = emb[dim].reindex(expr.index).values
                rhos = {}
                for gene in expr.columns:
                    gv = expr[gene].values
                    mask = ~np.isnan(z) & ~np.isnan(gv)
                    if mask.sum() < 3: continue
                    rho, _ = stats.spearmanr(z[mask], gv[mask])
                    rhos[gene] = rho
                cs = pd.Series(rhos).dropna().sort_values(ascending=False)
                fn = os.path.join(mdir, f"{prefix}_{method}_{dim}_corr_rank.rnk")
                cs.to_csv(fn, sep='\t', header=False)

            ngenes = len(cs)
            print(f"dims x {ngenes} genes  [done]")

    print(f"\n  All .rnk files -> {RNK_DIR}")


# ═══════════════════════════════════════════════════════
# STEP 2: RUN JAVA GSEA
# ═══════════════════════════════════════════════════════
def step2_run_gsea():
    """For each .rnk file, run Java GSEA Preranked (GO-BP, 1000 perm, 15-500)"""
    print("\n" + "="*60)
    print("  STEP 2: Run Java GSEA")
    print("="*60)

    java_cmd = os.path.join(os.path.dirname(GSEA_CLI), 'jdk', 'bin', 'java.exe')
    if not os.path.exists(java_cmd):
        java_cmd = 'java'
    modules_path = os.path.join(GSEA_DIR_BIN, 'modules')
    log_props = os.path.join(GSEA_DIR_BIN, 'logging.properties')

    # Collect all .rnk files
    all_rnk = []
    for dset_name in DATASETS:
        dset_rnk = os.path.join(RNK_DIR, dset_name)
        if not os.path.exists(dset_rnk): continue
        for method in os.listdir(dset_rnk):
            mdir = os.path.join(dset_rnk, method)
            if not os.path.isdir(mdir): continue
            for fn in sorted(os.listdir(mdir)):
                if fn.endswith('.rnk'):
                    all_rnk.append((dset_name, method, os.path.join(mdir, fn)))

    print(f"  Total .rnk files: {len(all_rnk)}")

    for dset_name, method, rnk_file in all_rnk:
        basename = os.path.splitext(os.path.basename(rnk_file))[0]
        out_dir = os.path.join(GSEA_DIR, dset_name, method, basename)

        # Skip if already done
        if os.path.exists(os.path.join(out_dir, 'index.html')):
            print(f"  [{dset_name}/{method}] {basename}: already done, skip")
            continue

        print(f"  [{dset_name}/{method}] {basename}: running GSEA...", end='', flush=True)
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            java_cmd, f'--module-path={modules_path}',
            '-Xmx4g', '-Djava.awt.headless=true',
            f'-Djava.util.logging.config.file={log_props}',
            '--module=org.gsea_msigdb.gsea/xapps.gsea.CLI',
            'GSEAPreranked',
            '-rnk', rnk_file,
            '-gmx', GMT,
            '-out', out_dir,
            '-rpt_label', basename,
            '-plot_top_x', '0',
            '-set_max', '500',
            '-set_min', '15',
            '-nperm', '1000',
            '-scoring_scheme', 'weighted',
            '-make_sets', 'true',
            '-rnd_seed', '42',
            '-zip_report', 'false'
        ]

        t0 = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=7200, shell=False, cwd=GSEA_DIR_BIN)
            elapsed = time.time() - t0
            if result.returncode == 0 and os.path.exists(os.path.join(out_dir, 'index.html')):
                print(f" done ({elapsed:.0f}s)")
            else:
                err = (result.stderr or '')[-200:]
                print(f" FAILED (rc={result.returncode}): {err}")
        except subprocess.TimeoutExpired:
            print(f" TIMEOUT (>2h)")
        except Exception as e:
            print(f" ERROR: {e}")

    print(f"\n  All GSEA runs complete -> {GSEA_DIR}")


# ═══════════════════════════════════════════════════════
# STEP 3: PARSE GSEA RESULTS & COMPUTE WEIGHTED INDEPENDENCE
# ═══════════════════════════════════════════════════════
FDR_CUTOFF = 0.05 / 4  # 0.0125

def parse_gsea_report(tsv_path):
    """Parse a single GSEA report .tsv, return significant pathways list."""
    if not os.path.exists(tsv_path): return pd.DataFrame()
    try:
        df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
        if 'NAME' not in df.columns: return pd.DataFrame()
        for c in ['NES', 'FDR q-val']:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except: return pd.DataFrame()

def get_significant_pathways(gsea_dir, fdr_cutoff=FDR_CUTOFF):
    """
    For a given model x dim GSEA run, return set of significant pathway names
    (FDR < fdr_cutoff, from both pos and neg reports).
    """
    all_rows = []
    for direction in ['pos', 'neg']:
        pat = os.path.join(gsea_dir, f'gsea_report_for_na_{direction}_*.tsv')
        files = glob.glob(pat)
        for fp in files:
            df = parse_gsea_report(fp)
            if not df.empty:
                df_sig = df[df['FDR q-val'] < fdr_cutoff].copy()
                if not df_sig.empty:
                    all_rows.append(df_sig)
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()

def weighted_independence(path_dict, n_perm=1000):
    """
    Per-dim unique_frac, then weighted average (weighted by pathway count in each dim).
    """
    dims = sorted(path_dict.keys())
    rng = np.random.RandomState(42)
    all_pathways = list(set().union(*path_dict.values()))
    if not all_pathways: return {'model_independence': np.nan, 'per_dim': {}}

    dim_results = {}
    for d in dims:
        this = path_dict[d]
        others = set().union(*[path_dict[od] for od in dims if od != d])
        n_unique = len(this - others)
        uf = n_unique / len(this) if len(this) > 0 else 0

        perm_ufs = []
        for _ in range(n_perm):
            perm_this = set(rng.choice(all_pathways, len(this), replace=False))
            perm_others = set(rng.choice(all_pathways, len(others), replace=False))
            perm_ufs.append(len(perm_this - perm_others) / len(perm_this)) if len(perm_this) > 0 else perm_ufs.append(0)
        pv = (np.sum(np.array(perm_ufs) >= uf) + 1) / (n_perm + 1)
        dim_results[d] = {'unique_frac': round(uf, 4), 'p_value': round(pv, 6), 'n_pathways': len(this)}

    # Weighted average
    scores = np.array([v['unique_frac'] for v in dim_results.values()])
    weights = np.array([v['n_pathways'] for v in dim_results.values()])
    wmean = np.average(scores, weights=weights) if weights.sum() > 0 else scores.mean()

    pvals = [v['p_value'] for v in dim_results.values()]
    comb_p = min(min(pvals) * len(pvals), 1.0)

    return {'model_independence': round(wmean, 4), 'model_p_value': round(comb_p, 6),
            'n_dims': len(dim_results), 'per_dim': dim_results}


def step3_analyze():
    """Parse all GSEA results and compute pathway independence."""
    print("\n" + "="*60)
    print("  STEP 3: Compute Pathway Independence Scores")
    print("="*60)

    all_results = {}
    all_per_dim = []
    all_rep_paths = []

    for dset_name in DATASETS:
        print(f"\n  [{dset_name}]")
        dset_gsea = os.path.join(GSEA_DIR, dset_name)
        if not os.path.exists(dset_gsea): continue

        for method in ALL_METHODS:
            display = MODEL_DISPLAY.get(method, method)
            method_dir = os.path.join(dset_gsea, method)
            if not os.path.exists(method_dir): continue

            # Collect significant pathways per dim
            path_dict = {}
            for dim in range(4):
                dim_label = f"{DATASETS[dset_name]['rnk_prefix']}_{method}_{dim}_corr_rank"
                run_dir = os.path.join(method_dir, dim_label)
                if not os.path.exists(run_dir): continue
                df_sig = get_significant_pathways(run_dir)
                if df_sig.empty: continue
                pathways = set(df_sig['NAME'].dropna().unique())
                if len(pathways) >= 3:
                    path_dict[dim] = pathways

            dim_info = ', '.join([f"dim{k}: {len(v)}" for k, v in sorted(path_dict.items())])
            n_total = sum(len(v) for v in path_dict.values())

            if len(path_dict) < 2 or n_total == 0:
                print(f"    {display:12s}: no sig pathways (FDR<{FDR_CUTOFF})")
                continue

            indep = weighted_independence(path_dict)
            print(f"    {display:12s}: {n_total:4d} sig ({dim_info})", end='')
            print(f" -> w-indep={indep['model_independence']:.3f} (p={indep['model_p_value']})")

            all_results[(dset_name, display)] = {
                'pathway_independence': indep['model_independence'],
                'p_value': indep['model_p_value'],
                'n_total': n_total,
                'n_dims': indep['n_dims'],
                'per_dim': indep['per_dim']
            }

            # Per-dim table
            for d_str, v in indep['per_dim'].items():
                all_per_dim.append({'Dataset': dset_name, 'Model': display,
                                    'Dimension': f'dim{d_str}', 'Unique_Frac': v['unique_frac'],
                                    'P_Value': v['p_value'], 'N_Pathways': v['n_pathways']})

            # Representative pathways
            for d in sorted(path_dict.keys()):
                this = path_dict[d]
                others = set().union(*[path_dict[od] for od in path_dict if od != d])
                unique = this - others
                for pw in sorted(unique)[:5]:
                    all_rep_paths.append({'Dataset': dset_name, 'Model': display,
                                          'Dimension': f'dim{d}', 'Pathway': pw})

    # Save tables
    # 1. Summary
    summary = []
    for (dset, model), m in all_results.items():
        summary.append({'Dataset': dset, 'Model': model,
                        'Pathway_Independence': m['pathway_independence'],
                        'P_Value': m['p_value'], 'Total_Sig': m['n_total'],
                        'Dims_Valid': m['n_dims']})
    pd.DataFrame(summary).to_csv(os.path.join(OUT, 'pathway_independence_scores.csv'), index=False)
    print(f"\n  Saved: pathway_independence_scores.csv")

    # 2. Per-dim
    pd.DataFrame(all_per_dim).to_csv(os.path.join(OUT, 'per_dim_independence.csv'), index=False)
    print(f"  Saved: per_dim_independence.csv")

    # 3. Representative pathways
    if all_rep_paths:
        rep_df = pd.DataFrame(all_rep_paths)
        rep_df['Pathway_Clean'] = rep_df['Pathway'].str.replace('GOBP_', '').str.replace('_', ' ').str.title()
        rep_df.to_csv(os.path.join(OUT, 'representative_pathways.csv'), index=False)
        print(f"  Saved: representative_pathways.csv")

    return all_results


# ═══════════════════════════════════════════════════════
# STEP 4: FIGURES
# ═══════════════════════════════════════════════════════
def step4_figures(all_results):
    print("\n" + "="*60)
    print("  STEP 4: Generate Figures")
    print("="*60)

    summary = pd.read_csv(os.path.join(OUT, 'pathway_independence_scores.csv'))
    per_dim = pd.read_csv(os.path.join(OUT, 'per_dim_independence.csv'))

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 11, 'axes.linewidth': 1.0})

    # ─── Fig 1: Horizontal bars per dataset ───
    datasets_list = ['METABRIC', 'TCGA', 'SCAN-B']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('Pathway Independence (Weighted, FDR<0.0125)', fontsize=16, fontweight='bold', y=1.02)

    for idx, dset in enumerate(datasets_list):
        ax = axes[idx]
        data = summary[summary['Dataset'] == dset].copy()
        existing = [m for m in MODEL_ORDER if m in data['Model'].values]
        data = data.set_index('Model').loc[existing].reset_index()
        y = np.arange(len(data))
        scores = data['Pathway_Independence'].values
        ax.barh(y, scores, 0.5, color=[COLORS.get(m, '#888') for m in data['Model']],
                edgecolor='white', linewidth=0.5, zorder=3)
        for i, (s, pv) in enumerate(zip(scores, data['P_Value'].values)):
            if pv < 0.01:
                ax.text(s + 0.01, i, '**', va='center', fontsize=9, fontweight='bold')
            elif pv < 0.05:
                ax.text(s + 0.01, i, '*', va='center', fontsize=9, fontweight='bold')
        ax.set_yticks(y); ax.set_yticklabels(data['Model'], fontsize=10)
        ax.set_xlim(0, 1.08); ax.set_xlabel('Pathway Independence (weighted)', fontsize=10)
        ax.set_title(dset, fontsize=12, fontweight='bold')
        ax.axvline(0.5, color='gray', ls='--', lw=0.5, alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        for i, (_, row) in enumerate(data.iterrows()):
            if row['Dims_Valid'] > 0:
                ax.text(0.01, i - 0.15, f'n={int(row["Total_Sig"])}', va='bottom', fontsize=7, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'pathway_independence_figure.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'pathway_independence_figure.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 1: pathway_independence_figure.pdf/png")

    # ─── Fig 2: Per-dim heatmap ───
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    fig.suptitle('Per-Dimension Unique Fraction', fontsize=16, fontweight='bold', y=1.02)

    for idx, dset in enumerate(datasets_list):
        ax = axes[idx]
        pdd = per_dim[per_dim['Dataset'] == dset].copy()
        if pdd.empty:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            continue
        pivot = pdd.pivot_table(index='Model', columns='Dimension', values='Unique_Frac', aggfunc='first')
        existing = [m for m in MODEL_ORDER if m in pivot.index]
        pivot = pivot.loc[existing] if existing else pivot
        for d in [f'dim{i}' for i in range(4)]:
            if d not in pivot.columns: pivot[d] = np.nan
        pivot = pivot[[f'dim{i}' for i in range(4)]]

        im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='white' if v > 0.6 else 'black')
        ax.set_xticks(range(4)); ax.set_xticklabels([f'dim{i}' for i in range(4)], fontsize=9)
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title(dset, fontsize=11, fontweight='bold')
        if idx == 2:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Unique Frac')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'per_dim_heatmap.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIG_DIR, 'per_dim_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 2: per_dim_heatmap.pdf/png")

    print(f"\n  All figures saved to {FIG_DIR}")


# ═══════════════════════════════════════════════════════
# STEP: GOSEMSIM SIMPLIFICATION (via R script)
# ═══════════════════════════════════════════════════════
SIMPLIFIED_DIR = os.path.join(OUT, 'simplified_gobp_cut06')
VENN_CSV = os.path.join(OUT, 'venn_data_cut06.csv')
FIG_VENN_DIR = os.path.join(OUT, 'figures_venn')
R_SCRIPT = os.path.join(BASE, 'TwinPurify_code', 'simplify_gsea_msigdb_go.R')
SIMILARITY_CUTOFF = 0.6

def step_gosemsim():
    """Run GOSemSim simplification on each GSEA output via R script."""
    print("\n" + "="*60)
    print("  STEP: GOSemSim Simplification (cutoff={})".format(SIMILARITY_CUTOFF))
    print("="*60)
    os.makedirs(SIMPLIFIED_DIR, exist_ok=True)

    for dset_name in DATASETS:
        dset_gsea = os.path.join(GSEA_DIR, dset_name)
        if not os.path.exists(dset_gsea): continue
        prefix = DATASETS[dset_name]['rnk_prefix']
        for method in ALL_METHODS:
            for dim in range(4):
                dim_label = f"{prefix}_{method}_{dim}_corr_rank"
                run_dir = os.path.join(dset_gsea, method, dim_label)
                if not os.path.exists(run_dir): continue
                for direction in ['pos', 'neg']:
                    pat = os.path.join(run_dir, f'gsea_report_for_na_{direction}_*.tsv')
                    files = glob.glob(pat)
                    if not files: continue
                    tsv_file = files[0]
                    out_csv = os.path.join(SIMPLIFIED_DIR, f'simplified_GO_{method}_dim{dim}_{direction}.csv')
                    if os.path.exists(out_csv):
                        print(f"  [{dset_name}/{method}] dim{dim} {direction}: already done, skip")
                        continue
                    print(f"  [{dset_name}/{method}] dim{dim} {direction}: running GOSemSim...", end='', flush=True)
                    cmd = ['Rscript', R_SCRIPT, tsv_file, out_csv, 'FWER', '0.01', str(SIMILARITY_CUTOFF)]
                    try:
                        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=SIMPLIFIED_DIR)
                        if r.returncode == 0 and os.path.exists(out_csv):
                            print(f" done")
                        else:
                            print(f" FAILED: {(r.stderr or '')[-100:]}")
                    except Exception as e:
                        print(f" ERROR: {e}")
    print(f"\n  All simplified -> {SIMPLIFIED_DIR}")


# ═══════════════════════════════════════════════════════
# STEP: GENERATE VENN DATA from GOSemSim outputs
# ═══════════════════════════════════════════════════════
def step_venn_data():
    """From simplified_gobp_cut06/, compute per-method dimension intersections → venn_data_cut06.csv."""
    print("\n" + "="*60)
    print("  STEP: Generate Venn Data")
    print("="*60)
    os.makedirs(FIG_VENN_DIR, exist_ok=True)

    def load_dim_terms(method, dim):
        terms = set()
        for direction in ['pos', 'neg']:
            f = os.path.join(SIMPLIFIED_DIR, f'simplified_GO_{method}_dim{dim}_{direction}.csv')
            if os.path.exists(f):
                df = pd.read_csv(f)
                if 'TERM_STRING' in df.columns:
                    terms |= set(df['TERM_STRING'].dropna().str.strip())
        return terms

    venn_rows = []
    for method in ALL_METHODS:
        dims = {d: load_dim_terms(method, d) for d in range(4)}
        if all(len(v) == 0 for v in dims.values()):
            print(f"    {method:10s}: no data, skip")
            continue

        # Compute Venn region counts
        all4 = dims[0] & dims[1] & dims[2] & dims[3]
        only = {}
        for d in range(4):
            others = set().union(*[dims[od] for od in range(4) if od != d])
            only[d] = len(dims[d] - others)

        pairs = {}
        for (i,j) in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
            inter = dims[i] & dims[j]
            other_union = set().union(*[dims[k] for k in range(4) if k != i and k != j]) if 4-2 > 0 else set()
            pairs[(i,j)] = len(inter - other_union - all4)

        triples = {}
        for i,j,k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
            triples[(i,j,k)] = len(dims[i] & dims[j] & dims[k] - all4)

        row = {
            'method': method,
            'only_D0': only[0], 'only_D1': only[1], 'only_D2': only[2], 'only_D3': only[3],
            'D0D1': pairs[(0,1)], 'D0D2': pairs[(0,2)], 'D0D3': pairs[(0,3)],
            'D1D2': pairs[(1,2)], 'D1D3': pairs[(1,3)], 'D2D3': pairs[(2,3)],
            'D0D1D2': triples[(0,1,2)], 'D0D1D3': triples[(0,1,3)], 'D0D2D3': triples[(0,2,3)], 'D1D2D3': triples[(1,2,3)],
            'all4': len(all4),
            'total': sum(len(dims[d]) for d in range(4))  # total across all dims
        }
        venn_rows.append(row)
        print(f"    {method:10s}: D0={only[0]} D1={only[1]} D2={only[2]} D3={only[3]} (4-set={len(all4)})")

    venn_df = pd.DataFrame(venn_rows)
    venn_df.to_csv(VENN_CSV, index=False)
    print(f"\n  Saved: {VENN_CSV}")


# ═══════════════════════════════════════════════════════
# STEP: COMPUTE JACCARD from Venn data
# ═══════════════════════════════════════════════════════
def step_compute_jaccard():
    """Compute mean Jaccard from venn data, print summary."""
    print("\n" + "="*60)
    print("  STEP: Compute Mean Jaccard")
    print("="*60)
    vdf = pd.read_csv(VENN_CSV)
    for _, row in vdf.iterrows():
        method = row['method']
        sizes = {
            0: row['only_D0'] + row['D0D1'] + row['D0D2'] + row['D0D3'] + row['D0D1D2'] + row['D0D1D3'] + row['D0D2D3'] + row['all4'],
            1: row['only_D1'] + row['D0D1'] + row['D1D2'] + row['D1D3'] + row['D0D1D2'] + row['D0D1D3'] + row['D1D2D3'] + row['all4'],
            2: row['only_D2'] + row['D0D2'] + row['D1D2'] + row['D2D3'] + row['D0D1D2'] + row['D0D2D3'] + row['D1D2D3'] + row['all4'],
            3: row['only_D3'] + row['D0D3'] + row['D1D3'] + row['D2D3'] + row['D0D1D3'] + row['D0D2D3'] + row['D1D2D3'] + row['all4'],
        }
        pair_inter = {
            (0,1): row['D0D1'] + row['D0D1D2'] + row['D0D1D3'] + row['all4'],
            (0,2): row['D0D2'] + row['D0D1D2'] + row['D0D2D3'] + row['all4'],
            (0,3): row['D0D3'] + row['D0D1D3'] + row['D0D2D3'] + row['all4'],
            (1,2): row['D1D2'] + row['D0D1D2'] + row['D1D2D3'] + row['all4'],
            (1,3): row['D1D3'] + row['D0D1D3'] + row['D1D2D3'] + row['all4'],
            (2,3): row['D2D3'] + row['D0D2D3'] + row['D1D2D3'] + row['all4'],
        }
        jacs = []
        for (i,j), inter in pair_inter.items():
            union = sizes[i] + sizes[j] - inter
            jacs.append(inter / union if union > 0 else 0.0)
        mean_j = np.mean(jacs)
        print(f"    {method:10s}: MeanJ = {mean_j:.4f}  (by-pair: {[round(x,3) for x in jacs]})")


# ═══════════════════════════════════════════════════════
# STEP: VENN FIGURES
# ═══════════════════════════════════════════════════════
def step_venn_figures():
    """Generate 9 Venn diagrams + BT-SN combined figure."""
    print("\n" + "="*60)
    print("  STEP: Generate Venn Figures")
    print("="*60)
    # Run the existing venn_4set_plot.py
    venn_plot = os.path.join(OUT, 'venn_4set_plot.py')
    bt_plot = os.path.join(OUT, 'bt_sn_venn_table.py')
    if os.path.exists(venn_plot):
        r = subprocess.run([sys.executable, venn_plot], capture_output=True, text=True, cwd=OUT)
        if r.returncode == 0: print(f"  Venn 4-set: done")
        else: print(f"  Venn 4-set: FAILED {r.stderr[-200:]}")
    if os.path.exists(bt_plot):
        r = subprocess.run([sys.executable, bt_plot], capture_output=True, text=True, cwd=OUT)
        if r.returncode == 0: print(f"  BT-SN combined: done")
        else: print(f"  BT-SN combined: FAILED {r.stderr[-200:]}")
    print(f"  Figures -> {FIG_VENN_DIR}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
from paths import BASE, CHECKPOINTS, p
if __name__ == '__main__':
    import subprocess


    phases = sys.argv[1:] if len(sys.argv) > 1 else ['1', '2', 'gs', 'vd', 'jc', 'vf']

    if '1' in phases:
        step1_generate_rnk()
    if '2' in phases:
        step2_run_gsea()
    if 'gs' in phases:
        step_gosemsim()
    if 'vd' in phases:
        step_venn_data()
    if 'jc' in phases:
        step_compute_jaccard()
    if '3' in phases:
        results = step3_analyze()
    if 'vf' in phases:
        step_venn_figures()
    if '4' in phases:
        step4_figures(None)
    if 'all' in phases:
        step1_generate_rnk()
        step2_run_gsea()
        step_gosemsim()
        step_venn_data()
        step_compute_jaccard()
        step3_analyze()
        step_venn_figures()
        step4_figures(None)

    print(f"\n{'='*60}")
    print(f"  ALL DONE!")
    print(f"  Outputs in: {OUT}")
    print(f"{'='*60}")
