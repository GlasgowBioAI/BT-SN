"""
Generate publication-grade Macro-F1 line charts for Grade classification
(SCANB & METABRIC) with separate legend file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# ── global style ─────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'lines.linewidth': 2.2,
    'lines.markersize': 6,
})

# ── distinctive, colorblind-friendly palette ─────────────────────────
colors = {
    'BT':        '#D62728',   # strong red
    'BT_noise':  '#FF9896',   # light red / pink
    'AE':        '#FF7F0E',   # vibrant orange
    'VAE':       '#9467BD',   # purple
    'PCA':       '#1F77B4',   # blue
    'NMF':       '#2CA02C',   # green
    'SCARF':     '#17BECF',   # cyan
    'BYOL':      '#8C564B',   # brown
    'VIME':      '#7F7F7F',   # grey
}

display_names = {
    'BT':        'BT-SN',
    'BT_noise':  'BT-GN',
    'PCA':       'PCA',
    'NMF':       'NMF',
    'AE':        'AE',
    'VAE':       'VAE',
    'SCARF':     'SCARF',
    'BYOL':      'BYOL',
    'VIME':      'VIME',
}

# ── line style: make BT stand out with larger marker, others subtle ──
line_styles = {
    'BT':        {'marker': 'o', 'markersize': 8, 'linewidth': 3.0, 'markerfacecolor': 'white', 'markeredgewidth': 2.0},
    'BT_noise':  {'marker': 's', 'markersize': 5, 'linewidth': 1.8, 'linestyle': '--'},
    'AE':        {'marker': '^', 'markersize': 5, 'linewidth': 1.8},
    'VAE':       {'marker': 'v', 'markersize': 5, 'linewidth': 1.8},
    'PCA':       {'marker': 'D', 'markersize': 5, 'linewidth': 1.8},
    'NMF':       {'marker': 's', 'markersize': 5, 'linewidth': 1.8},
    'SCARF':     {'marker': 'p', 'markersize': 5, 'linewidth': 1.8},
    'BYOL':      {'marker': '*', 'markersize': 6, 'linewidth': 1.8},
    'VIME':      {'marker': 'h', 'markersize': 5, 'linewidth': 1.8},
}

# ── order for legend ─────────────────────────────────────────────────
legend_order = ['BT', 'BT_noise', 'AE', 'VAE', 'PCA', 'NMF', 'SCARF', 'BYOL', 'VIME']

# ── process each dataset ─────────────────────────────────────────────
datasets = [
    ('scanb',    'SCAN-B'),
    ('metabric', 'METABRIC'),
]

for subdir, display_name in datasets:
    csv_path = os.path.join(subdir, 'grade_macro_F1_summary.csv')
    if not os.path.exists(csv_path):
        print(f'[SKIP] {csv_path} not found')
        continue

    df = pd.read_csv(csv_path)

    lam_cols = [c for c in df.columns if c.startswith('lam_')]
    lambdas  = np.array([float(c.replace('lam_', '')) for c in lam_cols])
    methods  = df['method'].unique()

    # ── figure (no legend) ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for method in methods:
        mean_row = df[(df['method'] == method) & (df['stat'] == 'mean')]
        std_row  = df[(df['method'] == method) & (df['stat'] == 'std')]
        if mean_row.empty or std_row.empty:
            continue

        means = mean_row.iloc[0][lam_cols].astype(float).values
        stds  = std_row.iloc[0][lam_cols].astype(float).values

        color = colors.get(method, '#333333')
        style = line_styles.get(method, {})

        # cap outlier std values for visual clarity
        stds_plot = np.minimum(stds, 0.1)

        # draw line + markers (no label — legend is separate)
        ax.plot(lambdas, means,
                color=color,
                **style)

        # error band
        ax.fill_between(lambdas,
                        means - stds_plot,
                        means + stds_plot,
                        color=color, alpha=0.12)

    # ── labels & layout ──────────────────────────────────────────────
    ax.set_xlabel('Contamination level', fontweight='bold')
    ax.set_ylabel('Macro-F1', fontweight='bold')
    ax.set_title(f'{display_name} Grade', fontweight='bold', pad=12)

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xticklabels([f'{t:.1f}' for t in np.arange(0, 1.01, 0.1)])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}'))

    ax.grid(True, linestyle='--', alpha=0.35)

    # ── save main figure (no legend) ────────────────────────────────
    os.makedirs(subdir, exist_ok=True)
    png_path = os.path.join(subdir, 'grade_macroF1_vs_contamination.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    print(f'[OK] Saved -> {png_path}')
    plt.close()

# ── generate separate legend file ────────────────────────────────────
fig_leg, ax_leg = plt.subplots(figsize=(8, 1.2))
ax_leg.axis('off')

# build legend handles in desired order
handles = []
labels = []
for m in legend_order:
    if m not in colors:
        continue
    style = line_styles.get(m, {})
    color = colors[m]
    label = display_names.get(m, m)
    h = mpl.lines.Line2D([], [], color=color, label=label, **style)
    handles.append(h)
    labels.append(label)

legend = ax_leg.legend(handles, labels,
                       loc='center',
                       ncol=5,
                       frameon=True,
                       fancybox=False,
                       edgecolor='#cccccc',
                       columnspacing=1.5,
                       handletextpad=0.8)

plt.savefig(os.path.join('scanb', 'grade_legend.png'), dpi=600, bbox_inches='tight')
print('[OK] Saved -> scanb/grade_legend.png')
plt.close()

print('Done.')
