# BT-SN: Barlow Twins with Structured Noise for Cancer Gene Expression

**Barlow Twins self-supervised learning with structured noise to create robust cancer sample embeddings**

Gene expression data from biopsy remains widely used for studying cancer at scale. However, tumor biopsies inevitably contain adjacent normal tissue, introducing contamination that can mask tumor-specific signals. BT-SN (Barlow Twins with Structural Noise) is a self-supervised framework that adapts the Barlow Twins objective to bulk cancer transcriptomics by mixing tumour profiles with adjacent-normal samples during training.

BT-SN demonstrates robustness to non-tumor contamination across three independent breast cancer cohorts (SCAN-B, METABRIC, and TCGA-BRCA), achieving strong performance on cancer grade classification and survival prediction. The latent dimensions show high independence and correspond to biologically coherent transcriptional programs.

## Repository structure

```
BT-SN/
├── training/
│   ├── common.py                  Shared encoder, embedding, and loss functions
│   ├── barlow_config.py           Hyperparameter configuration
│   ├── dataset_input.py           Data loading with structural noise augmentation
│   ├── main.py                    Training entry point (all methods)
│   ├── model_structure.py         Model architectures and loss functions
│   └── training.py                Training loops for all methods
│
├── evaluation/
│   ├── grade/                     Grade classification (SCAN-B, METABRIC)
│   ├── survival/                  Survival analysis + KM curves + forest plot
│   ├── pathway/                   GSEA pathway enrichment (GO-BP)
│   └── gene_independence/         Dimension independence via gene overlap
│
└── paths.py                       Centralised data and checkpoint paths
```
## Datasets

| Dataset | Description |
|---------|-------------|
| **SCAN-B** | RNA-seq, 8,269 tumour samples + 66 normal (training/test split per original study) |
| **METABRIC** | Microarray, 997 discovery + 995 validation + 144 normal |
| **TCGA-BRCA** | RNA-seq, 814 tumour + 113 normal (80/20 split) |

All three datasets are processed to a unified gene space of 16,736 protein-coding genes. No cross-cohort normalisation is applied.

## Training

### Requirements

- Python 3.8+
- PyTorch ≥ 1.12
- NumPy, pandas, scikit-learn
- Lifelines
- Matplotlib, scipy

### Running training

```bash
cd training
python main.py                          # BT-SN (structural noise)
python main.py --gaussian               # BT-GN (Gaussian noise)
```

### Supported methods

BT-SN is compared against several baselines. `model_type` in `barlow_config.py` selects the method:

| Method | Key | Description |
|--------|-----|-------------|
| BT-SN | `barlow_twins` | Structural noise augmentation (ours) |
| BT-GN | `barlow_twins` + `--gaussian` | Gaussian noise augmentation (ours) |
| AE | `auto_encoder` | Autoencoder baseline |
| VAE | `VAE` | Variational autoencoder baseline |
| SCARF | `scarf` | Contrastive learning with feature corruption |
| VIME | `vime` | Self-supervised mask prediction |
| BYOL | `byol` | Bootstrap Your Own Latent |

## Evaluation

### Grade classification

5-fold cross-validated logistic regression on clean ($\lambda=0.0$) training embeddings, evaluated on test embeddings at 11 contamination levels ($\lambda = 0.0$ to $1.0$). Reports macro-averaged F1 score.

```bash
cd evaluation/grade
python scanb_eval.py
python metabric_eval.py
```

### Survival analysis

Cox proportional hazards model fitted on training set embeddings. Risk scores predicted on held-out test set. Reports C-index and log-rank p-value.

```bash
cd evaluation/survival
python scanb_eval.py
python metabric_eval.py
python tcga_eval.py
```

Kaplan–Meier curves and hazard ratio forest plot:
```bash
python km_scanb_panels.py
python forest_plot.py
```

### Pathway enrichment

Spearman correlation between each embedding dimension and all genes, followed by GSEA Preranked (GO Biological Process, 1000 permutations).

```bash
cd evaluation/pathway
python run_bt_sn_fast.py           # BT-SN only (SCAN-B, quick)
python run_pipeline.py              # All 9 methods (full pipeline)
```

### Gene independence

Pairwise Jaccard of top-500 positively and bottom-500 negatively correlated genes per dimension. Lower Jaccard = more independent dimensions.

```bash
cd evaluation/gene_independence
python scanb.py
python metabric.py
```

## Reproducibility

All absolute paths are centralised in `paths.py`. Override the project root with the environment variable `BT_SN_ROOT`, or data root with `BT_SN_DATA`.

Checkpoints referenced in `paths.py` must be placed at the expected locations. The `CHECKPOINTS` dictionary maps method and dataset keys to checkpoint paths.

