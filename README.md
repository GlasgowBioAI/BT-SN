# BT-SN

Code accompanying **Barlow Twins self-supervised learning with structured noise to create robust cancer sample embeddings**.

BT-SN learns cancer gene-expression representations that are robust to non-tumour contamination. During training, each linear-scale tumour profile is independently paired with two adjacent-normal profiles. For each view, the tumour coefficient is sampled independently as

```text
alpha ~ Uniform(0.15, 0.85)
view = log2(alpha * tumour + (1 - alpha) * adjacent_normal + 1)
```

The two views pass through a shared encoder (`input -> 1024 -> 512 -> 256 -> 4`) and projector (`4 -> 128 -> 256`). The projector is optimized with the Barlow Twins objective and discarded after training.

## Installation

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively, create the complete Conda environment with `conda env create -f environment.yml`.

## Example workflow

The example creates a small synthetic dataset under `data/example/`, trains a reduced-size BT-SN model for two epochs, saves its embedding, and runs a grade-classification smoke test.

```bash
python example/run_example.py
```

This example checks program flow only; it does not reproduce the article's numerical results.

## Data layout

Downloaded data should be placed under repository-relative paths. Training expects samples in rows, genes in columns, and non-negative expression on the linear scale:

```text
data/
  scanb/
    train_tumour.csv
    train_adjacent_normal.csv
  metabric/
    train_tumour.csv
    train_adjacent_normal.csv
  tcga/
    train_tumour.csv
    train_adjacent_normal.csv
```

Tumour and adjacent-normal files must contain the same genes in the same order. No machine-specific absolute paths are required. The cohort-specific evaluation scripts retain the filenames used in the study; adjust only the repository-relative entries in `training/paths.py` if your downloaded archive uses different subdirectories.

## Training

The default command performs three independent runs with seeds 42, 43, and 44 and retains the parameter state with the lowest validation loss:

```bash
python training/main.py --dataset scanb
python training/main.py --dataset metabric
python training/main.py --dataset tcga
```

Paths and run settings can also be supplied explicitly:

```bash
python training/main.py \
  --tumour data/scanb/train_tumour.csv \
  --normal data/scanb/train_adjacent_normal.csv \
  --output outputs/scanb/bt_sn
```

The article settings are defined in `training/barlow_config.py`: Adam with learning rate `1e-3`, batch size 256, at most 1,500 epochs, `ReduceLROnPlateau` with factor 0.1 and patience 100, minimum learning rate `1e-6`, and elastic penalties `2.18e-6` (L1) and `4.50e-5` (L2).

## Evaluation

Evaluation code is organized by analysis:

```text
evaluation/grade/              four-class grade classification and contamination series
evaluation/survival/           five-fold Cox proportional-hazards evaluation
evaluation/gene_independence/  top/bottom-500 gene overlap between latent dimensions
evaluation/pathway/            GO Biological Process enrichment and semantic redundancy
```

Checkpoint locations are repository-relative and defined in `training/paths.py`. The evaluation scripts compare four-dimensional representations from PCA, NMF, AE, VAE, SCARF, VIME, BYOL, BT-GN, and BT-SN.

The pathway workflow requires a local GSEA installation and the GO Biological Process gene-set file. It uses 1,000 permutations, gene-set sizes 15--500, dimension-wise FDR threshold `0.05 / 4`, and semantic-similarity cutoff 0.6, as described in the manuscript.

The held-out SCAN-B Kaplan--Meier analysis fits one Cox model for each training seed, averages the three risk predictions on the test set, and applies a median split:

```bash
python evaluation/survival/km_scanb.py \
  --train-expression data/scanb/train_tumour_log2.csv \
  --test-expression data/scanb/test_tumour_log2.csv \
  --train-survival data/scanb/train_survival.csv \
  --test-survival data/scanb/test_survival.csv
```

## Reproducibility notes

- Training labels are not used by BT-SN.
- Adjacent-normal profiles used for training views must come from the training partition.
- Mixing is performed on the linear expression scale before `log2(x + 1)`.
- Each view samples its adjacent-normal profile and tumour coefficient independently at every epoch.
- Downstream analyses use the four-dimensional encoder output; the projector is discarded.
- Article results aggregate three independently trained representations within each downstream fold and then summarize the five fold-level values.
