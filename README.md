# BT-SN

Code accompanying **Barlow Twins self-supervised learning with structured noise to create robust cancer sample embeddings**.

BT-SN learns cancer gene-expression representations that are robust to non-tumour contamination. During training, two views of each tumour profile are generated independently using adjacent-normal expression. A shared encoder and projector are optimized with the Barlow Twins objective. After training, the projector is discarded and the encoder produces the representations used in downstream analyses.

## Installation

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively, create the complete Conda environment with `conda env create -f environment.yml`.

## Example workflow

The example creates a small synthetic dataset, trains BT-SN, saves the learned representations, and exercises the grade and survival evaluation workflow.

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

Run the training workflow separately for each cohort:

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

The training settings used in the article are defined in `training/barlow_config.py`.

## Evaluation

Evaluation code is organized by analysis:

```text
evaluation/grade/              grade classification and contamination series
evaluation/survival/           Cox proportional-hazards and Kaplan--Meier analyses
evaluation/gene_independence/  gene-level overlap between latent dimensions
evaluation/pathway/            GO Biological Process enrichment and semantic redundancy
```

Checkpoint locations are repository-relative and defined in `training/paths.py`. The evaluation scripts compare BT-SN with the baseline representation methods described in the article.

The pathway workflow requires a local GSEA installation and the GO Biological Process gene-set file. Analysis settings are documented in the manuscript and implemented in the corresponding scripts.

The held-out SCAN-B Kaplan--Meier workflow fits Cox models using the training representations, predicts risk in the test set, and divides the test samples into risk groups:

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
- Each view is generated independently during training.
- Downstream analyses use the encoder output; the projector is discarded.
- Independent training runs are combined within downstream evaluation folds as described in the article.

## License

This project is released under the MIT License. See `LICENSE` for details.
