"""Input and controlled-contamination utilities used by BT-SN."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_expression(path):
    """Read samples-by-genes, non-negative, linear-scale expression."""
    frame = pd.read_csv(Path(path), index_col=0)
    frame = frame.apply(lambda row: row.fillna(row.mean()), axis=1).astype("float32")
    if frame.empty or (frame.values < 0).any():
        raise ValueError("Expression must be non-empty, non-negative, and on the linear scale")
    return frame


def align_gene_columns(tumour, adjacent_normal):
    """Require identical genes and return both frames in tumour-column order."""
    if set(tumour.columns) != set(adjacent_normal.columns):
        raise ValueError("Tumour and adjacent-normal inputs must contain the same genes")
    return tumour, adjacent_normal.loc[:, tumour.columns]


class ExpressionDataset(Dataset):
    """Return untransformed profiles; augmentation precedes log transformation."""
    def __init__(self, frame):
        self.values = torch.as_tensor(frame.values, dtype=torch.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def controlled_contamination(tumour, adjacent_normal, fractions=None, seed=42):
    """Generate the held-out contamination series used for grade evaluation.

    One adjacent-normal profile is sampled for every held-out tumour and retained
    across fractions 0.0--1.0. Mixing occurs before log2(x+1).
    """
    fractions = np.arange(0.0, 1.01, 0.1) if fractions is None else np.asarray(fractions)
    rng = np.random.default_rng(seed)
    tumour_values = tumour.values.astype("float32")
    normal_values = adjacent_normal.values.astype("float32")
    sampled = normal_values[rng.integers(0, len(normal_values), size=len(tumour_values))]
    return {
        float(fraction): pd.DataFrame(
            np.log2((1.0 - fraction) * tumour_values + fraction * sampled + 1.0),
            index=tumour.index,
            columns=tumour.columns,
        )
        for fraction in fractions
    }
