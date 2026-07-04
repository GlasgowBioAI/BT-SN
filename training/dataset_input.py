"""
Data loading and structural noise distortion generation.

Provides:
  - DynamicBarlowDataset: on-the-fly distortion generation during training
  - Barlow_dataloader: data pipeline for TCGA / METABRIC
  - scanb_Barlow_dataloader: data pipeline for SCAN-B
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from barlow_config import config


def get_scaler(method):
    if method == 'standard':
        return StandardScaler()
    elif method == 'minmax':
        return MinMaxScaler()
    elif method in ('None', None):
        return None
    raise ValueError(f"Unknown scaling method: {method}")


def apply_scaling_bulk(bulk_data, scaling_method):
    if scaling_method in (None, 'None'):
        return bulk_data
    scaler = MinMaxScaler() if scaling_method == 'minmax' else StandardScaler()
    scaled = scaler.fit_transform(bulk_data)
    return pd.DataFrame(scaled, index=bulk_data.index, columns=bulk_data.columns)


# ── Structural noise: weighted normal-bulk mixing (NumPy) ──

def generate_weighted_normal_bulk_numpy(normal_bulk_arr, num_samples=20):
    """Generate one weighted pseudo-normal row by sampling & averaging normal profiles."""
    N, G = normal_bulk_arr.shape
    idx = np.random.randint(0, N, size=num_samples)
    sampled = normal_bulk_arr[idx]
    weights = np.random.rand(num_samples).astype(np.float32)
    weights /= weights.sum()
    weighted = weights @ sampled
    return weighted.reshape(1, G).astype(np.float32)


# ── Distortion set generation (shared by both data loaders) ──

def gen_distortion_set_numpy(X_df, normal_bulk, config):
    """
    Generate 11 distortion levels (lambda 0.0-1.0) for eval samples.
    Each level mixes X_df with weighted normal-bulk at ratio lambda.
    Returns list of 11 Tensors, each shape [N, G].
    """
    X_arr = X_df.values.astype(np.float32)
    N, G = X_arr.shape
    normal_arr = normal_bulk.values.astype(np.float32)

    # Generate noise array: one weighted normal per sample
    w1_list = []
    for _ in range(N):
        w = generate_weighted_normal_bulk_numpy(normal_arr, num_samples=20)
        w1_list.append(w.reshape(G))
    w1_arr = np.vstack(w1_list).astype(np.float32)

    rates = np.arange(0.0, 1.01, 0.1, dtype=np.float32)
    distortion_list = []
    for rate in rates:
        tmp = (1.0 - rate) * X_arr + rate * w1_arr
        tmp = np.log2(tmp + 1.0).astype(np.float32)
        distortion_list.append(torch.from_numpy(tmp))
    return distortion_list


# ── On-the-fly distortion dataset (for training) ──

class DynamicBarlowDataset(Dataset):
    """Returns (log2(x+1), sample_idx) for training or evaluation."""

    def __init__(self, bulk_df, normal_bulk_df, lambda1, lambda2, config, mode='train'):
        self.bulk_arr = bulk_df.values.astype(np.float32)
        self.normal_arr = normal_bulk_df.values.astype(np.float32)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.config = config
        self.mode = mode
        self.N = len(bulk_df)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.bulk_arr[idx]
        if self.mode == 'eval':
            return torch.from_numpy(np.log2(x + 1.0)), idx
        # Training: return raw expression (mixing + log2 done in training loop)
        return torch.from_numpy(x), idx


def numpy_collate_fn(batch):
    """Collate function: stacks bulk tensors and returns indices."""
    bulks = [item[0] for item in batch]
    idxs = [item[1] for item in batch]
    return torch.stack(bulks, dim=0), torch.tensor(idxs, dtype=torch.long)


# ── Data loaders ──

def Barlow_dataloader(bulk_path, bulk_id_path, survival_path,
                      normal_bulk_path, batchsize, lambda_rate1, lambda_rate2, test_ratio):
    """Data loader for TCGA / METABRIC. Merges bulk + normal, returns datasets + distortions."""
    print(f"Loading {config['testing_dataset_name']}...")

    # Load labels & survival
    survival_df = pd.read_csv(survival_path, sep=',', index_col=0).dropna().sort_index()
    bulk_labels = pd.read_csv(config['bulk_label_path'], index_col=0)
    bulk_labels = bulk_labels[~bulk_labels['Pam50Subtype'].isin(['NC', 'Normal'])]

    # Load expression data
    normal_bulk = pd.read_csv(normal_bulk_path, index_col=0)
    bulk = pd.read_csv(bulk_path, sep=',', index_col=0)
    bulk = bulk.apply(lambda r: r.fillna(r.mean()), axis=1)

    # Filter common genes and samples
    bulk = bulk.loc[:, ~bulk.columns.duplicated()]
    normal_bulk = normal_bulk.loc[:, ~normal_bulk.columns.duplicated()]
    common_idx = bulk.index.intersection(bulk_labels.index)
    bulk = bulk.loc[common_idx]
    bulk_labels = bulk_labels.loc[common_idx]

    # Merge bulk + normal
    normal_labels = pd.DataFrame({'Pam50Subtype': 'PN'}, index=normal_bulk.index)
    merged_bulk = pd.concat([bulk, normal_bulk]).sort_index()
    merged_labels = pd.concat([bulk_labels, normal_labels]).sort_index()
    survival_df = survival_df.loc[merged_bulk.index.intersection(survival_df.index)].sort_index()

    return _build_dataloaders(merged_bulk, normal_bulk, merged_labels, survival_df, batchsize, lambda_rate1, lambda_rate2)


def scanb_Barlow_dataloader(bulk_path, bulk_id_path, normal_bulk_path,
                            batchsize, lambda_rate1, lambda_rate2):
    """Data loader for SCAN-B."""
    print("Loading SCAN-B...")
    labels = pd.read_csv(config['bulk_label_path'], index_col=0)
    merged_labels = labels[['NCN.PAM50']].rename(columns={'NCN.PAM50': 'Pam50Subtype'})
    survival_df = labels[['OS_days', 'OS_event']].rename(columns={'OS_days': 'time', 'OS_event': 'status'})

    bulk = pd.read_csv(bulk_path, index_col=0)
    normal_bulk = pd.read_csv(normal_bulk_path, index_col=0)
    merged_bulk = pd.concat([bulk, normal_bulk]).sort_index()

    return _build_dataloaders(merged_bulk, normal_bulk, merged_labels, survival_df, batchsize, lambda_rate1, lambda_rate2)


def _build_dataloaders(merged_bulk, normal_bulk, merged_labels, survival_df, batchsize, lambda_rate1, lambda_rate2):
    """Common routine: create datasets, dataloaders, and distortion set."""
    train_dataset = DynamicBarlowDataset(merged_bulk, normal_bulk, lambda_rate1, lambda_rate2, config, mode='train')
    eval_dataset = DynamicBarlowDataset(merged_bulk, normal_bulk, lambda_rate1, lambda_rate2, config, mode='eval')

    train_dl = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0,
                          pin_memory=True, collate_fn=numpy_collate_fn)
    eval_dl = DataLoader(eval_dataset, batch_size=batchsize, shuffle=False, num_workers=0,
                         pin_memory=True, collate_fn=numpy_collate_fn)

    distortion_list = gen_distortion_set_numpy(merged_bulk, normal_bulk, config)

    return (train_dl, eval_dl, merged_bulk, merged_labels, survival_df, distortion_list, normal_bulk)
