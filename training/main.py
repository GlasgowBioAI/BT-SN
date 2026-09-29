"""Train BT-SN using the architecture and augmentation reported in the paper."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from barlow_config import config as DEFAULT_CONFIG
from common import set_seed
from model_structure import BarlowTwinsLoss, Encoder, Projector
from training import compute_regularization_loss, sample_normal_profiles


class ExpressionDataset(Dataset):
    def __init__(self, frame): self.values = torch.as_tensor(frame.values, dtype=torch.float32)
    def __len__(self): return len(self.values)
    def __getitem__(self, index): return self.values[index]


def read_expression(path):
    frame = pd.read_csv(path, index_col=0)
    if frame.empty: raise ValueError(f"No expression profiles found in {path}")
    frame = frame.apply(lambda row: row.fillna(row.mean()), axis=1).astype("float32")
    if (frame.values < 0).any(): raise ValueError("Input must be non-negative linear-scale expression")
    return frame


def make_views(tumour, normal, alpha_min, alpha_max):
    """Independently sample one normal profile and tumour fraction for each view."""
    size, shape = tumour.shape[0], (tumour.shape[0], 1)
    n1, n2 = sample_normal_profiles(normal, size), sample_normal_profiles(normal, size)
    a1 = torch.empty(shape, device=tumour.device).uniform_(alpha_min, alpha_max)
    a2 = torch.empty(shape, device=tumour.device).uniform_(alpha_min, alpha_max)
    v1 = torch.log2((a1 * tumour + (1 - a1) * n1).clamp_min(0) + 1)
    v2 = torch.log2((a2 * tumour + (1 - a2) * n2).clamp_min(0) + 1)
    return v1, v2


def run_epoch(encoder, projector, criterion, loader, normal, cfg, device, optimizer=None):
    training = optimizer is not None
    encoder.train(training); projector.train(training)
    total = count = 0
    with (torch.enable_grad() if training else torch.no_grad()):
        for tumour in loader:
            tumour = tumour.to(device)
            v1, v2 = make_views(tumour, normal, cfg["alpha_min"], cfg["alpha_max"])
            loss, _ = criterion(projector(encoder(v1)), projector(encoder(v2)))
            loss += compute_regularization_loss(encoder, projector, cfg["regularization"], loss,
                                                cfg["l1_regularization_lambda"], cfg["l2_regularization_lambda"])
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(tumour); count += len(tumour)
    return total / max(count, 1)


def train_one_seed(tumour, normal_frame, cfg, seed, output_dir, device):
    set_seed(seed)
    train_i, val_i = train_test_split(np.arange(len(tumour)), test_size=0.1, random_state=seed)
    train_loader = DataLoader(ExpressionDataset(tumour.iloc[train_i]), cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(ExpressionDataset(tumour.iloc[val_i]), cfg["batch_size"], shuffle=False)
    normal = torch.as_tensor(normal_frame.values, dtype=torch.float32, device=device)
    encoder = Encoder(tumour.shape[1], cfg["encoder_hidden_dims"], cfg["encoder_output_dim"]).to(device)
    projector = Projector(cfg["encoder_output_dim"], cfg["projector_hidden_dim"], cfg["projector_output_dim"]).to(device)
    criterion = BarlowTwinsLoss(cfg["lambda_param"], cfg["projector_output_dim"]).to(device)
    optimizer = torch.optim.Adam([*encoder.parameters(), *projector.parameters()], lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg["lr_decay_factor"],
                                                            patience=cfg["lr_patience"], min_lr=cfg["min_lr"])
    run_dir = output_dir / f"seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    for epoch in range(1, cfg["num_epochs"] + 1):
        tr = run_epoch(encoder, projector, criterion, train_loader, normal, cfg, device, optimizer)
        va = run_epoch(encoder, projector, criterion, val_loader, normal, cfg, device)
        scheduler.step(va); history.append((epoch, tr, va))
        if va < best:
            best = va
            torch.save(encoder.state_dict(), run_dir / "encoder.pt")
            torch.save(projector.state_dict(), run_dir / "projector.pt")
    pd.DataFrame(history, columns=["epoch", "training_loss", "validation_loss"]).to_csv(run_dir / "loss_history.csv", index=False)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({**cfg, "device": str(cfg.get("device", "auto")), "seed": seed,
                   "best_validation_loss": best}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="example")
    parser.add_argument("--tumour", type=Path); parser.add_argument("--normal", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--epochs", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    tumour_path = args.tumour or ROOT / "data" / args.dataset / "train_tumour.csv"
    normal_path = args.normal or ROOT / "data" / args.dataset / "train_adjacent_normal.csv"
    output = args.output or ROOT / "outputs" / args.dataset / "bt_sn"
    cfg = dict(DEFAULT_CONFIG)
    if args.epochs is not None: cfg["num_epochs"] = args.epochs
    tumour, normal = read_expression(tumour_path), read_expression(normal_path)
    if list(tumour.columns) != list(normal.columns): raise ValueError("Tumour and normal gene columns must match")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); output.mkdir(parents=True, exist_ok=True)
    for seed in (args.seeds or cfg["seeds"]): train_one_seed(tumour, normal, cfg, seed, output, device)


if __name__ == "__main__": main()
