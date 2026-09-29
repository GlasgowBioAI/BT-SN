"""Held-out SCAN-B Kaplan--Meier analysis using seed-averaged Cox risk."""
from pathlib import Path
import argparse, sys
import matplotlib.pyplot as plt
import pandas as pd
import torch
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Encoder, build_hidden_dims, generate_embeddings
from paths import checkpoint_runs
from common_survival import heldout_risk_groups


def load_runs(expression, checkpoints):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for checkpoint in checkpoints:
        encoder = Encoder(expression.shape[1], build_hidden_dims(expression.shape[1]), 4).to(device)
        encoder.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        encoder.eval(); runs.append(generate_embeddings(expression, encoder, device))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-expression", type=Path, required=True)
    parser.add_argument("--test-expression", type=Path, required=True)
    parser.add_argument("--train-survival", type=Path, required=True)
    parser.add_argument("--test-survival", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/scanb/km_bt_sn.pdf"))
    args = parser.parse_args()
    train_x = pd.read_csv(args.train_expression, index_col=0)
    test_x = pd.read_csv(args.test_expression, index_col=0)
    train_y = pd.read_csv(args.train_survival, index_col=0)[["time", "status"]].dropna()
    test_y = pd.read_csv(args.test_survival, index_col=0)[["time", "status"]].dropna()
    checkpoints = checkpoint_runs("BT", "scanb")
    if not checkpoints: raise FileNotFoundError("No SCAN-B BT-SN checkpoints found")
    train_runs, test_runs = load_runs(train_x, checkpoints), load_runs(test_x, checkpoints)
    _, groups, p_value = heldout_risk_groups(train_runs, test_runs, train_y, test_y)
    outcomes = test_y.loc[groups.index]
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    KaplanMeierFitter().fit(outcomes.loc[groups, "time"] / 30.44,
                           outcomes.loc[groups, "status"], label="High risk").plot(ax=ax)
    KaplanMeierFitter().fit(outcomes.loc[~groups, "time"] / 30.44,
                           outcomes.loc[~groups, "status"], label="Low risk").plot(ax=ax)
    ax.set(xlabel="Time (months)", ylabel="Survival probability",
           title=f"BT-SN (log-rank p={p_value:.2e})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.output); print(f"Saved {args.output}")


if __name__ == "__main__": main()
