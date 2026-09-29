"""Generate a small synthetic dataset and exercise the BT-SN workflow."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "evaluation" / "survival"))
from barlow_config import config
from common import Encoder
from main import train_one_seed
from common_survival import cross_validated_cindex, heldout_risk_groups


def main():
    rng = np.random.default_rng(42)
    genes = [f"gene_{i:03d}" for i in range(64)]
    tumour_ids = [f"tumour_{i:03d}" for i in range(48)]
    normal_ids = [f"normal_{i:03d}" for i in range(16)]
    labels = np.repeat([1, 2, 3], 16)
    tumour = rng.gamma(2.0, 4.0, (48, 64))
    tumour[:, :8] += labels[:, None] * 2.5
    normal = rng.gamma(2.0, 3.0, (16, 64))
    tumour_df = pd.DataFrame(tumour, index=tumour_ids, columns=genes)
    normal_df = pd.DataFrame(normal, index=normal_ids, columns=genes)

    data_dir = ROOT / "data" / "example"
    out_dir = ROOT / "outputs" / "example" / "bt_sn"
    data_dir.mkdir(parents=True, exist_ok=True)
    tumour_df.to_csv(data_dir / "train_tumour.csv")
    normal_df.to_csv(data_dir / "train_adjacent_normal.csv")
    survival_time = 3000 - labels * 250 + rng.normal(0, 250, len(labels))
    events = rng.binomial(1, 0.7, len(labels))
    pd.DataFrame({"grade": labels, "time": survival_time, "status": events},
                 index=tumour_ids).to_csv(data_dir / "labels.csv")

    cfg = dict(config)
    cfg.update({"encoder_hidden_dims": [32, 16, 8], "projector_hidden_dim": 8,
                "projector_output_dim": 16, "batch_size": 16, "num_epochs": 2})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds, embeddings = [42, 43, 44], []
    for seed in seeds:
        train_one_seed(tumour_df, normal_df, cfg, seed, out_dir, device)
        encoder = Encoder(64, cfg["encoder_hidden_dims"], 4).to(device)
        encoder.load_state_dict(torch.load(out_dir / f"seed_{seed}" / "encoder.pt",
                                           map_location=device, weights_only=True))
        encoder.eval()
        with torch.no_grad():
            emb = encoder(torch.tensor(np.log2(tumour + 1), dtype=torch.float32,
                                       device=device)).cpu().numpy()
        embeddings.append(emb)
        pd.DataFrame(emb, index=tumour_ids).to_csv(out_dir / f"embeddings_seed_{seed}.csv")

    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(embeddings[0], labels))
    fold_scores = []
    for train_index, test_index in folds:
        seed_scores = []
        for seed, emb in zip(seeds, embeddings):
            classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            classifier.fit(emb[train_index], labels[train_index])
            seed_scores.append(f1_score(labels[test_index], classifier.predict(emb[test_index]),
                                        average="macro"))
        fold_scores.append(np.mean(seed_scores))
    print(f"Example workflow completed; grade macro-F1={np.mean(fold_scores):.3f} +/- {np.std(fold_scores, ddof=1):.3f}")
    survival = pd.DataFrame({"time": survival_time, "status": events}, index=tumour_ids)
    run_frames = [pd.DataFrame(embedding, index=tumour_ids) for embedding in embeddings]
    cindices = cross_validated_cindex(run_frames, survival)
    print(f"Survival C-index={cindices.mean():.3f} +/- {cindices.std(ddof=1):.3f}")
    train_ids, test_ids = tumour_ids[:32], tumour_ids[32:]
    _, groups, p_value = heldout_risk_groups(
        [frame.loc[train_ids] for frame in run_frames],
        [frame.loc[test_ids] for frame in run_frames],
        survival.loc[train_ids], survival.loc[test_ids]
    )
    print(f"Held-out KM split: high={int(groups.sum())}, low={int((~groups).sum())}, p={p_value:.3g}")


if __name__ == "__main__":
    main()
