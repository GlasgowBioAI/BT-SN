"""Repository-relative data, output, and checkpoint locations.

Place downloaded data under ``data/`` and trained weights under ``checkpoints/``.
No machine-specific absolute paths are required.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT
DATA = ROOT / "data"
CHECKPOINT_ROOT = ROOT / "checkpoints"


def p(relative):
    """Return a repository-relative data path as a string."""
    return str(DATA / relative)


CHECKPOINTS = {
    f"{method}_{cohort}": str(CHECKPOINT_ROOT / cohort / method / "encoder.pt")
    for cohort in ("scanb", "metabric", "tcga")
    for method in ("AE", "VAE", "BT", "SCARF", "BYOL", "VIME", "BT_noise")
}


def checkpoint_runs(method, cohort):
    """Return all seed-specific encoder checkpoints, with legacy fallback."""
    run_root = CHECKPOINT_ROOT / cohort / method
    runs = sorted(run_root.glob("seed_*/encoder.pt"))
    if runs:
        return [str(path) for path in runs]
    legacy = Path(CHECKPOINTS[f"{method}_{cohort}"])
    return [str(legacy)] if legacy.exists() else []
