from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 160
    batch_size: int = 32
    num_workers: int = 2
    seed: int = 42
    scratch_epochs: int = 8
    transfer_epochs: int = 5
    fine_tune_epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


def ensure_output_dirs() -> None:
    for path in (METRICS_DIR, FIGURES_DIR, CHECKPOINT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def select_device(preferred: str | None = None) -> torch.device:
    requested = preferred or os.environ.get("OXPETS_DEVICE")
    if requested:
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
