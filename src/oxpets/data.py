from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from oxpets.config import DATA_DIR, TrainConfig


TaskName = Literal["breed", "species"]

BREED_CLASSES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
    "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue",
    "Siamese", "Sphynx", "american bulldog", "american pit bull terrier",
    "basset hound", "beagle", "boxer", "chihuahua", "english cocker spaniel",
    "english setter", "german shorthaired", "great pyrenees", "havanese",
    "japanese chin", "keeshond", "leonberger", "miniature pinscher",
    "newfoundland", "pomeranian", "pug", "saint bernard", "samoyed",
    "scottish terrier", "shiba inu", "staffordshire bull terrier",
    "wheaten terrier", "yorkshire terrier",
]
SPECIES_CLASSES = ["cat", "dog"]


@dataclass(frozen=True)
class TaskSpec:
    name: TaskName
    target_type: str
    num_classes: int
    class_names: list[str]


TASKS: dict[TaskName, TaskSpec] = {
    "breed": TaskSpec("breed", "category", 37, BREED_CLASSES),
    "species": TaskSpec("species", "binary-category", 2, SPECIES_CLASSES),
}


def image_transforms(image_size: int, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


def make_dataset(
    task: TaskName,
    split: Literal["trainval", "test"],
    image_size: int,
    train: bool,
    root: Path = DATA_DIR,
    download: bool = False,
):
    spec = TASKS[task]
    return datasets.OxfordIIITPet(
        root=root,
        split=split,
        target_types=spec.target_type,
        transform=image_transforms(image_size, train=train),
        download=download,
    )


def _extract_targets(dataset, task: TaskName) -> list[int]:
    if task == "species" and hasattr(dataset, "_bin_labels"):
        labels = dataset._bin_labels
    elif hasattr(dataset, "_labels"):
        labels = dataset._labels
    elif hasattr(dataset, "_bin_labels"):
        labels = dataset._bin_labels
    else:
        labels = [dataset[i][1] for i in range(len(dataset))]
    return [int(label) for label in labels]


def _limit_stratified(indices: list[int], targets: list[int], limit: int | None, seed: int) -> list[int]:
    if limit is None or limit >= len(indices):
        return indices
    subset_targets = [targets[index] for index in indices]
    stratify = subset_targets if limit >= len(set(subset_targets)) else None
    selected, _ = train_test_split(
        indices,
        train_size=limit,
        random_state=seed,
        stratify=stratify,
    )
    return list(selected)


def make_loaders(
    task: TaskName,
    config: TrainConfig,
    root: Path = DATA_DIR,
    download: bool = False,
    val_fraction: float = 0.2,
    limit_train: int | None = None,
    limit_val: int | None = None,
    limit_test: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, TaskSpec]:
    trainval_train = make_dataset(task, "trainval", config.image_size, train=True, root=root, download=download)
    trainval_eval = make_dataset(task, "trainval", config.image_size, train=False, root=root, download=False)
    test_dataset = make_dataset(task, "test", config.image_size, train=False, root=root, download=False)

    indices = list(range(len(trainval_train)))
    targets = _extract_targets(trainval_train, task)
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=config.seed,
        stratify=targets,
    )

    if limit_train is not None:
        train_idx = _limit_stratified(train_idx, targets, limit_train, config.seed)
    if limit_val is not None:
        val_idx = _limit_stratified(val_idx, targets, limit_val, config.seed + 1)
    test_idx = list(range(len(test_dataset)))
    test_targets = _extract_targets(test_dataset, task)
    if limit_test is not None:
        test_idx = _limit_stratified(test_idx, test_targets, limit_test, config.seed + 2)

    train_loader = DataLoader(
        Subset(trainval_train, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        Subset(trainval_eval, val_idx),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_idx),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, test_loader, TASKS[task]
