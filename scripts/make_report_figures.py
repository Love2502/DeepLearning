from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from oxpets.config import CHECKPOINT_DIR, DATA_DIR, FIGURES_DIR, TrainConfig, ensure_output_dirs
from oxpets.data import BREED_CLASSES, SPECIES_CLASSES, make_loaders
from oxpets.models import build_transfer_model


def _load_raw_dataset():
    return datasets.OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=None,
        download=False,
    )


def save_sample_grid() -> None:
    dataset = _load_raw_dataset()
    selected_indices = np.linspace(0, len(dataset) - 1, 12, dtype=int)

    fig, axes = plt.subplots(3, 4, figsize=(11, 8))
    for ax, idx in zip(axes.flat, selected_indices):
        image, label = dataset[int(idx)]
        ax.imshow(image)
        ax.set_title(BREED_CLASSES[int(label)], fontsize=9)
        ax.axis("off")
    fig.suptitle("Oxford Pets trainval examples", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_sample_grid.png", dpi=200)
    plt.close(fig)


def save_distribution_plots() -> None:
    dataset = _load_raw_dataset()
    breed_counts = Counter(int(label) for label in dataset._labels)
    species_counts = Counter(int(label) for label in dataset._bin_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    breed_x = np.arange(len(BREED_CLASSES))
    axes[0].bar(breed_x, [breed_counts[i] for i in breed_x], color="#3b82f6")
    axes[0].set_title("Breed-label distribution")
    axes[0].set_ylabel("Images")
    axes[0].set_xticks(breed_x)
    axes[0].set_xticklabels(BREED_CLASSES, rotation=90, fontsize=7)
    axes[0].grid(axis="y", alpha=0.25)

    species_x = np.arange(len(SPECIES_CLASSES))
    bars = axes[1].bar(
        species_x,
        [species_counts[i] for i in species_x],
        color=["#14b8a6", "#f97316"],
    )
    axes[1].set_title("Species-label distribution")
    axes[1].set_ylabel("Images")
    axes[1].set_xticks(species_x)
    axes[1].set_xticklabels(SPECIES_CLASSES)
    axes[1].grid(axis="y", alpha=0.25)
    for bar in bars:
        height = int(bar.get_height())
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 15,
            str(height),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_label_distribution.png", dpi=200)
    plt.close(fig)


def save_augmentation_preview() -> None:
    dataset = datasets.OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=None,
        download=False,
    )
    image, label = dataset[0]
    preview_transform = transforms.Compose([
        transforms.RandomResizedCrop(160, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for axis_id, ax in enumerate(axes[1:], start=1):
        augmented = preview_transform(image)
        ax.imshow(augmented)
        ax.set_title(f"Augment {axis_id}")
        ax.axis("off")
    fig.suptitle(f"Training augmentation preview: {BREED_CLASSES[int(label)]}", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "augmentation_preview.png", dpi=200)
    plt.close(fig)


def _unnormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor.cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def _save_prediction_grid(
    selected: list[tuple[torch.Tensor, int, int, float]],
    output_path,
    rows: int,
    cols: int,
    figsize: tuple[float, float],
    title: str,
) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_list = list(np.atleast_1d(axes).flat)
    for ax, (image, label, prediction, score) in zip(axes_list, selected):
        ax.imshow(_unnormalize(image))
        is_correct = label == prediction
        color = "#166534" if is_correct else "#b91c1c"
        ax.set_title(
            f"true: {SPECIES_CLASSES[label]}\npred: {SPECIES_CLASSES[prediction]} ({score:.2f})",
            fontsize=12,
            color=color,
        )
        ax.axis("off")
    for ax in axes_list[len(selected):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=17, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.8, w_pad=1.0)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_species_prediction_grid() -> None:
    checkpoint_path = CHECKPOINT_DIR / "species_transfer.pt"
    if not checkpoint_path.exists():
        print(f"Skipping prediction grid because {checkpoint_path} does not exist yet.")
        return

    config = TrainConfig(batch_size=32, image_size=96, num_workers=0)
    _, _, test_loader, spec = make_loaders(
        "species",
        config,
        download=False,
        limit_train=400,
        limit_val=120,
        limit_test=400,
    )
    model = build_transfer_model(spec.num_classes, freeze_backbone=False, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    examples: list[tuple[torch.Tensor, int, int, float]] = []
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
            confidence = probabilities.max(dim=1).values
            for image, label, prediction, score in zip(images, labels, predictions, confidence):
                examples.append((image, int(label), int(prediction), float(score)))

    correct = [row for row in examples if row[1] == row[2]]
    mistakes = [row for row in examples if row[1] != row[2]]
    selected = correct[:8] + mistakes[:4]
    if len(selected) < 12:
        selected = examples[:12]

    _save_prediction_grid(
        selected,
        FIGURES_DIR / "species_transfer_prediction_grid.png",
        rows=3,
        cols=4,
        figsize=(12, 9),
        title="Species transfer model predictions on test images",
    )
    slide_selected = correct[:5] + mistakes[:3]
    if len(slide_selected) < 8:
        slide_selected = examples[:8]
    _save_prediction_grid(
        slide_selected,
        FIGURES_DIR / "species_transfer_prediction_slide.png",
        rows=2,
        cols=4,
        figsize=(12, 6.6),
        title="Prediction examples: correct cases and mistakes",
    )


def main() -> None:
    ensure_output_dirs()
    save_sample_grid()
    save_distribution_plots()
    save_augmentation_preview()
    save_species_prediction_grid()
    print("Saved report figures to", FIGURES_DIR)


if __name__ == "__main__":
    main()
