from __future__ import annotations

import argparse

import torch
from torch import nn

from oxpets.config import TrainConfig, select_device, set_seed
from oxpets.data import TASKS, make_loaders
from oxpets.models import build_scratch_model, build_transfer_model
from oxpets.train import evaluate, train_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quick project checks.")
    parser.add_argument("--download", action="store_true", help="Download Oxford Pets if needed.")
    parser.add_argument("--training-check", action="store_true", help="Run a short training check per task.")
    args = parser.parse_args()

    set_seed(42)
    device = select_device()
    config = TrainConfig(batch_size=4, image_size=96, num_workers=0)
    print(f"Using device: {device}")

    for task, spec in TASKS.items():
        print(f"\nChecking task: {task}")
        train_loader, val_loader, _, _ = make_loaders(
            task,
            config,
            download=args.download,
            limit_train=8,
            limit_val=8,
            limit_test=8,
        )
        images, labels = next(iter(train_loader))
        print(f"batch images={tuple(images.shape)} labels={labels.tolist()} classes={spec.num_classes}")

        for model_name, model in {
            "scratch": build_scratch_model(spec.num_classes),
            "transfer": build_transfer_model(spec.num_classes, pretrained=False),
        }.items():
            model = model.to(device)
            with torch.no_grad():
                logits = model(images.to(device))
            assert logits.shape == (images.size(0), spec.num_classes)
            print(f"{model_name} forward ok: logits={tuple(logits.shape)}")

        if args.training_check:
            model = build_scratch_model(spec.num_classes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            print(
                f"training check ok: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

    print("\nSmoke test completed.")


if __name__ == "__main__":
    main()
