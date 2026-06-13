from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
from torch import nn

from oxpets.config import CHECKPOINT_DIR, FIGURES_DIR, METRICS_DIR, TrainConfig, ensure_output_dirs, select_device, set_seed
from oxpets.data import TaskName, make_loaders
from oxpets.metrics import compute_metrics, save_confusion_matrix, save_history_plot
from oxpets.models import build_scratch_model, build_transfer_model, unfreeze_final_resnet_block
from oxpets.train import evaluate, train_model, trainable_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Oxford Pets experiment.")
    parser.add_argument("--task", choices=["breed", "species"], required=True)
    parser.add_argument("--model", choices=["scratch", "transfer"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fine-tune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    set_seed(42)
    device = select_device(None if args.device == "auto" else args.device)
    config = TrainConfig(batch_size=args.batch_size, image_size=args.image_size)
    train_loader, val_loader, test_loader, spec = make_loaders(
        args.task,
        config,
        download=args.download,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
    )

    if args.model == "scratch":
        model = build_scratch_model(spec.num_classes)
        epochs = args.epochs or config.scratch_epochs
        fine_tune_epochs = 0
    else:
        model = build_transfer_model(
            spec.num_classes,
            freeze_backbone=True,
            pretrained=not args.no_pretrained,
        )
        epochs = args.epochs or config.transfer_epochs
        fine_tune_epochs = config.fine_tune_epochs if args.fine_tune_epochs is None else args.fine_tune_epochs

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Training {args.model} model for {args.task} on {device}")

    history = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)

    if args.model == "transfer" and fine_tune_epochs > 0:
        print("Fine-tuning final ResNet block")
        unfreeze_final_resnet_block(model)
        optimizer = torch.optim.AdamW(trainable_parameters(model), lr=args.lr * 0.1, weight_decay=args.weight_decay)
        history.extend(
            train_model(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                epochs=fine_tune_epochs,
                start_epoch=len(history) + 1,
            )
        )

    test_loss, test_accuracy, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(y_true, y_pred, spec.class_names)
    metrics.update({
        "task": args.task,
        "model": args.model,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_macro_f1": float(metrics["macro_f1"]),
        "num_test_samples": len(y_true),
    })

    run_name = f"{args.task}_{args.model}"
    pd.DataFrame(history).to_csv(METRICS_DIR / f"{run_name}_history.csv", index=False)
    with (METRICS_DIR / f"{run_name}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task": args.task,
            "model": args.model,
            "class_names": spec.class_names,
            "metrics": metrics,
        },
        CHECKPOINT_DIR / f"{run_name}.pt",
    )
    save_history_plot(history, FIGURES_DIR / f"{run_name}_history.png", f"{args.task} {args.model} training")
    save_confusion_matrix(
        y_true,
        y_pred,
        spec.class_names,
        FIGURES_DIR / f"{run_name}_confusion_matrix.png",
        f"{args.task} {args.model} confusion matrix",
    )
    print(f"Test accuracy={test_accuracy:.3f}, macro_f1={metrics['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
