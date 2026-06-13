from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figsize = (12, 10) if len(class_names) > 10 else (6, 5)
    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        square=False,
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_history_plot(history: list[dict], output_path: Path, title: str) -> None:
    frame = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    if "train_loss" in frame:
        plt.plot(frame["epoch"], frame["train_loss"], marker="o", linewidth=2, label="train loss")
    if "val_loss" in frame:
        plt.plot(frame["epoch"], frame["val_loss"], marker="s", linewidth=2, label="validation loss")
    if "val_accuracy" in frame:
        plt.plot(frame["epoch"], frame["val_accuracy"], marker="^", linewidth=2, label="validation accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.xticks(frame["epoch"].astype(int).tolist())
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_summary_barplot(summary_csv: Path, output_path: Path) -> None:
    frame = pd.read_csv(summary_csv)
    if frame.empty:
        return
    labels = frame["task"] + " / " + frame["model"]
    x = np.arange(len(frame))
    width = 0.35
    plt.figure(figsize=(9, 5))
    bars_accuracy = plt.bar(x - width / 2, frame["test_accuracy"], width, label="accuracy")
    bars_f1 = plt.bar(x + width / 2, frame["test_macro_f1"], width, label="macro F1")
    for bars in (bars_accuracy, bars_f1):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                max(height, 0.02),
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Oxford Pets model comparison")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
