from __future__ import annotations

import json

import pandas as pd

from oxpets.config import FIGURES_DIR, METRICS_DIR, ensure_output_dirs
from oxpets.metrics import save_summary_barplot


def main() -> None:
    ensure_output_dirs()
    rows = []
    for path in sorted(METRICS_DIR.glob("*_metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append({
            "task": metrics["task"],
            "model": metrics["model"],
            "test_loss": metrics["test_loss"],
            "test_accuracy": metrics["test_accuracy"],
            "test_macro_f1": metrics["test_macro_f1"],
            "num_test_samples": metrics["num_test_samples"],
        })
    summary = pd.DataFrame(rows)
    output_csv = METRICS_DIR / "model_comparison_summary.csv"
    summary.to_csv(output_csv, index=False)
    if summary.empty:
        print("No metrics found yet. Run make train or scripts/train_experiment.py first.")
        return
    save_summary_barplot(output_csv, FIGURES_DIR / "model_comparison_summary.png")
    print(summary.sort_values(["task", "model"]).to_string(index=False))


if __name__ == "__main__":
    main()
