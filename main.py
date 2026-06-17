from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    command = [sys.executable, *args]
    print("\nRunning:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Oxford Pets project.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a shorter check instead of all final experiments.",
    )
    args = parser.parse_args()

    run_step(["scripts/prepare_data.py", "--download"])

    if args.quick:
        experiments = [
            [
                "scripts/train_experiment.py",
                "--task", "species",
                "--model", "scratch",
                "--epochs", "2",
                "--limit-train", "256",
                "--limit-val", "64",
                "--limit-test", "128",
                "--batch-size", "16",
                "--image-size", "128",
                "--run-name", "quick_species_scratch",
            ],
        ]
    else:
        experiments = [
            [
                "scripts/train_experiment.py",
                "--task", "breed",
                "--model", "scratch",
                "--epochs", "8",
                "--batch-size", "32",
                "--image-size", "160",
            ],
            [
                "scripts/train_experiment.py",
                "--task", "breed",
                "--model", "transfer",
                "--epochs", "5",
                "--fine-tune-epochs", "3",
                "--batch-size", "32",
                "--image-size", "160",
            ],
            [
                "scripts/train_experiment.py",
                "--task", "species",
                "--model", "scratch",
                "--epochs", "8",
                "--batch-size", "32",
                "--image-size", "160",
            ],
            [
                "scripts/train_experiment.py",
                "--task", "species",
                "--model", "transfer",
                "--epochs", "5",
                "--fine-tune-epochs", "3",
                "--batch-size", "32",
                "--image-size", "160",
            ],
        ]

    for experiment in experiments:
        run_step(experiment)

    if args.quick:
        print("\nQuick check finished. Run python main.py for the final four experiments.")
        return

    run_step(["scripts/evaluate_results.py"])
    run_step(["scripts/make_report_figures.py"])

    print("\nDone. Results are saved in the outputs folder.")


if __name__ == "__main__":
    main()
