from __future__ import annotations

import argparse

from oxpets.config import DATA_DIR, TrainConfig
from oxpets.data import TASKS, make_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect Oxford-IIIT Pet dataset.")
    parser.add_argument("--download", action="store_true", help="Download dataset if it is missing.")
    args = parser.parse_args()

    config = TrainConfig()
    print(f"Dataset root: {DATA_DIR}")
    for task in TASKS:
        trainval = make_dataset(task, "trainval", config.image_size, train=False, download=args.download)
        test = make_dataset(task, "test", config.image_size, train=False, download=False)
        sample, label = trainval[0]
        print(
            f"{task}: trainval={len(trainval)} test={len(test)} "
            f"sample_shape={tuple(sample.shape)} first_label={label}"
        )


if __name__ == "__main__":
    main()
