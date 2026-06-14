# Oxford Pets Deep Learning Project

This project is built for the **Deep Learning and Big Data** project assignment. It uses the Oxford-IIIT Pet dataset to compare deep learning models on two tasks from the same dataset:

1. **Breed classification**: 37-class image classification.
2. **Species classification**: binary cat-vs-dog classification.

The project compares:

- **Model 1, from scratch**: a custom CNN with convolution, batch normalization, ReLU, max pooling, dropout, and fully connected layers.
- **Model 2, alternative approach**: transfer learning with a pretrained ResNet18 backbone.

## Quick Start, Simple Student Version

The easiest way to run the project is the notebook:

1. Open `notebooks/01_project_walkthrough.ipynb` in VS Code or Jupyter.
2. Select the kernel `Oxford Pets Project`.
3. Run all cells from top to bottom.

The notebook loads the dataset, shows real Oxford Pets images, builds both models, runs a tiny training demo, and displays the final figures used in the report and presentation.

The required exam documents are already here:

- `reports/Project_Report.pdf`
- `slides/Project_Presentation.pdf`

## Optional Command Line Version

Use this only if you want to regenerate everything from the terminal.

Run these commands from this folder on macOS or Linux:

```bash
make setup
make data
make visuals
make smoke
```

Then train all required experiments:

```bash
make train-laptop-final
make evaluate
make visuals
```

`make train-laptop-final` runs all four required combinations with stratified laptop-sized subsets. Use `make train` for the longer full-default training run.

Build the written deliverables:

```bash
make report
make slides
```

On Windows, use the PowerShell commands in the Windows section below.

## Run It Live

If the environment is already created, activate it first:

macOS or Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

For a quick live check that loads the dataset, builds both models, runs forward passes, and trains tiny batches:

macOS or Linux:

```bash
make smoke
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts\smoke_test.py --download --tiny-overfit
```

For a short training demo that finishes much faster than the full project:

macOS or Linux:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_experiment.py --task species --model scratch --epochs 1 --limit-train 128 --limit-val 64 --limit-test 64 --image-size 96 --device cpu
PYTHONPATH=src .venv/bin/python scripts/evaluate_results.py
PYTHONPATH=src .venv/bin/python scripts/make_report_figures.py
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts\train_experiment.py --task species --model scratch --epochs 1 --limit-train 128 --limit-val 64 --limit-test 64 --image-size 96 --device cpu
python scripts\evaluate_results.py
python scripts\make_report_figures.py
```

For the submitted experiment results, run:

macOS or Linux:

```bash
make train-laptop-final
make evaluate
make visuals
make report
make slides
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts\train_experiment.py --task breed --model scratch --epochs 3 --limit-train 370 --limit-val 111 --limit-test 370 --batch-size 32 --image-size 96 --device cpu
python scripts\train_experiment.py --task breed --model transfer --epochs 3 --fine-tune-epochs 1 --limit-train 370 --limit-val 111 --limit-test 370 --batch-size 32 --image-size 96 --device cpu
python scripts\train_experiment.py --task species --model scratch --epochs 3 --limit-train 400 --limit-val 120 --limit-test 400 --batch-size 32 --image-size 96 --device cpu
python scripts\train_experiment.py --task species --model transfer --epochs 3 --fine-tune-epochs 1 --limit-train 400 --limit-val 120 --limit-test 400 --batch-size 32 --image-size 96 --device cpu
python scripts\evaluate_results.py
python scripts\make_report_figures.py
```

The full submitted run trains four model and task combinations, so it takes longer than the demo.

## Manual Environment Setup

Use Python 3.10 or newer. Python 3.11 is recommended.

### macOS

If `make setup` is not available on macOS, run:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name oxford-pets-project --display-name "Oxford Pets Project"
```

If `/opt/homebrew/bin/python3.11` does not exist, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name oxford-pets-project --display-name "Oxford Pets Project"
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name oxford-pets-project --display-name "Oxford Pets Project"
```

If `venv` is missing on Ubuntu or Debian, install it first:

```bash
sudo apt update
sudo apt install python3-venv
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name oxford-pets-project --display-name "Oxford Pets Project"
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `py -3.11` is not available, install Python 3.11 from <https://www.python.org/downloads/> and make sure it is added to PATH.

The official PyTorch instructions recommend Python 3.10 or newer.

## Main Files

- `src/oxpets/data.py`: Oxford Pets download, transforms, train/validation/test loaders.
- `src/oxpets/models.py`: scratch CNN and transfer learning model.
- `src/oxpets/train.py`: training and evaluation loops.
- `src/oxpets/metrics.py`: accuracy, macro F1, confusion matrices, and plots.
- `scripts/train_experiment.py`: train one model/task pair.
- `scripts/make_report_figures.py`: generate dataset, augmentation, and prediction figures for the report and slides.
- `notebooks/01_project_walkthrough.ipynb`: guided notebook for the project story.
- `reports/Project_Report.tex`: 6-10 page report source.
- `slides/Project_Presentation.tex`: presentation source.

## Dataset

Official dataset page: <https://www.robots.ox.ac.uk/~vgg/data/pets/>

Torchvision dataset documentation: <https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html>

Oxford Pets contains 37 pet categories with roughly 200 images per class. The dataset also includes category labels, binary cat/dog labels, and segmentation trimaps. This project uses category and binary labels because they directly match the assignment option “multi-class classification + binary classification.”

## Training Notes

Defaults are laptop-friendly:

- image size: `160x160`
- batch size: `32`
- scratch CNN: `8` epochs
- transfer model: `5` frozen-head epochs plus `3` final-block fine-tuning epochs
- device: Apple Silicon MPS if available, then CUDA, otherwise CPU

For a quick demo run:

macOS or Linux:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_experiment.py --task species --model scratch --epochs 1 --limit-train 128 --limit-val 64 --limit-test 64
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python scripts\train_experiment.py --task species --model scratch --epochs 1 --limit-train 128 --limit-val 64 --limit-test 64
```

## Course Material Connection

The project follows the lecture material and notebooks on tensors, PyTorch datasets, layers, activation functions, convolution, loss functions, backpropagation, optimizers, dropout, validation, and model comparison. The report and slides include a short mapping from these course topics to the implementation.

## Outputs

Result files go into:

- `outputs/metrics/`: histories, JSON metrics, comparison CSV.
- `outputs/figures/`: loss curves, confusion matrices, summary plots.
- `outputs/checkpoints/`: saved model checkpoints.

These generated files are ignored by git so the repository stays lightweight.

## Academic Integrity

The implementation is original project code written for this assignment. External sources used for facts or APIs must be cited in the report and slides, especially the Oxford dataset page, Torchvision documentation, PyTorch documentation, and the ResNet paper/model documentation where relevant.
