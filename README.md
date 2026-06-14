# Oxford Pets Deep Learning Project

## Project Overview

This project was created for the **Deep Learning and Big Data** final project.

We use the Oxford-IIIT Pet dataset to classify pet images with PyTorch. The project compares two model types:

- a CNN trained from scratch
- a ResNet18 transfer learning model

The project includes two classification tasks:

- breed classification with 37 classes
- species classification with 2 classes, cat and dog

## Dataset Description

- Source: Oxford-IIIT Pet Dataset
- Dataset link: <https://www.robots.ox.ac.uk/~vgg/data/pets/>
- Number of classes: 37 pet breeds
- Approximate size: about 200 images per class
- Labels used in this project:
  - breed label for multi-class classification
  - cat/dog label for binary classification

## Project Structure

```text
oxford_pets_project/
|-- data/                     # Downloaded dataset, ignored by Git
|-- docs/                     # Team notes and contribution plan
|-- notebooks/                # Main project notebook
|-- outputs/                  # Generated figures, metrics, and checkpoints
|-- reports/                  # Final report source and PDF
|-- scripts/                  # Python scripts used for experiments
|-- slides/                   # Final presentation source and PDF
|-- src/oxpets/               # Project code for data, models, training, metrics
|-- main.py                   # Runs the project experiments
|-- streamlit_app.py          # Live presentation demo
|-- requirements.txt          # Python packages
`-- README.md
```

## Requirements

Python 3.10 or newer is required. Python 3.11 is recommended.

Required Python packages are listed in `requirements.txt`.

Main packages:

- torch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- pillow
- jupyter
- streamlit

## Setup

Create a virtual environment and install the packages.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name oxford-pets-project --display-name "Oxford Pets Project"
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name oxford-pets-project --display-name "Oxford Pets Project"
```

If PowerShell blocks activation, run this once in the same PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## How to Run

The easiest way to understand the project is the notebook:

1. Open `notebooks/01_project_walkthrough.ipynb`.
2. Select the kernel `Oxford Pets Project`.
3. Run all cells from top to bottom.

To run the full project experiments from the terminal:

```bash
python main.py
```

For a shorter check before running the full experiments:

```bash
python main.py --quick
```

The full run downloads the dataset if needed, trains the models, evaluates the results, and creates the figures. The quick run only checks that training works.

## Live Presentation Demo

The project also includes a Streamlit app for the presentation.

```bash
streamlit run streamlit_app.py
```

The app shows dataset images, saved result figures, and a live training section where the CNN trains epoch by epoch.

## Final Documents

The final exam documents are included here:

- `reports/Project_Report.pdf`
- `slides/Project_Presentation.pdf`

## Outputs

Generated files are saved in the `outputs/` folder:

- `outputs/metrics/`: accuracy, macro F1, and training history files
- `outputs/figures/`: dataset images, training curves, confusion matrices, and result plots
- `outputs/checkpoints/`: trained model files

Downloaded data, model checkpoints, and generated outputs are ignored by Git.

## Team Information

Team members:

- Love
- Jamal Dassrath
- Erbakan Ahmad
- Muhammad Ahtisham Bhatti

## Academic Use

This project was developed for a university deep learning assignment. The code is written for learning purposes, and the report and slides cite the dataset and library sources used in the project.
