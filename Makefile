PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTHONPATH := src

.PHONY: setup data smoke train train-laptop-final evaluate visuals report slides clean

setup:
	/opt/homebrew/bin/python3.11 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -m ipykernel install --user --name oxford-pets-project --display-name "Oxford Pets Project"

data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/prepare_data.py --download

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/smoke_test.py --download --tiny-overfit

train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_experiment.py --task breed --model scratch
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_experiment.py --task breed --model transfer
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_experiment.py --task species --model scratch
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_experiment.py --task species --model transfer

train-laptop-final:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_experiment.py --task breed --model scratch --epochs 3 --limit-train 370 --limit-val 111 --limit-test 370 --batch-size 32 --image-size 96 --device cpu
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_experiment.py --task breed --model transfer --epochs 3 --fine-tune-epochs 1 --limit-train 370 --limit-val 111 --limit-test 370 --batch-size 32 --image-size 96 --device cpu
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_experiment.py --task species --model scratch --epochs 3 --limit-train 400 --limit-val 120 --limit-test 400 --batch-size 32 --image-size 96 --device cpu
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_experiment.py --task species --model transfer --epochs 3 --fine-tune-epochs 1 --limit-train 400 --limit-val 120 --limit-test 400 --batch-size 32 --image-size 96 --device cpu

evaluate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/evaluate_results.py

visuals:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/make_report_figures.py

report:
	cd reports && latexmk -pdf Project_Report.tex

slides:
	cd slides && latexmk -pdf Project_Presentation.tex

clean:
	rm -rf outputs/metrics/* outputs/figures/* outputs/checkpoints/*
