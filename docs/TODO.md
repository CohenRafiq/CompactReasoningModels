# Repository Standards Checklist

## Structure & Standards
- [x] Clean: modular code, clear comments, type hints throughout
- [x] Traversable: obvious folder structure (`src/compactreasoningmodels/`), root README
- [x] Reproducible: seeded randomness, locked dependencies (pixi.lock)
- [x] Manageable: experiment tracking (wandb logger), easy comparison of runs
- [x] Clear Setup: single-command install (`pixi install`), working examples

## Core Infrastructure
- [x] **Pixi**: dependency management and task running
- [x] **Hydra**: hierarchical configs, sweeps, experiment output directories
- [ ] **Pre-commit**: ruff, mypy, nbstripout, large-file checks
- [x] **GitHub Actions**: CI (lint/type-check/test on push/PR)

## Abstract Base Classes
- [x] `NonogramDataset`: data loading interface (`datasets/nonogram_dataset.py`)
- [x] `BaseModel`: model architecture interface (`models/base.py`)
- [x] `BaseTrainer`: training loop interface (`trainers/base.py`)
- [x] `BaseCriterion`: loss interface (`losses/base.py`)
- [x] `BaseLogger`: experiment tracking interface (`loggers/base.py`)

## Experiment Tracking (W&B)
- [x] Auto-log config, code, system info
- [x] Log metrics per step/epoch
- [x] Log model architecture and gradients (`watch_model`)
- [ ] Artifact logging (checkpoints, best model)
- [ ] Prediction tables for debugging
- [ ] Alert on crash or NaN

## Data Versioning (DVC)
- [ ] Initialize DVC with cloud remote
- [ ] Define pipeline stages (download → validate → clean → split → tokenize)
- [ ] Git-track .dvc files, not data
- [ ] Reproducibility: `dvc repro` runs only changed stages

## Code Quality
- [x] **Ruff**: linting and formatting (line length 100)
- [x] **MyPy**: type checking over `src`
- [ ] **Pre-commit**: auto-run on every commit
- [ ] **Nbstripout**: clean notebook outputs before git

## Documentation
- [x] Root README: install, quickstart, project overview
- [x] `docs/architecture.md`: system design
- [ ] `docs/experiments/`: dated experiment logs

## Safety & Validation
- [ ] Data schemas (validate on load)
- [ ] Input distribution checks (detect drift)
- [ ] Model cards (performance, limitations)
- [x] Deterministic splits (seeded random)
