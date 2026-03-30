## Well Formatted Repo

### Structure & Standards
- [ ] Clean: modular code, clear comments, type hints throughout
- [ ] Traversable: obvious folder structure, README per directory
- [ ] Reproducible: seeded randomness, locked dependencies (pixi.lock), git tags for experiments
- [ ] Manageable: experiment tracking, easy comparison of runs
- [ ] Clear Setup: single-command install, working examples

### Core Infrastructure
- [ ] **Pixi**: dependency management and task running
- [ ] **Hydra**: hierarchical configs, sweeps, experiment output directories
- [ ] **Pre-commit**: ruff, mypy, nbstripout, large-file checks
- [ ] **GitHub Actions**: CI (lint/test), CD (smoke training on merge)

### Abstract Base Classes
- [ ] `BaseDataset`: data loading interface
- [ ] `BaseModel`: model architecture interface
- [ ] `BaseTrainer`: training loop interface
- [ ] `BaseMetric`: evaluation interface
- [ ] `BaseLogger`: experiment tracking interface

### Experiment Tracking (W&B)
- [ ] Auto-log config, code, system info
- [ ] Log metrics per step/epoch
- [ ] Log model architecture and gradients
- [ ] Artifact logging (checkpoints, best model)
- [ ] Prediction tables for debugging
- [ ] Alert on crash or NaN

### Data Versioning (DVC)
- [ ] Initialize DVC with cloud remote
- [ ] Define pipeline stages (download → validate → clean → split → tokenize)
- [ ] Git-track .dvc files, not data
- [ ] Reproducibility: `dvc repro` runs only changed stages

### Code Quality
- [ ] **Ruff**: linting and formatting (line length 100)
- [ ] **MyPy**: strict type checking
- [ ] **Pre-commit**: auto-run on every commit
- [ ] **Nbstripout**: clean notebook outputs before git

### Testing
- [ ] **Unit tests**: data shapes, model forward/backward, metric correctness
- [ ] **Integration tests**: full pipeline on tiny data
- [ ] **Regression tests**: known failure cases
- [ ] **CI**: run fast tests on PR, slow tests on schedule

### Documentation
- [ ] Root README: install, quickstart, project overview
- [ ] `docs/architecture.md`: system design with diagrams
- [ ] `docs/experiments/`: dated experiment logs
- [ ] Docstrings: Google or NumPy style, generate API docs

### Safety & Validation
- [ ] Data schemas (validate on load)
- [ ] Input distribution checks (detect drift)
- [ ] Model cards (performance, limitations)
- [ ] Deterministic splits (seeded random)