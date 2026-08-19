# Compact Reasoning Models

Research codebase for compact neural models that solve nonogram (grid reasoning)
puzzles. Models take puzzle clues as input and predict the solution grid, with
optional support for an "abstain" output channel so the model can flag cells it
is unsure about.

## Setup

Requires [pixi](https://pixi.sh). The environment is defined entirely in
`pyproject.toml` (dependencies + tasks); there is no separate
`environment.yml`.

```bash
pixi install
```

This installs PyTorch (CUDA), pandas, pyarrow, wandb, Hydra, and dev tools
(ruff, mypy, pytest) into a locked pixi environment.

## Quickstart

Train a small 5x5 model with the abstain loss:

```bash
pixi run train --config-name n5_mlp_abstain data=nng5x5_parquet logger=null_logger
```

Train the plain supervised baseline:

```bash
pixi run train --config-name n5_mlp_s data=nng5x5_parquet logger=null_logger
```

All experiment configs live in `configs/` and are composed from Hydra groups
(`data/`, `model/`, `criterion/`, `trainer/`, `optimizer/`, `scheduler/`,
`dataloader/`, `split/`, `logger/`).

## Project structure

```
src/compactreasoningmodels/   # namespace package (the import root is `compactreasoningmodels`)
├── data_generation/          # puzzle synthesis + constraint propagation + parquet writing
├── datasets/                 # Dataset classes (base, in-memory, parquet)
├── losses/                   # criteria: NonogramLoss, AbstainLoss
├── models/                   # architectures: MLP, Transformer, CNN, GridMLP, RecursiveMLP, RecursiveGridMLP
├── trainers/                 # training loops: supervised, reward
├── loggers/                  # experiment tracking: wandb, null
└── utils/                    # IO helpers, NullTarget scheduler
configs/                      # Hydra experiment + group configs
data/                         # raw parquet datasets (git-ignored content, see data_info.md)
scripts/                      # entry points: train.py, inspect_data.py
tests/                        # pytest suite
notebooks/                    # exploratory notebooks
```

## Development tasks

```bash
pixi run lint        # ruff check
pixi run format      # ruff format
pixi run type-check  # mypy src
pixi run test        # pytest tests
```

## Models

All models subclass `BaseModel` and take `(batch, features)` inputs (flat clue
encoding); `require_flat_input = True` marks architectures that expect the flat
representation.

- `MultiLayerPerceptron` / `RecursiveMLP` — fully-connected baselines
- `Transformer` — standard transformer over clue tokens
- `GridMLP` / `RecursiveGridMLP` — grid-structured residual blocks, the models
  used for the main experiments
- `CNN` — 1-D/2-D convolutional variant

## Losses

- `NonogramLoss` — differentiable run-length (clue) loss that compares the
  predicted grid's run-length encoding against the true clues.
- `AbstainLoss` — 3-channel per-cell cross-entropy where the last channel is an
  abstain class (`output_channels = 3`). Includes an entropy bonus to prevent
  collapse and an abstain penalty calibrated against the random-guess baseline.

## Tests

The suite covers the abstain loss (loss values, masks, validation), the
nonogram loss (perfect-prediction optimum, reductions), parquet dataset
loading/shapes, and a supervised-trainer smoke test.

```bash
pixi run test
```

## CI

`.github/workflows/ci.yml` runs `pixi run lint`, `pixi run type-check`, and
`pixi run test` on push/PR, and validates notebooks are parseable.