# Architecture

## Overview

`compactreasoningmodels` is a small PyTorch research package for training
compact neural networks on nonogram puzzles. Data flows through a standard
pipeline:

```
parquet dataset ──> Dataset ──> DataLoader ──> Model ──> Loss ──> Trainer ──> Logger
                                        ^
                                        └──────── Config (Hydra) ─────────────┘
```

All components are instantiated by Hydra from YAML configs; nothing is
hard-coded in `scripts/train.py`.

## Package layout

```
compactreasoningmodels/
├── data_generation/
│   ├── clue_generator.py          # random puzzle grid + run-length clue synthesis
│   ├── constraint_propagator.py   # forward/backward DP for row/col probabilities
│   ├── generate_dataset.py        # multiprocess puzzle generation → parquet writer
│   └── parquet_reader.py          # parquet → in-memory tensors + dataloaders
├── datasets/
│   ├── base.py                    # BaseDataset (Dataset ABC + shared loaders)
│   ├── puzzle.py                  # PuzzleDataset (tensors/paths)
│   └── parquet.py                 # ParquetPuzzleDataset (parquet + metadata)
├── losses/
│   ├── base.py                    # BaseCriterion
│   ├── nonogram.py                # differentiable run-length (clue) loss
│   └── categorical_abstain.py     # AbstainLoss with 3rd "abstain" channel
├── models/
│   ├── base.py                    # BaseModel (nn.Module ABC)
│   ├── layers.py                  # CluePositionalEmbedding, GridResidualBlock
│   ├── mlp.py / tfm.py / cnn.py   # MLP, Transformer, CNN
│   ├── gridmlp.py                 # GridMLP
│   └── recursive_mlp.py / recursive_gridmlp.py
├── trainers/
│   ├── base.py                    # BaseTrainer (train/evaluate loop + early stopping)
│   ├── supervised.py              # NNGSupervisedTrainer (CE / abstain accuracy)
│   └── reward.py                  # NNGRewardTrainer
├── loggers/
│   ├── base.py                    # BaseLogger
│   ├── wandb.py                   # WandbLogger
│   └── null.py                    # NullLogger
└── utils/
    ├── io.py                      # save_model / get_next_model_number
    └── null_target.py             # NullTarget (no-op scheduler)
```

## Design decisions

### Namespace package (`src/` layout)
Code lives under `src/compactreasoningmodels/`, installed editable via
`[tool.pixi.pypi-dependencies]`. The import root is `compactreasoningmodels`,
not `src` — internal imports use that root, and `PYTHONPATH` is set by pixi's
activation so scripts and notebooks resolve the package directly.

### Config-driven instantiation (Hydra)
`scripts/train.py` is a thin `@hydra.main` wrapper. The `defaults` list
composes independent groups; every `_target_` string points at a class in
`compactreasoningmodels`. Overriding on the CLI (e.g. `data=nng5x5_parquet`,
`logger=null_logger`) swaps components without code changes. Experiment
configs (`n5_mlp_s.yaml`, `n5_mlp_r.yaml`, `n5_mlp_abstain.yaml`) live at the
`configs/` root so defaults resolve relative to the config file location.

### Abstract base classes
`BaseDataset`, `BaseModel`, `BaseCriterion`, `BaseTrainer`, and `BaseLogger`
define the interfaces. Concrete implementations only fill in the abstract
methods, so new architectures or losses plug in with a config change.

### Abstain loss
`AbstainLoss` outputs one logit per cell per class plus a final abstain
channel (`output_channels = 3`). The supervised trainer reads the abstain
channel to compute abstain rate and per-label accuracy separately from the
binary accuracy. See the class docstring for the loss math.

### Data versioning
Raw parquet datasets live under `data/raw/` and are documented in
`data/data_info.md`. `data/josebambu_dataset` is an untracked nested
repository and is not part of this package.

## Verification

- `pixi run lint` — ruff (line length 100; E/F/I/W/UP/B rules)
- `pixi run type-check` — mypy over `src` (strict-optional; pandas typed via
  `pandas-stubs`)
- `pixi run test` — pytest suite in `tests/`
- `.github/workflows/ci.yml` runs all three on push/PR