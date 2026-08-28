from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

ClueList = list
GridArray = list


class NonogramParser:
    def __init__(self, strict_meta: bool = False):
        self.strict_meta = strict_meta

    def parse(self, data: Any) -> list[dict]:
        if isinstance(data, (str, Path)):
            return self._parse_file(Path(data))
        if isinstance(data, np.ndarray):
            return self._parse_grid_array(data)
        if isinstance(data, dict):
            return self._parse_dict(data)
        if isinstance(data, (list, tuple)):
            return self._parse_sequence(data)
        raise TypeError(f"Unsupported input type: {type(data)!r}")

    def _parse_file(self, path: Path) -> list[dict]:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with open(path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            return self.parse(records)
        if suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
            return self._parse_grid_array(arr)
        if suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            return self.parse(df.to_dict(orient="records"))
        raise ValueError(f"Unsupported file extension: {suffix!r}")

    # --------------------------------------------------------------------------
    # Data format parsing
    # --------------------------------------------------------------------------

    def _parse_dict(self, data: dict) -> list[dict]:
        if "grid" not in data:
            raise ValueError(f"Unrecognized dict keys: {list(data.keys())!r}")

        # Field-major format: clues and grid are separate batches
        if "clues" in data and self._get_depth(data["clues"]) >= 4:
            return self._parse_field_major(data["clues"], data["grid"])

        # Single example
        return [self._parse_single_example(data)]

    def _parse_sequence(self, data: list | tuple) -> list[dict]:
        """Parse a sequence into examples."""
        if not data:
            return []

        # List of example dicts
        if isinstance(data[0], dict) and "grid" in data[0]:
            return [self._parse_single_example(item) for item in data]

        # 2-element pair format
        if len(data) == 2:
            clue_entry, grid = data
            if self._is_field_major(clue_entry, grid):
                return self._parse_field_major(clue_entry, grid)
            # Single example pair
            return [self._create_example(
                *self._extract_clues(clue_entry),
                self._normalize_grid(grid)
            )]

        raise ValueError("Unrecognized sequence format")

    def _parse_grid_array(self, arr: np.ndarray) -> list[dict]:
        """Parse grid-only numpy array into examples."""
        # Handle flattened grids: (batch, H*W)
        if arr.ndim == 2:
            n_cells = arr.shape[1]
            side = int(round(n_cells ** 0.5))
            if side * side != n_cells:
                raise ValueError(
                    f"Flattened grid with {n_cells} cells isn't square; "
                    "pass a (batch, H, W) array instead."
                )
            arr = arr.reshape(arr.shape[0], side, side)
        elif arr.ndim != 3:
            raise ValueError(
                f"Expected (batch, H, W) or (batch, H*W) array, got ndim={arr.ndim}"
            )

        examples = []
        for grid_2d in arr:
            row_clues = [self._rle_encode(row) for row in grid_2d]
            col_clues = [self._rle_encode(col) for col in grid_2d.T]
            examples.append(
                self._create_example(row_clues, col_clues, grid_2d.reshape(-1).tolist())
            )
        return examples

    def _parse_single_example(self, data: dict) -> dict:
        """Parse a single example from dictionary format."""
        grid = self._normalize_grid(data["grid"])

        if "row_clues" in data and "col_clues" in data:
            row_clues, col_clues = data["row_clues"], data["col_clues"]
        else:
            row_clues, col_clues = self._extract_clues(data["clues"])

        return self._create_example(row_clues, col_clues, grid, data.get("meta"))

    def _parse_field_major(self, clue_batch: Any, grid_batch: Any) -> list[dict]:
        """Parse field-major format where clues and grids are separate batches."""
        if isinstance(grid_batch, np.ndarray):
            grid_batch = grid_batch.reshape(grid_batch.shape[0], -1).tolist()

        if len(clue_batch) != len(grid_batch):
            raise ValueError(
                f"Clue and grid batch length mismatch: {len(clue_batch)} vs {len(grid_batch)}"
            )

        examples = []
        for clue_entry, grid in zip(clue_batch, grid_batch):
            row_clues, col_clues = self._extract_clues(clue_entry)
            examples.append(self._create_example(row_clues, col_clues, grid))
        return examples

    # --------------------------------------------------------------------------
    # Example creation
    # --------------------------------------------------------------------------

    def _create_example(
        self,
        row_clues: ClueList,
        col_clues: ClueList,
        grid: GridArray,
        stated_meta: dict | None = None,
    ) -> dict:
        """Create a normalized example dictionary."""
        # Derive metadata
        grid_arr = np.asarray(grid)
        derived = {
            "shape": (len(row_clues), len(col_clues)),
            "density": float(np.mean(grid_arr != 0)) if grid_arr.size else 0.0,
        }

        # Handle metadata reconciliation
        if stated_meta is not None:
            mismatches = [
                (key, stated_meta[key], value)
                for key, value in derived.items()
                if key in stated_meta and stated_meta[key] != value
            ]
            if mismatches:
                message = "; ".join(
                    f"{key}: stated={stated!r} derived={derived!r}"
                    for key, stated, derived in mismatches
                )
                if self.strict_meta:
                    raise ValueError(f"Meta mismatch: {message}")
                warnings.warn(f"Meta mismatch, using derived values: {message}")
            derived = {**stated_meta, **derived}

        return {
            "X": {"row_clues": row_clues, "col_clues": col_clues},
            "y": grid,
            "meta": derived,
        }

    # --------------------------------------------------------------------------
    # Clue derivation and extraction
    # --------------------------------------------------------------------------

    def _rle_encode(self, line: np.ndarray) -> list[int]:
        """Run-length encode a line of cells."""
        runs = []
        count = 0

        for value in line:
            if value:
                count += 1
            elif count:
                runs.append(count)
                count = 0

        if count:
            runs.append(count)

        return runs or [0]

    def _extract_clues(self, clue_data: Any) -> tuple[ClueList, ClueList]:
        """Extract row and column clues from various formats."""
        if isinstance(clue_data, dict):
            if "row_clues" in clue_data and "col_clues" in clue_data:
                return clue_data["row_clues"], clue_data["col_clues"]
            raise ValueError(
                f"Expected dict with 'row_clues'/'col_clues', got {list(clue_data.keys())!r}"
            )

        if isinstance(clue_data, (list, tuple)):
            if len(clue_data) != 2:
                raise ValueError(f"Expected 2-element pair, got length {len(clue_data)}")
            return clue_data[0], clue_data[1]

        raise TypeError(f"Cannot extract clues from {type(clue_data)!r}")

    # --------------------------------------------------------------------------
    # Utility methods
    # --------------------------------------------------------------------------

    def _get_depth(self, data: Any) -> int:
        """Get nesting depth of data structure."""
        if isinstance(data, np.ndarray):
            return data.ndim

        if isinstance(data, (list, tuple)):
            if not data:
                return 1
            return 1 + self._get_depth(data[0])

        return 0

    def _normalize_grid(self, grid: Any) -> list:
        """Normalize grid to flat list format."""
        if isinstance(grid, np.ndarray):
            return grid.reshape(-1).tolist()
        return grid

    def _is_field_major(self, clue_entry: Any, grid: Any) -> bool:
        """Determine if pair is field-major or single example format."""
        clue_depth = self._get_depth(clue_entry)
        grid_depth = self._get_depth(grid)

        if clue_depth >= 4 and grid_depth <= 2:
            return True
        if clue_depth >= 4 and grid_depth >= 3:
            return True
        if clue_depth == 3 and grid_depth == 1:
            return True

        warnings.warn(
            "Ambiguous pair format; assuming field-major. "
            "Use explicit dict format to avoid ambiguity."
        )
        return True


class NonogramDataset(Dataset):
    """PyTorch Dataset for Nonogram puzzles."""

    def __init__(self, data: Any, strict_meta: bool = False):
        parser = NonogramParser(strict_meta=strict_meta)
        examples = parser.parse(data)

        self.X = [example["X"] for example in examples]
        self.y = [torch.tensor(example["y"], dtype=torch.long) for example in examples]
        self.meta = [example["meta"] for example in examples]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[dict, torch.Tensor, dict]:
        return self.X[idx], self.y[idx], self.meta[idx]

    @staticmethod
    def collate_fn(
        batch: list[tuple[dict, torch.Tensor, dict]]
    ) -> tuple[dict, torch.Tensor, list[dict]]:
        """Collate function for DataLoader."""
        return {
            "row_clues": [x["row_clues"] for x, _, _ in batch],
            "col_clues": [x["col_clues"] for x, _, _ in batch],
        }, torch.stack([y for _, y, _ in batch]), [meta for _, _, meta in batch]
