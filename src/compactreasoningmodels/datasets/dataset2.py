from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ClueList = list
GridArray = list


class NonogramParser:
    """Parser for converting various input formats into Nonogram examples."""

    def __init__(self, strict_meta: bool = False):
        self.strict_meta = strict_meta
        self._warnings: list[str] = []

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def parse(self, data: Any) -> list[dict]:
        """Parse any supported input format into a list of examples."""
        if isinstance(data, (str, Path)):
            return self._parse_file(Path(data))
        if isinstance(data, np.ndarray):
            return self._parse_grid_array(data)
        if isinstance(data, dict):
            return self._parse_dict(data)
        if isinstance(data, (list, tuple)):
            return self._parse_sequence(data)
        raise TypeError(f"Unsupported input type: {type(data)!r}")

    # --------------------------------------------------------------------------
    # File parsing
    # --------------------------------------------------------------------------

    def _parse_file(self, path: Path) -> list[dict]:
        """Parse data from a file based on its extension."""
        suffix = path.suffix.lower()

        if suffix == ".jsonl":
            return self._parse_jsonl(path)
        if suffix == ".npy":
            return self._parse_npy(path)
        if suffix == ".parquet":
            return self._parse_parquet(path)

        raise ValueError(f"Unsupported file extension: {suffix!r}")

    def _parse_jsonl(self, path: Path) -> list[dict]:
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        return self.parse(records)

    def _parse_npy(self, path: Path) -> list[dict]:
        arr = np.load(path, allow_pickle=False)
        return self._parse_grid_array(arr)

    def _parse_parquet(self, path: Path) -> list[dict]:
        df = pd.read_parquet(path)
        records = df.to_dict(orient="records")
        return self.parse(records)

    # --------------------------------------------------------------------------
    # Data format parsing
    # --------------------------------------------------------------------------

    def _parse_dict(self, data: dict) -> list[dict]:
        """Parse a dictionary into examples."""
        self._validate_dict_keys(data)

        if self._has_explicit_clues(data):
            return [self._parse_single_example(data)]

        if self._has_nested_clues(data):
            return self._parse_field_major(data["clues"], data["grid"])

        return [self._parse_single_example(data)]

    def _parse_sequence(self, data: list | tuple) -> list[dict]:
        """Parse a sequence into examples."""
        if not data:
            return []

        if self._is_list_of_dicts(data):
            return [self._parse_single_example(item) for item in data]

        if self._is_pair_format(data):
            return self._parse_pair(data)

        raise ValueError("Unrecognized sequence format")

    def _parse_grid_array(self, arr: np.ndarray) -> list[dict]:
        """Parse grid-only numpy array into examples."""
        arr = self._normalize_grid_array(arr)

        examples = []
        for grid_2d in arr:
            row_clues, col_clues = self._derive_clues_from_grid(grid_2d)
            examples.append(
                self._create_example(
                    row_clues,
                    col_clues,
                    grid_2d.reshape(-1).tolist()
                )
            )
        return examples

    def _parse_single_example(self, data: dict) -> dict:
        """Parse a single example from dictionary format."""
        grid = self._normalize_grid(data["grid"])

        if self._has_explicit_clues(data):
            row_clues, col_clues = data["row_clues"], data["col_clues"]
        else:
            row_clues, col_clues = self._extract_clues(data["clues"])

        return self._create_example(
            row_clues,
            col_clues,
            grid,
            stated_meta=data.get("meta")
        )

    def _parse_field_major(self, clue_batch: Any, grid_batch: Any) -> list[dict]:
        """Parse field-major format where clues and grids are separate batches."""
        grid_batch = self._normalize_grid_batch(grid_batch)

        if len(clue_batch) != len(grid_batch):
            raise ValueError(
                f"Clue and grid batch length mismatch: {len(clue_batch)} vs {len(grid_batch)}"
            )

        examples = []
        for clue_entry, grid in zip(clue_batch, grid_batch):
            row_clues, col_clues = self._extract_clues(clue_entry)
            examples.append(self._create_example(row_clues, col_clues, grid))
        return examples

    def _parse_pair(self, data: tuple) -> list[dict]:
        """Parse a 2-element pair format."""
        clue_entry, grid = data

        if self._is_field_major_format(clue_entry, grid):
            return self._parse_field_major(clue_entry, grid)

        # Single example pair
        grid = self._normalize_grid(grid)
        row_clues, col_clues = self._extract_clues(clue_entry)
        return [self._create_example(row_clues, col_clues, grid)]

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
        meta = self._reconcile_meta(stated_meta, row_clues, col_clues, grid)

        return {
            "X": {"row_clues": row_clues, "col_clues": col_clues},
            "y": grid,
            "meta": meta,
        }

    def _reconcile_meta(
        self,
        stated_meta: dict | None,
        row_clues: ClueList,
        col_clues: ClueList,
        grid: GridArray,
    ) -> dict:
        """Merge stated metadata with derived metadata."""
        derived = self._derive_meta(row_clues, col_clues, grid)

        if stated_meta is None:
            return derived

        self._check_meta_mismatches(stated_meta, derived)
        return {**stated_meta, **derived}

    def _derive_meta(
        self, row_clues: ClueList, col_clues: ClueList, grid: GridArray
    ) -> dict:
        """Derive metadata from the example data."""
        grid_arr = np.asarray(grid)
        density = float(np.mean(grid_arr != 0)) if grid_arr.size else 0.0

        return {
            "shape": (len(row_clues), len(col_clues)),
            "density": density,
        }

    def _check_meta_mismatches(self, stated: dict, derived: dict) -> None:
        """Check for and handle metadata mismatches."""
        mismatches = [
            (key, stated[key], value)
            for key, value in derived.items()
            if key in stated and stated[key] != value
        ]

        if not mismatches:
            return

        message = "; ".join(
            f"{key}: stated={stated_value!r} derived={derived_value!r}"
            for key, stated_value, derived_value in mismatches
        )

        if self.strict_meta:
            raise ValueError(f"Meta mismatch: {message}")

        warnings.warn(f"Meta mismatch, using derived values: {message}")

    # --------------------------------------------------------------------------
    # Clue derivation and extraction
    # --------------------------------------------------------------------------

    def _derive_clues_from_grid(self, grid_2d: np.ndarray) -> tuple[ClueList, ClueList]:
        """Derive clues from a 2D grid using run-length encoding."""
        from compactreasoningmodels.utils.grid import derive_clues_from_grid

        return derive_clues_from_grid(grid_2d)

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
    # Format detection and validation
    # --------------------------------------------------------------------------

    def _validate_dict_keys(self, data: dict) -> None:
        """Validate that dictionary has required keys."""
        if "grid" not in data:
            raise ValueError(f"Unrecognized dict keys: {list(data.keys())!r}")

    def _has_explicit_clues(self, data: dict) -> bool:
        """Check if dict has explicit row_clues and col_clues keys."""
        return "row_clues" in data and "col_clues" in data

    def _has_nested_clues(self, data: dict) -> bool:
        """Check if dict has nested clues format."""
        return "clues" in data and self._get_depth(data["clues"]) >= 4

    def _is_list_of_dicts(self, data: list | tuple) -> bool:
        """Check if sequence is a list of example dicts."""
        return isinstance(data[0], dict) and "grid" in data[0]

    def _is_pair_format(self, data: list | tuple) -> bool:
        """Check if sequence is a 2-element pair format."""
        return len(data) == 2

    def _is_field_major_format(self, clue_entry: Any, grid: Any) -> bool:
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

    def _normalize_grid_batch(self, grid_batch: Any) -> list:
        """Normalize grid batch to list of flat lists."""
        if isinstance(grid_batch, np.ndarray):
            return grid_batch.reshape(grid_batch.shape[0], -1).tolist()
        return grid_batch

    def _normalize_grid_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize grid-only array to 3D format (batch, height, width)."""
        if arr.ndim == 2:
            # Flattened grids, need to infer shape
            n_cells = arr.shape[1]
            side = int(round(n_cells ** 0.5))

            if side * side != n_cells:
                raise ValueError(
                    f"Flattened grid with {n_cells} cells isn't square; "
                    "pass a (batch, H, W) array instead."
                )

            return arr.reshape(arr.shape[0], side, side)

        if arr.ndim != 3:
            raise ValueError(
                f"Expected (batch, H, W) or (batch, H*W) array, got ndim={arr.ndim}"
            )

        return arr


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
