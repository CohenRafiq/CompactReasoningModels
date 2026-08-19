from pathlib import Path

import numpy as np
import torch
import json

from compactreasoningmodels.datasets.puzzle import PuzzleDataset


class JsonlDataset(PuzzleDataset):
    def __init__(
        self,
        input_data: str | Path | torch.Tensor | np.ndarray | None = None,
        target_data: None = None,
        target_shape: tuple[int, ...] | None = None,
        split_categories: bool = False,
    ):
        super().__init__(
            input_data=input_data,
            target_data=target_data,
            target_shape=target_shape,
            split_categories=split_categories,
        )

    @staticmethod
    def _load(input_data: str | Path | torch.Tensor | np.ndarray | None,
              target_data: None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        base_dir = Path(__import__("os").environ.get("DATA_DIR", "data"))
        with open(base_dir / input_data) as f:
            records = [json.loads(line) for line in f if line.strip()]

        height, width = records[0]["height"], records[0]["width"]
        for rec in records:
            if rec["height"] != height or rec["width"] != width:
                raise ValueError(
                    f"All puzzles must have the same dimensions. "
                    f"Expected {height}x{width}, got {rec['height']}x{rec['width']} (id={rec.get('id')})"
                )

        max_row_clue_len = (width + 1) // 2
        max_col_clue_len = (height + 1) // 2

        n = len(records)
        rows_t = torch.zeros(n, height, max_row_clue_len)
        cols_t = torch.zeros(n, width, max_col_clue_len)
        grid_t = torch.zeros(n, height, width)

        for i, rec in enumerate(records):
            for j, clue in enumerate(rec["rows"]):
                if len(clue) > max_row_clue_len:
                    raise ValueError(f"Row clue {clue} exceeds max length {max_row_clue_len} (id={rec.get('id')})")
                rows_t[i, j, :len(clue)] = torch.tensor(clue, dtype=torch.float32)
            for j, clue in enumerate(rec["cols"]):
                if len(clue) > max_col_clue_len:
                    raise ValueError(f"Col clue {clue} exceeds max length {max_col_clue_len} (id={rec.get('id')})")
                cols_t[i, j, :len(clue)] = torch.tensor(clue, dtype=torch.float32)
            grid_t[i] = torch.tensor(rec["grid"], dtype=torch.float32)

        X = torch.cat([rows_t.flatten(1), cols_t.flatten(1)], dim=1)
        return X, grid_t