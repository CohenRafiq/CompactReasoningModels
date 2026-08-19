from pathlib import Path
import os

import numpy as np
import torch
import json
import itertools

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
        base_dir = Path(os.environ.get("DATA_DIR", "data"))
        with open(base_dir / input_data) as f:
            records = [json.loads(line) for line in f if line.strip()]

        height, width = records[0]["height"], records[0]["width"]
        bad = next((r for r in records if r["height"] != height or r["width"] != width), None)
        if bad is not None:
            raise ValueError(
                f"All puzzles must have the same dimensions. "
                f"Expected {height}x{width}, got {bad['height']}x{bad['width']} (id={bad.get('id')})"
            )

        n = len(records)
        max_row_clue_len = (width + 1) // 2
        max_col_clue_len = (height + 1) // 2

        def pad_clues(all_clues: list[list[int]], count_per_rec: int, max_len: int, label: str) -> np.ndarray:
            lengths = np.fromiter((len(c) for c in all_clues), dtype=np.int64, count=len(all_clues))
            overflow = np.flatnonzero(lengths > max_len)
            if overflow.size:
                idx = int(overflow[0])
                rec = records[idx // count_per_rec]
                raise ValueError(f"{label} clue {all_clues[idx]} exceeds max length {max_len} (id={rec.get('id')})")

            out = np.zeros((len(all_clues), max_len), dtype=np.float32)
            mask = np.arange(max_len) < lengths[:, None]
            out[mask] = np.fromiter(itertools.chain.from_iterable(all_clues), dtype=np.float32)
            return out.reshape(n, count_per_rec, max_len)

        row_clues = [clue for rec in records for clue in rec["rows"]]
        col_clues = [clue for rec in records for clue in rec["cols"]]

        rows_t = torch.from_numpy(pad_clues(row_clues, height, max_row_clue_len, "Row"))
        cols_t = torch.from_numpy(pad_clues(col_clues, width, max_col_clue_len, "Col"))
        grid_t = torch.from_numpy(np.array([rec["grid"] for rec in records], dtype=np.float32))

        X = torch.cat([rows_t.flatten(1), cols_t.flatten(1)], dim=1)
        return X, grid_t