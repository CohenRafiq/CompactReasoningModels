from pathlib import Path

import numpy as np
import torch

from compactreasoningmodels.datasets.base import BaseDataset


class PuzzleDataset(BaseDataset):
    def __init__(
        self,
        input_data: str | Path | torch.Tensor | np.ndarray | None = None,
        target_data: str | Path | torch.Tensor | np.ndarray | None = None,
        target_shape: tuple[int, ...] | None = None,
        split_categories: bool = False,
    ):
        super().__init__()

        self.X = self._load_data(input_data)
        self.y = self._load_data(target_data)

        if self.X is None:
            raise ValueError("input_data cannot be None")

        self._input_shape = self.X.shape[1:]
        self._target_shape = (
            target_shape if target_shape else (self.y.shape[1:] if self.y is not None else None)
        )
        self.split_categories = split_categories

        if split_categories:
            self._split_categories()

    def _split_categories(self):
        self.y = torch.stack([self.y, 1 - self.y], dim=-1)
        self._target_shape = self.y.shape[1:]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx], None
