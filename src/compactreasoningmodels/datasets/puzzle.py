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

        self.X, self.y = self._load(input_data, target_data)

        if self.X is None:
            raise ValueError("input_data cannot be None")

        self._input_shape = self.X.shape[1:]
        self._target_shape = (
            target_shape if target_shape else (self.y.shape[1:] if self.y is not None else None)
        )
        self.split_categories = split_categories

        if split_categories:
            self._split_categories()

    @staticmethod
    def _load(input_data: str | Path | torch.Tensor | np.ndarray | None,
              target_data: str | Path | torch.Tensor | np.ndarray | None
              ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return PuzzleDataset._load_data(input_data), PuzzleDataset._load_data(target_data)

    @staticmethod
    def _load_data(data: str | Path | torch.Tensor | np.ndarray | None) -> torch.Tensor | None:
        if data is None:
            return None
        if isinstance(data, (str, Path)):
            base_dir = Path(__import__("os").environ.get("DATA_DIR", "data"))
            path = base_dir / data
            return torch.from_numpy(np.load(path)).float()
        elif isinstance(data, torch.Tensor):
            return data.float()
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        else:
            raise ValueError(
                f"data must be a path string, Path object, torch.Tensor, numpy array, or None. "
                f"Got {type(data).__name__}"
            )

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
