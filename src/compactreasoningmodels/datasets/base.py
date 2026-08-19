from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self._input_shape: tuple[int, ...] | None = None
        self._target_shape: tuple[int, ...] | None = None

    @abstractmethod
    def __getitem__(self, idx: int): ...

    @abstractmethod
    def __len__(self) -> int: ...

    @property
    def input_shape(self) -> tuple[int, ...] | None:
        return self._input_shape

    @property
    def target_shape(self) -> tuple[int, ...] | None:
        return self._target_shape

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

    def flatten(self):
        self.X = self.X.flatten(start_dim=1)
        self._input_shape = (self.X[0].numel(),)
        if self.y is not None:
            self.y = self.y.flatten(start_dim=1)
            self._target_shape = (int(np.prod(self._target_shape)),)
