from abc import ABC, abstractmethod

import numpy as np
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

    def flatten(self):
        self.X = self.X.flatten(start_dim=1)
        self._input_shape = (self.X[0].numel(),)
        if self.y is not None:
            self.y = self.y.flatten(start_dim=1)
            self._target_shape = (int(np.prod(self._target_shape)),)
