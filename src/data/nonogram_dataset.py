from src.data.base import BaseDataset
from torch import Tensor
import torch
from pathlib import Path
import numpy as np


class NonogramDataset(BaseDataset):
    def __init__(
            self, 
            input_path: Path | str, 
            target_path: Path | str, 
            nonogram_size: int,
            flat: bool = False):
        
        self.nonogram_size = nonogram_size
        self.flat = flat

        # Input shape (per sample): 
        #    - 2 channels: one for row clues, one for column clues
        #    - Each row/column has nonogram_size clues
        #    - each clue is represented with multiple numbers
        self._x_shape = (2, nonogram_size, (nonogram_size + 1) // 2)
        self._y_shape = (nonogram_size, nonogram_size)
        
        super().__init__(input_path, target_path, flat)

    @property
    def X_shape(self) -> tuple:
        """Shape of X tensor: (n_samples, *sample_shape)"""
        return self.X.shape if self.X is not None else None
    
    @property
    def y_shape(self) -> tuple:
        """Shape of y tensor: (n_samples, *sample_shape)"""
        return self.y.shape if self.y is not None else None

    def _import_inputs(self, path: Path) -> Tensor:
        inputs = np.load(path)['arr_0']
        num_samples = inputs.shape[0]
        x_shape_with_samples = (num_samples,) + self._x_shape
        inputs = inputs.reshape(x_shape_with_samples)
        return torch.tensor(inputs, dtype=torch.float32)

    def _import_targets(self, path: Path) -> Tensor:
        targets = np.load(path)['arr_0']
        num_samples = targets.shape[0]
        y_shape_with_samples = (num_samples,) + self._y_shape
        targets = targets.reshape(y_shape_with_samples)
        return torch.tensor(targets, dtype=torch.float32)