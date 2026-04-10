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

        # Input shape
        #    - 2 channels: one for row clues, one for column clues
        #    - Each row/column has nonogram_size clues
        #    - each clue is represented with multiple numbers (for a 5x5, max clue is [1,1,1])
        input_shape = (2, nonogram_size, (nonogram_size + 1) // 2)
        target_shape = (nonogram_size, nonogram_size)
        super().__init__(input_path, target_path, input_shape, target_shape, flat)

    def __import_inputs__(self, path: Path | str) -> Tensor:
        inputs = np.load(path)['arr_0']
        num_samples = inputs.shape[0]
        self.input_shape = (num_samples,) + self.input_shape
        inputs = inputs.reshape(self.input_shape)
        return torch.tensor(inputs, dtype=torch.float32)

    def __import_targets__(self, path: Path | str) -> Tensor:
        targets = np.load(path)['arr_0']
        num_samples = targets.shape[0]
        self.target_shape = (num_samples,) + self.target_shape
        targets = targets.reshape(self.target_shape)
        return torch.tensor(targets, dtype=torch.float32)