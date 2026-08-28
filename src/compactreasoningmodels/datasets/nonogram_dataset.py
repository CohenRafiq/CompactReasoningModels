from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from compactreasoningmodels.datasets.nonogram_parser import NonogramParser
from compactreasoningmodels.utils import puzzle_types as t



class NonogramDataset(Dataset):
    def __init__(self, data: str | Path | t.Dataset, 
                 batch_size: int = 256, padding: str = "square"):
        super().__init__()

        self.X, self.y, self.meta = NonogramParser(max_size=None, batch_size=batch_size).parse(data)

    

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx], None
