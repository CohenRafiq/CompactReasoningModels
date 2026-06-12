from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PuzzleDataset(Dataset):
    def __init__(
        self,
        input_path: str | Path,
        target_path: str | Path | None = None,
        target_shape: Tuple[int, ...] | None = None,
    ):
        super().__init__()
        base_dir = Path(__import__("os").environ.get("DATA_DIR", "data"))
        self.input_path = base_dir / input_path
        self.target_path = base_dir / target_path if target_path else None

        self.X = self._load(self.input_path)
        self.y = self._load(self.target_path) if self.target_path else None

        self.input_shape = self.X.shape[1:]
        self.target_shape = target_shape

    def flatten(self):
        self.X = self.X.flatten(start_dim=1)
        self.input_shape = self.X[0].numel()
        if self.y is not None:
            self.y = self.y.flatten(start_dim=1)
            self.target_shape = self.y[0].numel()
        else:
            self.target_shape = int(np.prod(self.target_shape)) if self.target_shape is not None else None
    
    def _load(self, path: Path) -> torch.Tensor:
        return torch.from_numpy(np.load(path)).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor | torch.Tensor]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:      
            return self.X[idx], None