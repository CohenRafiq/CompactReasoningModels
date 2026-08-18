from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class PuzzleDataset(Dataset):
    def __init__(
        self,
        input_data: Union[str, Path, torch.Tensor, np.ndarray] = None,
        target_data: Union[str, Path, torch.Tensor, np.ndarray, None] = None,
        target_shape: Tuple[int, ...] | None = None,
        split_categories: bool = False,
    ):
        super().__init__()
        
        # Use helper function to load data
        self.X = self._load_data(input_data)
        self.y = self._load_data(target_data)
        
        if self.X is None:
            raise ValueError("input_data cannot be None")
        
        self.input_shape = self.X.shape[1:]
        self.target_shape = target_shape if target_shape else (self.y.shape[1:] if self.y is not None else None)
        self.split_categories = split_categories
        
        if split_categories:
            self._split_categories()

    def _load_data(self, data: Union[str, Path, torch.Tensor, np.ndarray, None]) -> torch.Tensor | None:
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
        self.input_shape = self.X[0].numel()
        self.y = self.y.flatten(start_dim=1) if self.y is not None else None
        self.target_shape = int(np.prod(self.target_shape))

    def _split_categories(self):
        self.y = torch.stack([self.y, 1 - self.y], dim=-1)
        self.target_shape = self.y.shape[1:]
    
    def _load(self, path: Path) -> torch.Tensor:
        return torch.from_numpy(np.load(path)).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor | torch.Tensor]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:      
            return self.X[idx], None