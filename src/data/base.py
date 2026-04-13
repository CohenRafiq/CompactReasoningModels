from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from pathlib import Path
from torch import Tensor


class BaseDataset(Dataset, ABC):
    """
    Base Dataset class for PyTorch datasets.
    """
    def __init__(
            self, 
            input_path: Path | str, 
            target_path: Path | str,
            flat: bool = False):
        super().__init__()
        self.input_path = Path(input_path)
        self.target_path = Path(target_path)
        self.flat = flat

        self.X = self._import_inputs(self.input_path)
        self.y = self._import_targets(self.target_path)
        if flat:
            self.X = self.X.flatten(start_dim=1) if self.X is not None else None
            self.y = self.y.flatten(start_dim=1) if self.y is not None else None

    @abstractmethod
    def _import_inputs(self, path: Path) -> Tensor:
        """
        Method to import input data from the specified path.
        """
        pass

    @abstractmethod
    def _import_targets(self, path: Path) -> Tensor:
        """
        Method to import target data from the specified path.
        """
        pass

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]