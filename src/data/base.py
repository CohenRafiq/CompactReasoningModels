from  torch.utils.data import Dataset
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
            input_shape: tuple, 
            target_shape: tuple,
            flat: bool = False):
        super().__init__()
        self.input_path = input_path
        self.target_path = target_path
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.flat = flat

        self.inputs = self.__import_inputs__(self.input_path)
        self.targets = self.__import_targets__(self.target_path)


        self.flat_inputs = self.inputs.flatten(start_dim=1) if self.inputs is not None else None
        self.flat_targets = self.targets.flatten(start_dim=1) if self.targets is not None else None
        self.flat_input_size = self.flat_inputs.shape[1] if self.flat_inputs is not None else None
        self.flat_target_size = self.flat_targets.shape[1] if self.flat_targets is not None else None

        print(f"Flat input size: {self.flat_input_size}, Flat target size: {self.flat_target_size}")

    @abstractmethod
    def __import_inputs__(self, path: Path | str) -> Tensor:
        """
        Method to import input data from the specified path.
        """
        pass

    @abstractmethod
    def __import_targets__(self, path: Path | str) -> Tensor:
        """
        Method to import target data from the specified path.
        """
        pass

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        if self.flat:
            return self.flat_inputs[idx], self.flat_targets[idx]
        return self.inputs[idx], self.targets[idx]

