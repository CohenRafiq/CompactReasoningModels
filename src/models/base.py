import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor

class BaseModel(nn.Module, ABC):
    """
    Base Model class for PyTorch models.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.
        """
        pass