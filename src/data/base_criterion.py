from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor, nn


class BaseCriterion(nn.Module, ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        ...
