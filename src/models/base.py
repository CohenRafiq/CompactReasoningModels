from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    require_flat_input: bool = False

    @abstractmethod
    def __init__(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], **kwargs):
        super().__init__()
        self._input_shape = input_shape
        self._output_shape = output_shape

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._input_shape

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return self._output_shape
