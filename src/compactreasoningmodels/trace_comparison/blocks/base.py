from abc import ABC, abstractmethod

import torch


class Block(ABC):
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.requires_numpy = False
        self.input_shape: tuple[int, ...] | None = None
        self.output_shape: tuple[int, ...] | None = None

    @abstractmethod
    def check_input_shape(self, input_shape: tuple[int, ...]) -> None:
        pass

    @abstractmethod
    def _output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        pass

    def compute_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.check_input_shape(input_shape)
        return self._output_shape(input_shape)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        self.input_shape = data.shape
        self.check_input_shape(data.shape)
        output = self.forward(data)
        self.output_shape = output.shape
        if self.output_shape != self._output_shape(self.input_shape):
            raise ValueError(
                f"Output shape {self.output_shape} does not match "
                f"computed output shape {self._output_shape(self.input_shape)}"
            )
        return output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(input_shape={self.input_shape}, "
            f"output_shape={self.output_shape})"
         )
