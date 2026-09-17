from abc import ABC, abstractmethod

import numpy as np
import torch

from ...utils.types import ShapeError
from ..blocks.base import Block


class BaseExperiment(ABC):
    def __init__(
        self, blocks: list[Block], data: torch.Tensor | np.ndarray, name: str = "experiment"
    ):
        self.blocks: list[Block] = []
        self.name = name
        self._shape_validated = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if blocks:
            for block in blocks:
                self.add_block(block)

        self.data = self._to_tensor(data)
        self.validate_shapes(self.data.shape)
        self.run()

    def add_block(self, block: Block) -> "BaseExperiment":
        self.blocks.append(block)
        self._shape_validated = False
        return self

    def validate_shapes(self, input_shape: tuple[int, ...]) -> None:
        # (solvers, batch, steps, grid)
        if len(input_shape) != 4:
            raise ShapeError(
                f"Input shape must be 4D (solvers, batch, steps, grid), got {len(input_shape)}D"
            )

        current_shape: tuple[int, ...] = input_shape
        for i, block in enumerate(self.blocks):
            try:
                next_shape = block.compute_output_shape(current_shape)
                current_shape = next_shape
            except ShapeError as e:
                raise ShapeError(
                    f"Shape validation failed at block {i + 1} ('{block.name}'): {str(e)}"
                ) from e
        if current_shape != self.output_shape:
            raise ShapeError(f"Final output shape must be {self.output_shape}, got {current_shape}")
        self._shape_validated = True

    def _to_tensor(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, device=self.device)
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Input data must be a torch.Tensor or np.ndarray, got {type(data)}")
        return data.to(self.device)

    @property
    @abstractmethod
    def output_shape(self):
        pass

    @abstractmethod
    def run(self) -> torch.Tensor:
        pass

    def __repr__(self):
        return f"Experiment(name='{self.name}', blocks={len(self.blocks)})"
