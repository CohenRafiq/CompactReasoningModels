import torch
from ...trace_comparison.blocks.base import Block
from ...utils.types import ShapeError

class FlattenBlock(Block):
    
    def __init__(self, name: str = "Flatten", start_dim: int = 1):
        super().__init__(name)
        self.start_dim = start_dim

    def check_input_shape(self, input_shape: tuple) -> None:
        if len(input_shape) <= self.start_dim:
            raise ShapeError(f"Input shape must be at least {self.start_dim + 1}D, got {len(input_shape)}D")

    def _output_shape(self, input_shape: tuple) -> tuple:
        flattened_dim = 1
        for x in input_shape[self.start_dim:]:
            flattened_dim *= x
        new_shape = input_shape[:self.start_dim] + (flattened_dim,)
        return tuple(new_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=self.start_dim)

class MSEBlock(Block):
    
    def __init__(self, name: str = "MSE", start_dim: int = 0):
        super().__init__(name)
        self.start_dim = start_dim

    def check_input_shape(self, input_shape: tuple) -> None:
        if len(input_shape) <= self.start_dim:
            raise ShapeError(f"Input shape must be at least {self.start_dim + 1}D, got {len(input_shape)}D")
        if input_shape[0] != 2:
            raise ShapeError(f"First dimension must be 2 for MSE comparison, got {input_shape[0]}")

    def _output_shape(self, input_shape: tuple) -> tuple:
        if self.start_dim == 0:
            return (1,)
        return input_shape[:self.start_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.functional.F.mse_loss(x[0], x[1], reduction='none')
        out = out.flatten(start_dim=self.start_dim).mean().unsqueeze(-1)
        return out