import torch
import torch.nn.functional as F

from utils.types import ShapeError

from .base import Block


class FlattenBlock(Block):
    def __init__(self, name: str = "Flatten", start_dim: int = 1):
        super().__init__(name)
        self.start_dim = start_dim

    def check_input_shape(self, input_shape: tuple) -> None:
        if len(input_shape) <= self.start_dim:
            raise ShapeError(
                f"Input shape must be at least {self.start_dim + 1}D, got {len(input_shape)}D"
            )

    def _output_shape(self, input_shape: tuple) -> tuple:
        flattened_dim = 1
        for x in input_shape[self.start_dim :]:
            flattened_dim *= x
        new_shape = input_shape[: self.start_dim] + (flattened_dim,)
        return tuple(new_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, start_dim=self.start_dim)


class MSEBlock(Block):
    def __init__(self, name: str = "MSE", start_dim: int = 0):
        super().__init__(name)
        self.start_dim = start_dim

    def check_input_shape(self, input_shape: tuple) -> None:
        if len(input_shape) <= self.start_dim:
            raise ShapeError(
                f"Input shape must be at least {self.start_dim + 1}D, got {len(input_shape)}D"
            )
        if input_shape[0] != 2:
            raise ShapeError(f"First dimension must be 2 for MSE comparison, got {input_shape[0]}")

    def _output_shape(self, input_shape: tuple) -> tuple:
        if self.start_dim == 0:
            return (1,)
        return input_shape[: self.start_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.mse_loss(x[0], x[1], reduction="none")
        return out.flatten(start_dim=self.start_dim).mean().unsqueeze(-1)


class CorrelationMatrixWrapper(Block):
    def __init__(self, base_block: Block, name: str | None = None):
        if name is None:
            name = f"{base_block.name}_with_corr"
        super().__init__(name)
        self.base_block = base_block

    def check_input_shape(self, input_shape: tuple) -> None:
        if len(input_shape) < 3:
            raise ShapeError(f"Input shape must be at least 3D, got {len(input_shape)}D")
        if input_shape[0] < 2:
            raise ShapeError(f"First dimension must be at least 2, got {input_shape[0]}")
        self.base_block.check_input_shape((2,) + input_shape[1:])

    def _output_shape(self, input_shape: tuple) -> tuple:
        n = input_shape[0]
        return (n, n)

    def _compute_pairwise(self, x: torch.Tensor, compute_fn) -> torch.Tensor:
        num_tensors = x.shape[0]
        matrix = torch.zeros(num_tensors, num_tensors, device=x.device, dtype=x.dtype)

        for i in range(num_tensors):
            for j in range(i, num_tensors):
                pair_input = torch.stack([x[i], x[j]], dim=0)
                value = compute_fn(pair_input).mean()
                matrix[i, j] = value
                matrix[j, i] = value

        return matrix

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._compute_pairwise(x, lambda pair: self.base_block.forward(pair, **kwargs))
