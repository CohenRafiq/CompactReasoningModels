import torch

from ...trace_comparison.blocks.base import Block
from ...utils.types import ShapeError


class CKABlock(Block):
    def __init__(self, name: str = "CKA"):
        super().__init__(name)

    def check_input_shape(self, input_shape: tuple) -> None:
        if len(input_shape) != 3:
            raise ShapeError(
                f"Input shape must be 3D (2, batch_size, feature_dim), got {len(input_shape)}D"
            )
        if input_shape[0] != 2:
            raise ShapeError(f"First dimension must be 2 for CKA comparison, got {input_shape[0]}")

    def _output_shape(self, input_shape: tuple) -> tuple:
        return (1,)

    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        tensor_a, tensor_b = x[0], x[1]
        tensor_a = tensor_a - tensor_a.mean(dim=0, keepdim=True)
        tensor_b = tensor_b - tensor_b.mean(dim=0, keepdim=True)

        cross = torch.norm(tensor_a.T @ tensor_b, p="fro") ** 2
        norm_a = torch.norm(tensor_a.T @ tensor_a, p="fro") + eps
        norm_b = torch.norm(tensor_b.T @ tensor_b, p="fro") + eps

        cka_value = cross / (norm_a * norm_b + eps)
        return cka_value.view(1)
