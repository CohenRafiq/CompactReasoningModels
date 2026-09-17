import torch

from .base import Block
from .....utils.src.utils.types import ShapeError


class MDSBlock(Block):
    """
    Classical Multidimensional Scaling (MDS) embedding.

    Input: (batch_size, batch_size) - pairwise distance matrix
    Output: (batch_size, n_components) - embedded coordinates
    """

    def __init__(self, n_components: int = 2, name: str = "MDS"):
        super().__init__(name)
        self.n_components = n_components
        self.requires_numpy = True

    def check_input_shape(self, input_shape: tuple) -> None:
        if len(input_shape) != 2:
            raise ShapeError(
                f"Input shape must be 2D (batch_size, batch_size), got {len(input_shape)}D"
            )
        if input_shape[0] != input_shape[1]:
            raise ShapeError(f"Input must be square distance matrix, got {input_shape}")

    def _output_shape(self, input_shape: tuple) -> tuple:
        batch_size = input_shape[0]
        return (batch_size, self.n_components)

    def forward(self, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """
        Classical MDS using eigendecomposition.

        1. Convert distance matrix to similarity matrix
        2. Center the similarity matrix
        3. Eigendecomposition
        4. Take top eigenvectors scaled by sqrt(eigenvalues)
        """
        # Ensure symmetric
        D = x
        n = D.shape[0]

        # Convert distances to similarities (squared distances)
        D_squared = D**2

        # Centering matrix
        H = torch.eye(n) - torch.ones(n, n) / n

        # Double centering: B = -0.5 * H * D^2 * H
        B = -0.5 * H @ D_squared @ H

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(B)

        # Sort in descending order
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top components
        eigenvalues = eigenvalues[: self.n_components]
        eigenvectors = eigenvectors[:, : self.n_components]

        # Handle negative eigenvalues (numerical errors)
        eigenvalues = torch.clamp(eigenvalues, min=0)

        # Scale eigenvectors by sqrt of eigenvalues
        embedding = eigenvectors * torch.sqrt(eigenvalues + eps)

        return embedding
