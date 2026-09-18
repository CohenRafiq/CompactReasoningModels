import torch

from utils.types import ShapeError

from .base import Block


class DTWBlock(Block):
    """
    Computes pairwise DTW distances between solving traces.

    Input: (batch_size, seq_len, feature_dim) - batch of traces
    Output: (batch_size, batch_size) - pairwise DTW distance matrix
    """

    def __init__(self, name: str = "DTW"):
        super().__init__(name)
        self.requires_numpy = True  # DTW computation is easier in numpy

    def check_input_shape(self, input_shape: tuple) -> None:
        if len(input_shape) != 3:
            raise ShapeError(
                f"Input shape must be 3D (batch_size, seq_len, feature_dim), "
                f"got {len(input_shape)}D"
            )

    def _output_shape(self, input_shape: tuple) -> tuple:
        batch_size = input_shape[0]
        return (batch_size, batch_size)

    def _dtw_distance(self, seq_a: torch.Tensor, seq_b: torch.Tensor) -> float:
        """
        Compute DTW distance between two sequences.
        seq_a: (len_a, feature_dim)
        seq_b: (len_b, feature_dim)
        """
        len_a, len_b = seq_a.shape[0], seq_b.shape[0]

        # Initialize DTW matrix
        dtw_matrix = torch.full((len_a + 1, len_b + 1), float("inf"))
        dtw_matrix[0, 0] = 0

        # Compute pairwise distances
        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                cost = torch.norm(seq_a[i - 1] - seq_b[j - 1])
                dtw_matrix[i, j] = cost + torch.min(
                    torch.stack(
                        [
                            dtw_matrix[i - 1, j],  # insertion
                            dtw_matrix[i, j - 1],  # deletion
                            dtw_matrix[i - 1, j - 1],  # match
                        ]
                    )
                )

        return dtw_matrix[len_a, len_b].item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        distance_matrix = torch.zeros(batch_size, batch_size)

        # Compute pairwise DTW distances
        for i in range(batch_size):
            for j in range(i, batch_size):
                dist = self._dtw_distance(x[i], x[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetric

        return distance_matrix
