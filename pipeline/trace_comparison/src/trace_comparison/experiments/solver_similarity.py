import numpy as np
import torch
from matplotlib import pyplot as plt

from pipeline.trace_comparison.src.trace_comparison.experiments import BaseExperiment


class FullDatasetSimilarityExperiment(BaseExperiment):
    def __init__(
        self,
        blocks: list,
        data: torch.Tensor | np.ndarray,
        name: str = "full_dataset_similarity_experiment",
    ):
        super().__init__(blocks=blocks, data=data, name=name)

    def run(self) -> torch.Tensor:
        current_data = self.data
        for block in self.blocks:
            current_data = block(current_data)
        self.correlation_matrix = current_data
        return current_data

    def get_correlation_matrix(self) -> torch.Tensor:
        if not hasattr(self, "correlation_matrix"):
            raise ValueError("Experiment has not been run yet. Please call the 'run' method first.")
        return self.correlation_matrix

    def get_similarity(self, idx1: int, idx2: int) -> float:
        if not hasattr(self, "correlation_matrix"):
            raise ValueError("Experiment has not been run yet. Please call the 'run' method first.")
        return self.correlation_matrix[idx1, idx2].item()

    def display_correlation_matrix(self):
        if not hasattr(self, "correlation_matrix"):
            raise ValueError("Experiment has not been run yet. Please call the 'run' method first.")

        plt.figure(figsize=(8, 6))
        plt.imshow(self.correlation_matrix.cpu().numpy(), cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Similarity Score")
        plt.title(f"Correlation Matrix: {self.name}")
        plt.xlabel("Solver Index")
        plt.ylabel("Solver Index")
        plt.xticks(
            ticks=np.arange(self.correlation_matrix.shape[0]),
            labels=np.arange(self.correlation_matrix.shape[0]),
        )
        plt.yticks(
            ticks=np.arange(self.correlation_matrix.shape[0]),
            labels=np.arange(self.correlation_matrix.shape[0]),
        )
        plt.show()
