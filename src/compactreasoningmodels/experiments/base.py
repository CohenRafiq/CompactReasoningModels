from abc import ABC, abstractmethod
from collections.abc import Generator

import numpy as np
import torch

from compactreasoningmodels.datasets.jsonl import JsonlDataset
from compactreasoningmodels.solving_traces.arc_consistency import ArcConsistency
from compactreasoningmodels.solving_traces.discrete_genetic import DiscreteGeneticAlgorithm
from compactreasoningmodels.solving_traces.global_min_violations import GlobalMinViolations
from compactreasoningmodels.solving_traces.local_min_violations import LocalMinViolations
from compactreasoningmodels.solving_traces.model_solver import ModelSolver
from compactreasoningmodels.solving_traces.similarity_measures import (
    MAE,
    MSE,
    HuberLoss,
    MeanCosineSimilarity,
    PearsonCorrelation,
    SSIMSimilarity,
    WassersteinSimilarity,
)


class BaseExperiment(ABC):

    _all_solvers = {
        "arc_consistency": ArcConsistency,
        "local_min_violations": LocalMinViolations,
        "global_min_violations": GlobalMinViolations,
        "discrete_genetic": DiscreteGeneticAlgorithm,
        "model_solver": ModelSolver
    }

    _all_metrics = {
        "MSE": MSE,
        "MAE": MAE,
        "Huber Loss": HuberLoss,
        "Mean Cosine Similarity": MeanCosineSimilarity,
        "Pearson Correlation": PearsonCorrelation,
        "SSIM": SSIMSimilarity,
        "Wasserstein Similarity": WassersteinSimilarity
    }

    def __init__(self, solvers: list[str],
                 metrics: list[str], dataset_path: str,
                 target_shape: tuple[int, int] = (5, 5)):
        self.solvers = {solver_name: self._all_solvers[solver_name] for solver_name in solvers}
        self.metrics = {metric_name: self._all_metrics[metric_name] for metric_name in metrics}
        self.target_shape = target_shape
        self.dataset = JsonlDataset(dataset_path, target_shape=target_shape)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False)

    def generate_heatmaps(self, num_samples: int = 100, initial_grid: str = "random",
                           **solver_kwargs):
        for i, (input_tensor, target_grid) in enumerate(self.dataloader):
            if i >= num_samples:
                break
            if initial_grid == "random":
                initial_grid_array = np.random.rand(*self.target_shape)
            elif initial_grid == "zeros":
                initial_grid_array = np.zeros(self.target_shape)
            else:
                raise ValueError(f"Unknown initial_grid option: {initial_grid}")

            for solver_name, num_steps in self.next_solver(**solver_kwargs):
                solver_class = self.solvers[solver_name]
                solver_instance = solver_class(
                    input_tensor, self.target_shape,
                    initial_grid=initial_grid_array)
                heatmap = solver_instance.heatmap(num_steps=num_steps)
                target_np = target_grid.squeeze().cpu().numpy().reshape(self.target_shape)
                yield (solver_name, num_steps, heatmap, target_np)

    @abstractmethod
    def next_solver(self, **kwargs) -> Generator[tuple[str, int], None, None]:
        # solver name, number of steps
        ...

    @abstractmethod
    def run_experiment(self, num_samples: int = 100, initial_grid: str = "random"):
        ...
