import itertools
import pandas as pd
import torch
import numpy as np

from compactreasoningmodels.experiments.base import BaseExperiment
from compactreasoningmodels.solving_traces.similarity_measures import SimilarityMeasure
from compactreasoningmodels.utils.types import SolverProfile


class NoisySimilarityExperiment(BaseExperiment):

    def _add_noise_to_grid(self, grid: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        noise = np.random.rand(*grid.shape) * 2 * noise_level - noise_level
        return (1 - noise_level) * grid + noise

    def run_experiment(self, heatmaps: dict[SolverProfile, list[tuple[np.ndarray, np.ndarray]]], 
                       metrics: list[SimilarityMeasure] | None = None, 
                       **kwargs) -> dict[SolverProfile, dict[tuple[SolverProfile, SolverProfile], float]]:
        clean_heatmaps = {
            key: [grid for grid, _, _ in grids]
            for key, grids in heatmaps.items()
        }
        noisy_heatmaps = {
            key: [self._add_noise_to_grid(grid, noise_level=kwargs.get("noise_level", 0.1)) for grid in grids]
            for key, grids in clean_heatmaps.items()
        }
        results = {}
        for metric in metrics or []:
            metric_results = {}
            for clean_profile, noisy_profile in itertools.product(clean_heatmaps.keys(), repeat=2):
                score = metric()(clean_heatmaps[clean_profile], noisy_heatmaps[noisy_profile]).mean().item()
                metric_results[(clean_profile, noisy_profile)] = score
            results[metric] = metric_results
        return results

    def display_results(
        self,
        results: dict[SimilarityMeasure, dict[tuple[SolverProfile, SolverProfile], float]],
    ) -> None:
        for metric, metric_results in results.items():
            series = pd.Series(metric_results)
            series.index.names = ["clean", "noisy"]
            matrix = series.unstack("noisy").round(4)

            label = lambda p: f"{p.name} (steps={p.num_steps})"
            matrix.index = matrix.index.map(label)
            matrix.columns = matrix.columns.map(label)

            print(f"{metric.__name__} Matrix (rows=clean, cols=noisy):")
            print(matrix)
            print()
