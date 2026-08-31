from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from compactreasoningmodels.solving_traces.similarity_measures import SimilarityMeasure
from compactreasoningmodels.utils.types import SolverProfile


class BaseExperiment(ABC):
    @abstractmethod
    def run_experiment(
        self,
        heatmaps: dict[SolverProfile, list[tuple[np.ndarray, torch.Tensor, np.ndarray]]],
        metrics: list[SimilarityMeasure] | None = None,
        **kwargs,
    ) -> dict[Any, dict[Any, float]]: ...

    @abstractmethod
    def display_results(self, results: dict[Any, dict[Any, float]]) -> None: ...
