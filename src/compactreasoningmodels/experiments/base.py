from abc import ABC, abstractmethod
import numpy as np
from typing import Any

from compactreasoningmodels.utils.types import SolverProfile
from compactreasoningmodels.solving_traces.similarity_measures import SimilarityMeasure



class BaseExperiment(ABC):

    @abstractmethod
    def run_experiment(self, heatmaps: dict[SolverProfile, list[tuple[np.ndarray, np.ndarray]]],
                       metrics: list[SimilarityMeasure] | None = None, 
                        **kwargs) -> dict[SolverProfile, dict[Any, float]]:
        ...

    @abstractmethod
    def display_results(self, results: dict[SolverProfile, dict[Any, float]]):
        ...
