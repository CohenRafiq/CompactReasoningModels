from compactreasoningmodels.solving_traces.base import SolvingTrace

import numpy as np

class ArcConsistency(SolvingTrace):
    def step(self, grid: np.ndarray) -> np.ndarray:
        # for each row and column, add to list
        # shuffle the list
        
        pass

    def heatmap_step(self, grid: np.ndarray) -> np.ndarray:
        pass