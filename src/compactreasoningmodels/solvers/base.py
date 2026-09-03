from abc import ABC, abstractmethod

import numpy as np

from compactreasoningmodels.utils.puzzle_types import Clues
from compactreasoningmodels.utils.grid import normalise_clues, blank_grid, grid_shape_from_clues


class BaseSolver(ABC):
    
    def step(self, clues: Clues, prev: np.ndarray = None, 
             num_steps: int = 1, sampling_ratio: float = 1.0,
             step_ratio: int = 1) -> np.ndarray:
        grid_shape = grid_shape_from_clues(clues)
        clues = normalise_clues(clues)
        prev = prev if prev is not None else blank_grid(*grid_shape)
        actual_steps = num_steps * step_ratio
        all_steps = self._step(clues, prev, actual_steps, sampling_ratio=sampling_ratio)
        reduced_steps = [all_steps[i] for i in range(0, len(all_steps), step_ratio)]
        return reduced_steps

    @abstractmethod
    def _step(self, clues: np.ndarray, prev: np.ndarray, num_steps: int, sampling_ratio: float) -> np.ndarray:
        pass
