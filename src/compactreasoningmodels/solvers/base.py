from abc import ABC, abstractmethod

import numpy as np
import torch

from compactreasoningmodels.losses.clue_reconstruction import ClueReconstructionLoss
from compactreasoningmodels.utils.puzzle_types import Clues
from compactreasoningmodels.utils.grid import normalise_clues, blank_grid, grid_shape_from_clues, is_solved


class BaseSolver(ABC):

    defualt_step_ratio: int = 1

    def try_solve(self, clues: Clues, grid: np.ndarray, 
                  max_steps: int = 50, sampling_ratio: float = 1.0, 
                  step_ratio: int | None = None) -> tuple:

        if step_ratio is None:
            step_ratio = self.defualt_step_ratio

        steps = self.step(clues, num_steps=max_steps, sampling_ratio=sampling_ratio, step_ratio=step_ratio)
        min_solve_steps = []
        solved = False
        for step in steps:
            min_solve_steps.append(step)
            if np.array_equal(np.round(step).flatten(), grid):
                solved = True
                break
        
        cr_losses = [ClueReconstructionLoss(reduction="mean")(torch.from_numpy(step).reshape(1, 25), clues.unsqueeze(0).flatten(1)) for step in steps]
        mse_losses = [np.mean((step.flatten() - np.array(grid)) ** 2) for step in steps]

        return steps, cr_losses, mse_losses, len(steps), solved, step_ratio
    

    def step(self, clues: Clues, prev: np.ndarray = None, 
             num_steps: int = 1, sampling_ratio: float = 1.0,
             step_ratio: int | None = None) -> np.ndarray:
        if step_ratio is None:
            step_ratio = self.defualt_step_ratio
        grid_shape = grid_shape_from_clues(clues)
        clues = normalise_clues(clues)
        prev = prev if prev is not None else blank_grid(*grid_shape)
        actual_steps = num_steps * step_ratio
        all_steps = self._step(clues, prev, actual_steps, sampling_ratio=sampling_ratio)
        reduced_steps = [all_steps[i] for i in range(0, len(all_steps), step_ratio)]

        return np.array(reduced_steps)

    @abstractmethod
    def _step(self, clues: np.ndarray, prev: np.ndarray, num_steps: int, sampling_ratio: float) -> np.ndarray:
        pass
