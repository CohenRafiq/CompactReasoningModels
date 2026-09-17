from abc import ABC, abstractmethod

import numpy as np
import torch

from compactreasoningmodels.losses.clue_reconstruction import ClueReconstructionLoss
from compactreasoningmodels.utils.grid import (
    blank_grid,
    grid_shape_from_clues,
    normalise_clues,
)
from compactreasoningmodels.utils.puzzle_types import Clues


class BaseSolver(ABC):
    default_step_ratio: int = 1

    def __init__(self, **kwargs):
        self._cr_loss_fn = ClueReconstructionLoss(reduction="none")
        super().__init__(**kwargs)

    def try_solve(
        self,
        clues: Clues,
        grid: np.ndarray,
        max_steps: int = 10,
        sampling_ratio: float = 1.0,
        step_ratio: int | None = None,
    ) -> tuple:

        if step_ratio is None:
            step_ratio = self.default_step_ratio

        steps = self.step(
            clues, num_steps=max_steps, sampling_ratio=sampling_ratio, step_ratio=step_ratio
        )
        solved = False
        steps_to_solve = -1
        for i, step in enumerate(steps):
            if np.array_equal(np.round(step).flatten(), grid):
                solved = True
                steps_to_solve = i + 1
                break

        # Batch loss computation: compute CR loss and MSE for all steps at once
        num_steps_actual = len(steps)
        if num_steps_actual > 0:
            # Stack all steps into a single tensor: (num_steps, H, W)
            steps_tensor = torch.from_numpy(np.array(steps)).float()
            grid_flat = steps_tensor.reshape(num_steps_actual, -1)  # (num_steps, S)

            # Batch clues tensor: repeat for each step
            clues_tensor = torch.from_numpy(np.asarray(clues)).float()
            clues_flat = clues_tensor.unsqueeze(0).flatten(1).expand(num_steps_actual, -1)

            # Batch CR loss computation
            with torch.no_grad():
                cr_losses_batch = self._cr_loss_fn(grid_flat, clues_flat)
                if isinstance(cr_losses_batch, tuple):
                    cr_losses_batch = cr_losses_batch[0]  # per_sample loss
                cr_losses = cr_losses_batch.tolist()

            # Batch MSE computation
            target = np.array(grid).flatten()
            mse_losses = np.mean(
                (steps_tensor.numpy().reshape(num_steps_actual, -1) - target) ** 2, axis=1
            ).tolist()
        else:
            cr_losses = []
            mse_losses = []

        return steps, steps_to_solve, cr_losses, mse_losses, len(steps), solved, step_ratio

    def step(
        self,
        clues: Clues,
        prev: np.ndarray | None = None,
        num_steps: int = 1,
        sampling_ratio: float = 1.0,
        step_ratio: int | None = None,
    ) -> np.ndarray:
        if step_ratio is None:
            step_ratio = self.default_step_ratio
        clues_arr = np.array(clues)
        grid_shape = grid_shape_from_clues(clues_arr)
        clues_arr = normalise_clues(clues_arr)
        prev = prev if prev is not None else blank_grid(*grid_shape)
        actual_steps = (num_steps - 1) * step_ratio + 1
        all_steps = self._step(clues_arr, prev, int(actual_steps), sampling_ratio=sampling_ratio)
        indices = np.linspace(0, len(all_steps) - 1, num_steps, dtype=int)
        return np.array([all_steps[i] for i in indices])

    @abstractmethod
    def _step(
        self, clues: np.ndarray, prev: np.ndarray, num_steps: int, sampling_ratio: float
    ) -> np.ndarray:
        pass
