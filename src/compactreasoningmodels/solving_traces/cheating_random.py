from compactreasoningmodels.solving_traces.base import SolvingTrace

import numpy as np

class CheatingRandom(SolvingTrace):

    def __init__(self, clues: np.ndarray, grid_shape: tuple[int, int], initial_grid: np.ndarray | None = None):
        super().__init__(clues, grid_shape, initial_grid)
        self.correct = None

    def set_correct(self, grid: np.ndarray) -> np.ndarray:
        self.correct = grid.copy()

    def step(self, grid: np.ndarray) -> np.ndarray:
        if self.correct is None:
            raise ValueError("correct grid has not been set. Call set_correct() before step().")

        new_grid = grid.copy()
        height, width = new_grid.shape
        # pick a random square that is not yet filled correctly
        while True:
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            if new_grid[row, col] != self.correct[row, col]:
                break
        new_grid[row, col] = self.correct[row, col]
        return new_grid

    def heatmap_step(self, grid: np.ndarray) -> np.ndarray:
        if self.correct is None:
            raise ValueError("correct grid has not been set. Call set_correct() before heatmap_step().")

        unfilled_mask = grid != self.correct
        number_of_unfilled = np.sum(unfilled_mask)
        change_mask = np.where(unfilled_mask, 1 / number_of_unfilled, -1 / number_of_unfilled) * unfilled_mask
        return grid.copy() + change_mask
