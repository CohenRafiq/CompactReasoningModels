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
        if np.array_equal(grid, self.correct):
            return grid.copy()
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

    def heatmap_step(self, grid):
        unfilled_mask = grid != self.correct
        n_unfilled = np.sum(unfilled_mask)
        if n_unfilled == 0:
            return grid.copy()
        
        # Expected change for each cell
        expected_change = np.zeros_like(grid, dtype=float)
        expected_change[unfilled_mask] = (self.correct[unfilled_mask] - grid[unfilled_mask]) / n_unfilled
        
        return grid + expected_change
