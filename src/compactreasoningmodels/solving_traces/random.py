from compactreasoningmodels.solving_traces.base import SolvingTrace

import random
import numpy as np

class RandomSolver(SolvingTrace):

    def step(self, grid: np.ndarray) -> np.ndarray:

        new_grid = grid.copy()
        height, width = new_grid.shape
        row = np.random.randint(0, height)
        col = np.random.randint(0, width)
        new_grid[row, col] = random.randint(0, 1)
        return new_grid

    def heatmap_step(self, grid: np.ndarray) -> np.ndarray:
        width, height = grid.shape
        trajectory = np.full(grid.shape, 0.5, dtype=float) - grid.copy()
        return grid.copy() + trajectory / (width * height)
