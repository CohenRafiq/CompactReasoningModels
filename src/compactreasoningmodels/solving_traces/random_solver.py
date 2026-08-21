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

    def heatmap_step(self, grid):
        height, width = grid.shape
        # Each cell has 1/(w*h) chance of being selected
        # If selected, expected new value is 0.5
        # So expected change = (1/(w*h)) * (0.5 - grid[i,j])
        selection_prob = 1.0 / (height * width)
        expected_change = selection_prob * (0.5 - grid)
        return grid + expected_change
