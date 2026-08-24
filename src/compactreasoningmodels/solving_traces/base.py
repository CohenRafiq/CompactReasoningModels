from abc import ABC, abstractmethod

import numpy as np


class SolvingTrace(ABC):

    def __init__(self, clues: np.ndarray, grid_shape: tuple[int, int],
                 initial_grid: np.ndarray | None = None):
        super().__init__()
        self.clues = clues
        self.grid_size = grid_shape
        self.initial_grid = (
            initial_grid if initial_grid is not None else self._blank_grid(*grid_shape)
        )
        self.traces = [self.initial_grid]

    @staticmethod
    def _blank_grid(width: int, height: int) -> np.ndarray:
        return np.full((height, width), 0.5, dtype=float)

    @staticmethod
    def _is_solved(clues: np.ndarray, grid: np.ndarray) -> bool:
        num_rows = len(grid)
        num_cols = len(grid[0]) if num_rows > 0 else 0

        # Check rows
        for i in range(num_rows):
            if not SolvingTrace._check_clue(grid[i], clues[i]):
                return False

        # Check columns
        for j in range(num_cols):
            col = grid[:, j]  # Use numpy slicing to get column as array
            if not SolvingTrace._check_clue(col, clues[num_rows + j]):
                return False

        return True

    @staticmethod
    def _check_clue(line: np.ndarray, clue: np.ndarray, epsilon: float = 1e-2) -> bool:
        # Round if epsilon off 0 or 1
        line_rounded = [1 if cell > 1 - epsilon else 0 if cell < epsilon else -1 for cell in line]

        if -1 in line_rounded:
            return False

        runs = []
        current_run = 0
        for cell in line_rounded:
            if cell == 1:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)

        return runs == list(clue)

    def try_solve(self, max_steps: int = 1000, alpha: float = 1.0) -> tuple[bool, list[np.ndarray]]:
        solved = False
        for _ in range(max_steps):
            grid_step = self.heatmap_step(self.traces[-1])
            new_grid = alpha * grid_step + (1 - alpha) * self.traces[-1]
            self.traces.append(new_grid)
            if self._is_solved(self.clues, new_grid):
                solved = True
                break
        return solved, self.traces

    @abstractmethod
    def heatmap_step(self, grid: np.ndarray) -> np.ndarray:
        ...
