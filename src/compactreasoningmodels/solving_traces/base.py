from abc import ABC, abstractmethod

import numpy as np


class SolvingTrace(ABC):

    def __init__(self, clues: np.ndarray, grid_shape: tuple[int, int], initial_grid: np.ndarray | None = None):
        super().__init__()
        self.clues = clues
        self.grid_size = grid_shape
        self.initial_grid = initial_grid if initial_grid is not None else self._blank_grid(*grid_shape)
        self.traces = [self.initial_grid]

    @staticmethod
    def _blank_grid(width: int, height: int) -> np.ndarray:
        return np.full((height, width), 0.5, dtype=float)

    @staticmethod
    def _is_solved(clues: np.ndarray, grid: np.ndarray) -> bool:
        for i, row in enumerate(grid):
            if not SolvingTrace._check_clue(row, clues[i]):
                return False
        for j, col in enumerate(zip(*grid)):
            if not SolvingTrace._check_clue(col, clues[len(grid) + j]):
                return False
        return True

    @staticmethod
    def _check_clue(line: np.ndarray, clue: np.ndarray, epsilon: float = 1e-2) -> bool:
        # Round if epsilon off 0 or 1
        line = [1 if cell > 1 - epsilon else 0 if cell < epsilon else -1 for cell in line]
        if -1 in line:
            return False

        runs = []
        current_run = 0
        for cell in line:
            if cell == 1:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)

        return runs == list(clue)

    def try_solve(self, max_steps: int = 1000) -> tuple[bool, list[np.ndarray]]:
        solved = False
        for _ in range(max_steps):
            new_grid = self.step(self.traces[-1])
            self.traces.append(new_grid)
            if self._is_solved(self.clues, new_grid):
                solved = True
                break
        return solved, self.traces

    @abstractmethod
    def step(self, grid: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def heatmap_step(self, grid: np.ndarray) -> np.ndarray:
        ...