import random
from concurrent.futures import ThreadPoolExecutor
from functools import cache, lru_cache

import numpy as np

from . import BaseSolver
from ....utils.src.utils.grid import ternarise


class _StepLimitReached(Exception):
    """Internal signal used to unwind the recursion once the step budget is hit."""

    pass


class BacktrackingSearch(BaseSolver):
    default_step_ratio: int = 10

    def __init__(self, sample_size: int = 10):
        self.sample_size = sample_size
        super().__init__()

    @staticmethod
    @lru_cache(maxsize=200_000)
    def _line_feasible(vals: tuple, target: tuple) -> bool:
        """Cached across ALL calls, not just within one _check_line invocation."""
        n = len(vals)

        @cache
        def dp(i: int, j: int, run_len: int) -> bool:
            if i == n:
                if run_len > 0:
                    return j == len(target) - 1 and run_len == target[j]
                return j == len(target)

            v = vals[i]

            if v in (0, -1):
                if run_len > 0:
                    if j < len(target) and run_len == target[j]:
                        if dp(i + 1, j + 1, 0):
                            return True
                else:
                    if dp(i + 1, j, 0):
                        return True

            if v in (1, -1):
                if j < len(target) and run_len + 1 <= target[j]:
                    if dp(i + 1, j, run_len + 1):
                        return True

            return False

        return dp(0, 0, 0)

    def _check_line(self, line: np.ndarray, clue: np.ndarray, epsilon: float = 1e-2) -> bool:
        vals = tuple(line)
        target = tuple(int(x) for x in np.asarray(clue).flatten() if int(x) > 0)
        return self._line_feasible(vals, target)

    def _check_consistency(
        self, clues: np.ndarray, grid: np.ndarray, row_update: int, col_update: int
    ) -> bool:
        if not self._check_line(grid[row_update, :], clues[0][row_update]):
            return False
        if not self._check_line(grid[:, col_update], clues[1][col_update]):
            return False
        return True

    def _rec_search(
        self, clues, grid, steps, row_update=None, col_update=None, initial=False, max_steps=None
    ):
        if max_steps is not None and len(steps) >= max_steps:
            raise _StepLimitReached()

        if not initial and not self._check_consistency(clues, grid, row_update, col_update):
            return False

        if np.all(grid != -1):
            return True

        unknown_cells = np.argwhere(grid == -1)
        if len(unknown_cells) == 0:
            return True

        idx = random.randrange(len(unknown_cells))
        next_row, next_col = unknown_cells[idx]

        values = [0, 1]
        random.shuffle(values)

        for next_value in values:
            grid[next_row, next_col] = next_value
            steps.append(grid.copy().astype(np.int8))

            if self._rec_search(clues, grid, steps, next_row, next_col, max_steps=max_steps):
                return True

            grid[next_row, next_col] = -1

        return False

    def _single_run(self, clues: np.ndarray, prev: np.ndarray, num_steps: int) -> np.ndarray:
        """One independent backtracking trajectory. Returns array of shape (num_steps, H, W)."""
        known_grid = ternarise(prev, epsilon=0)
        steps = [known_grid.copy().astype(np.int8)]

        try:
            self._rec_search(clues, known_grid, steps, initial=True, max_steps=num_steps)
        except _StepLimitReached:
            pass

        while len(steps) < num_steps:
            steps.append(steps[-1].copy().astype(np.int8))

        steps = [np.where(g == -1, 0.5, g.astype(float)) for g in steps[:num_steps]]
        return np.array(steps)

    def _step(
        self, clues: np.ndarray, prev: np.ndarray, num_steps: int, sampling_ratio: float
    ) -> np.ndarray:
        num_samples = max(1, round(self.sample_size * sampling_ratio))

        # Use ThreadPoolExecutor to avoid nested ProcessPoolExecutor issues.
        # Each single_run is numpy/Python heavy and releases the GIL somewhat.
        with ThreadPoolExecutor(max_workers=num_samples) as ex:
            futures = [
                ex.submit(self._single_run, clues, prev, num_steps) for _ in range(num_samples)
            ]
            runs = [f.result() for f in futures]

        stacked = np.stack(runs, axis=0)  # (num_samples, num_steps, H, W)
        return stacked.mean(axis=0)[1:]  # (num_steps, H, W)
