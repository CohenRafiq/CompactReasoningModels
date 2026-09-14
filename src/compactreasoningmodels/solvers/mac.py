import random
from functools import lru_cache

import numpy as np

from compactreasoningmodels.solvers import BaseSolver
from compactreasoningmodels.utils.grid import ternarise, unpad_clue


class MAC(BaseSolver):

    default_step_ratio: int = 2

    def __init__(self, sampling_ratio_modifier: float = 0.5):
        self.sampling_ratio_modifier = sampling_ratio_modifier
        super().__init__()

    @staticmethod
    @lru_cache(maxsize=10_000)
    def _row_probabilities_cached(blocks: tuple, length: int, known: tuple) -> np.ndarray | None:
        """Cached row probability computation."""
        known_arr = list(known) if known is not None else [-1] * length
        k = len(blocks)

        if blocks and sum(blocks) + len(blocks) - 1 > length:
            return None

        zero_prefix = [0] * (length + 1)
        for i in range(length):
            zero_prefix[i + 1] = zero_prefix[i] + (1 if known_arr[i] == 0 else 0)

        def no_zeros(i: int, b: int) -> bool:
            return zero_prefix[i + b] - zero_prefix[i] == 0

        forward = [[0] * (k + 1) for _ in range(length + 1)]
        forward[0][0] = 1
        for i in range(length + 1):
            for j in range(k + 1):
                val = forward[i][j]
                if val == 0:
                    continue
                if i < length and known_arr[i] != 1:
                    forward[i + 1][j] += val
                if j < k:
                    b = blocks[j]
                    if i + b <= length and no_zeros(i, b):
                        if i + b == length:
                            forward[length][j + 1] += val
                        elif known_arr[i + b] != 1:
                            forward[i + b + 1][j + 1] += val

        total = forward[length][k]
        if total == 0:
            return None

        backward = [[0] * (k + 1) for _ in range(length + 2)]
        backward[length][k] = 1
        for p in range(length, -1, -1):
            for j in range(k, -1, -1):
                if p < length and known_arr[p] != 1:
                    backward[p][j] += backward[p + 1][j]
                if j < k:
                    b = blocks[j]
                    if p + b <= length and no_zeros(p, b):
                        if p + b == length:
                            backward[p][j] += backward[length][j + 1]
                        elif known_arr[p + b] != 1:
                            backward[p][j] += backward[p + b + 1][j + 1]

        diff = [0] * (length + 1)
        for j in range(k):
            b = blocks[j]
            for i in range(length - b + 1):
                if forward[i][j] == 0 or not no_zeros(i, b):
                    continue
                if i + b < length and known_arr[i + b] == 1:
                    continue
                suffix = backward[length][j + 1] if i + b == length else backward[i + b + 1][j + 1]
                if suffix == 0:
                    continue
                contrib = forward[i][j] * suffix
                diff[i] += contrib
                diff[i + b] -= contrib

        counts = [0] * length
        counts[0] = diff[0]
        for c in range(1, length):
            counts[c] = counts[c - 1] + diff[c]

        return np.array([c / total for c in counts], dtype=np.float64)

    def _row_probabilities(
        self, blocks: tuple[int, ...], length: int,
        known: tuple[int, ...] | None = None
        ) -> np.ndarray | None:
        """Wrapper that converts to hashable types for caching."""
        blocks_tuple = tuple(int(b) for b in blocks)
        known_tuple = tuple(int(x) for x in known) if known is not None else tuple([-1] * length)
        return self._row_probabilities_cached(blocks_tuple, length, known_tuple)

    @staticmethod
    @lru_cache(maxsize=10_000)
    def _combine_cached(p_tuple: tuple, q_tuple: tuple) -> tuple | None:
        """Cached combine operation."""
        p = np.array(p_tuple, dtype=np.float64)
        q = np.array(q_tuple, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            logit = np.log(p / (1.0 - p)) + np.log(q / (1.0 - q))
            result = 1.0 / (1.0 + np.exp(-logit))
        if np.any(np.isnan(result)):
            return None
        return tuple(result.flatten().tolist())

    def _combine(self, p: np.ndarray, q: np.ndarray) -> np.ndarray | None:
        """Wrapper that converts to hashable types for caching."""
        if p is None or q is None:
            return None
        p_tuple = tuple(p.flatten().tolist())
        q_tuple = tuple(q.flatten().tolist())
        result = self._combine_cached(p_tuple, q_tuple)
        if result is None:
            return None
        return np.array(result).reshape(p.shape)

    def _loop_directions(
        self, grid: np.ndarray, direction_clues, known_grid: np.ndarray, sampling_ratio: float
    ) -> np.ndarray | None:
        output_grid = grid.copy()
        for i in range(grid.shape[0]):
            blocks = unpad_clue(direction_clues[i])
            known = tuple(int(x) for x in known_grid[i, :])
            probs = self._row_probabilities(blocks, length=grid.shape[1], known=known)
            if probs is None:
                return None
            if random.random() <= sampling_ratio * self.sampling_ratio_modifier:
                output_grid[i, :] = probs
        return output_grid

    def _step(self, clues, initial_grid: np.ndarray, num_steps: int, sampling_ratio: float = 1.0) -> np.ndarray:
        row_belief = initial_grid.copy()
        col_belief = initial_grid.copy()

        combined0 = self._combine(row_belief, col_belief)
        if combined0 is None:
            combined0 = initial_grid.copy()
        steps = [combined0]

        for _ in range(num_steps):
            known_grid = ternarise(steps[-1], epsilon=0)

            row_grid = self._loop_directions(row_belief, clues[0], known_grid, sampling_ratio)
            if row_grid is None:
                print("Inconsistent clues or grid state encountered. DIRECTION: ROWS")
                return np.stack(steps, axis=0)

            col_grid_T = self._loop_directions(col_belief.T, clues[1], known_grid.T, sampling_ratio)
            if col_grid_T is None:
                print("Inconsistent clues or grid state encountered. DIRECTION: COLUMNS")
                return np.stack(steps, axis=0)
            col_grid = col_grid_T.T

            combined = self._combine(row_grid, col_grid)
            if combined is None:
                print("Inconsistent clues or grid state encountered. DIRECTION: COMBINATION")
                return np.stack(steps, axis=0)

            row_belief = row_grid
            col_belief = col_grid
            steps.append(combined)
        steps = steps[1:]
        return np.stack(steps, axis=0)