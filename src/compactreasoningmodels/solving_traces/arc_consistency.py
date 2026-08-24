from compactreasoningmodels.solving_traces.base import SolvingTrace

import numpy as np


class ArcConsistency(SolvingTrace):
    def _row_probabilities(
        self, blocks: tuple[int, ...], length: int, known: tuple[int, ...] | None = None
    ) -> np.ndarray | None:
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

    @staticmethod
    def _known_from_grid(grid: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
        known = np.full(grid.shape, -1, dtype=int)
        known[grid <= epsilon] = 0
        known[grid >= 1 - epsilon] = 1
        return known

    @staticmethod
    def _combine(p: np.ndarray, q: np.ndarray) -> np.ndarray | None:
        with np.errstate(divide="ignore", invalid="ignore"):
            logit = np.log(p / (1.0 - p)) + np.log(q / (1.0 - q))
            result = 1.0 / (1.0 + np.exp(-logit))
        return None if np.any(np.isnan(result)) else result

    @staticmethod
    def _to_blocks(clue) -> tuple[int, ...]:
        if hasattr(clue, "tolist"):
            clue = clue.tolist()
        flat: list[int] = []
        stack = [clue]
        while stack:
            item = stack.pop(0)
            if isinstance(item, (list, tuple)):
                stack = list(item) + stack
            else:
                flat.append(int(item))
        return tuple(b for b in flat if b > 0)

    def _loop_directions(self, grid: np.ndarray, direction_clues, known_grid: np.ndarray) -> np.ndarray | None:
        output_grid = np.empty_like(grid, dtype=np.float64)
        for i in range(grid.shape[0]):
            blocks = self._to_blocks(direction_clues[i])
            known = tuple(int(x) for x in known_grid[i, :])
            probs = self._row_probabilities(blocks, length=grid.shape[1], known=known)
            if probs is None:
                return None
            output_grid[i, :] = probs
        return output_grid

    def heatmap_step(self, grid: np.ndarray) -> np.ndarray:
        known_grid = self._known_from_grid(grid)

        row_grid = self._loop_directions(grid, self.clues[0], known_grid)
        if row_grid is None:
            return grid

        col_grid = self._loop_directions(grid.T, self.clues[1], known_grid.T)
        if col_grid is None:
            return grid

        combined = self._combine(row_grid, col_grid.T)
        return grid if combined is None else combined


if __name__ == "__main__":
    clues = np.array([[
        [0, 0, 0],
        [3, 0, 0],
        [1, 1, 0],
        [3, 0, 0],
        [0, 0, 0]
    ],[
        [0, 0, 0],
        [3, 0, 0],
        [1, 1, 0],
        [3, 0, 0],
        [0, 0, 0]
    ]])
    solver = ArcConsistency(clues, (5, 5))
    grid = np.full((5, 5), 0.5)
    for _ in range(5):
        grid = solver.heatmap_step(grid)
        print(grid)