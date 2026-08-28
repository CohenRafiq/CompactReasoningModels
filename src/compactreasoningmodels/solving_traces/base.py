from abc import ABC, abstractmethod

import numpy as np
import torch


class SolvingTrace(ABC):

    def __init__(self, clues: np.ndarray | torch.Tensor,
                 grid_shape: tuple[int, int],
                 initial_grid: np.ndarray | torch.Tensor | None = None):
        super().__init__()
        self.clues = self._normalize_clues(clues, grid_shape)
        self.grid_size = grid_shape
        self.initial_grid = (
            self._as_numpy(initial_grid) if initial_grid is not None
            else self._blank_grid(*grid_shape)
        )
        self.traces = [self.initial_grid]

    @staticmethod
    def _blank_grid(rows: int, cols: int) -> np.ndarray:
        return np.full((rows, cols), 0.5, dtype=float)

    @staticmethod
    def _as_numpy(data: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @staticmethod
    def _normalize_clues(clues: np.ndarray | torch.Tensor,
                         grid_shape: tuple[int, int]) -> np.ndarray:
        arr = SolvingTrace._as_numpy(clues)
        rows, cols = grid_shape
        k_row = (cols + 1) // 2
        k_col = (rows + 1) // 2
        flat_len = rows * k_row + cols * k_col

        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr.reshape(-1)

        if arr.ndim == 1:
            if arr.shape[0] != flat_len:
                raise ValueError(
                    f"Expected {flat_len} flattened clue values for a {rows}x{cols} "
                    f"grid ({rows}x{k_row} row clues + {cols}x{k_col} column clues), "
                    f"got {arr.shape[0]}"
                )
            if k_row != k_col:
                raise ValueError(
                    "Flattened clues are only supported for square grids; pass "
                    "row/column clues as a (2, ...) array instead"
                )
            row_clues = arr[: rows * k_row].reshape(rows, k_row)
            col_clues = arr[rows * k_row:].reshape(cols, k_col)
            return np.stack([row_clues, col_clues])

        if arr.ndim == 3 and arr.shape[0] == 2:
            return arr

        raise ValueError(
            f"clues must be a tensor/array of shape ({flat_len},) or "
            f"(1, {flat_len}) with the flat [row clues; column clues] dataloader "
            f"layout, or a stacked array of shape (2, H, K). Got shape {arr.shape}"
        )

    @staticmethod
    def _is_solved(clues: np.ndarray, grid: np.ndarray) -> bool:
        if clues.ndim == 3:
            row_clues, col_clues = clues[0], clues[1]
        else:
            num_rows = len(grid)
            row_clues, col_clues = clues[:num_rows], clues[num_rows:]

        # Check rows
        for i in range(len(grid)):
            if not SolvingTrace._check_clue(grid[i], row_clues[i]):
                return False

        # Check columns
        for j in range(grid.shape[1]):
            col = grid[:, j]  # Use numpy slicing to get column as array
            if not SolvingTrace._check_clue(col, col_clues[j]):
                return False

        return True

    @staticmethod
    def _check_clue(line: np.ndarray, clue: np.ndarray, epsilon: float = 1e-2) -> bool:
        from compactreasoningmodels.utils.grid import get_line_clues

        line_rounded = [1 if cell > 1 - epsilon else 0 if cell < epsilon else -1 for cell in line]

        if -1 in line_rounded:
            return False

        return get_line_clues(line_rounded) == list(clue)

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

    def heatmap(self, num_steps: int = 1) -> np.ndarray:
        grids = [self.initial_grid]
        for _ in range(num_steps):
            grid_step = self.heatmap_step(grids[-1])
            grids.append(grid_step)
        return grids[-1]

    @abstractmethod
    def heatmap_step(self, grid: np.ndarray) -> np.ndarray:
        ...
