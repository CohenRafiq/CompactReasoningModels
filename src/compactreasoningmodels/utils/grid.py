"""Shared utilities for extracting nonogram clues from grids via run-length encoding."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def get_line_clues(line: Iterable[int], K: int | None = None):
    """Derive run-length clues from a single binary line.

    Args:
        line: A 1-D iterable of 0/1 values (list, np.ndarray, torch.Tensor, etc.).
        K: If given, zero-pad the output to this width and return a numpy array
           (or torch.Tensor if the input is a tensor).

    Returns:
        list[int] when *K* is None (empty list for lines with no filled cells),
        otherwise a K-padded array/tensor of run lengths.
    """
    is_tensor = isinstance(line, torch.Tensor)

    if is_tensor:
        values = line.detach().cpu().tolist()
    else:
        values = list(line)

    runs: list[int] = []
    count = 0
    for v in values:
        if v:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    if K is None:
        return runs
        return runs

    padded = runs[:K] + [0] * (K - len(runs))
    if is_tensor:
        return torch.tensor(padded, dtype=torch.float32)
    return np.array(padded, dtype=np.int32)


def derive_clues_from_grid(grid, K: int | None = None) -> tuple[list[list[int]], list[list[int]]] | tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    """Derive (row_clues, col_clues) from a 2-D binary grid.

    Args:
        grid: A 2-D structure of 0/1 values. Accepted types:
              list-of-lists, np.ndarray, or torch.Tensor.
        K: If given, zero-pad each clue line to this width.  The return
            type matches the input: ndarray → (ndarray, ndarray),
            Tensor → (Tensor, Tensor), otherwise (list, list).

    Returns:
        (row_clues, col_clues)
    """
    if isinstance(grid, torch.Tensor):
        return _derive_clues_from_grid_torch(grid, K)
    if isinstance(grid, np.ndarray):
        return _derive_clues_from_grid_np(grid, K)
    # Generic iterable path
    row_clues = [get_line_clues(row, K) for row in grid]
    col_clues = [get_line_clues(col, K) for col in zip(*grid, strict=True)]
    return row_clues, col_clues


# ── NumPy helpers ──────────────────────────────────────────────────────────


def _derive_clues_from_grid_np(grid: np.ndarray, K: int | None):
    row_clues = [get_line_clues(row, K) for row in grid]
    col_clues = [get_line_clues(col, K) for col in grid.T]
    return row_clues, col_clues


def batch_line_clues(lines: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised run-length extraction for a *batch* of binary lines.

    This is an optimised path used by the genetic-algorithm solver where
    thousands of lines must be processed each generation.

    Args:
        lines: (M, L) int array of 0/1.
        K: Maximum number of runs to capture (output zero-padded to this width).

    Returns:
        run_matrix: (M, K) run lengths, zero-padded.
        num_runs:   (M,) actual number of runs per line.
    """
    M, L = lines.shape
    pad = np.zeros((M, L + 2), dtype=lines.dtype)
    pad[:, 1:-1] = lines
    diff = pad[:, 1:] - pad[:, :-1]

    starts_mask = diff == 1
    ends_mask = diff == -1

    cum_starts = np.cumsum(starts_mask, axis=1)
    num_runs = cum_starts[:, -1]

    rows_s, cols_s = np.nonzero(starts_mask)
    _, cols_e = np.nonzero(ends_mask)

    lengths = cols_e - cols_s
    run_rank = cum_starts[rows_s, cols_s] - 1

    run_matrix = np.zeros((M, K), dtype=np.int32)
    valid = run_rank < K
    run_matrix[rows_s[valid], run_rank[valid]] = lengths[valid]

    return run_matrix, num_runs


# ── Torch helpers ──────────────────────────────────────────────────────────


def _derive_clues_from_grid_torch(grid: torch.Tensor, K: int | None):
    H, W = grid.shape
    if K is None:
        K = max(H, W)
    row_clues = torch.zeros(H, K, dtype=torch.float32)
    col_clues = torch.zeros(W, K, dtype=torch.float32)

    for i in range(H):
        r = get_line_clues(grid[i, :], K)
        row_clues[i, : r.numel()] = r
    for j in range(W):
        c = get_line_clues(grid[:, j], K)
        col_clues[j, : c.numel()] = c
    return row_clues, col_clues
