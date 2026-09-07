from collections import OrderedDict
from typing import cast

import numpy as np

_MISS = object()


class ConstraintPropagator:
    """
    Forward-backward DP for nonogram row probabilities, plus grid-level
    Bayesian combination of row and column evidence.

    Two-tier cache:
      - Tier 1 ("prior"): key = (blocks, length), only used when `known`
        is entirely unknown (-1 everywhere). High hit rate, since it's
        independent of any particular puzzle's fill state — can be shared
        across workers/puzzles. Unbounded (clue-space is small).
      - Tier 2 ("partial"): key = (blocks, length, known_tuple), used once
        any cell is known. Low hit rate by nature (fill patterns are
        mostly unique per puzzle/step), kept local per-worker and bounded
        with an LRU eviction policy so it can't leak memory.
    """

    def __init__(
        self,
        prior_cache: dict[tuple[tuple[int, ...], int], np.ndarray | None] | None = None,
        local_cache_maxsize: int = 5000,
    ):
        self._prior_cache = prior_cache if prior_cache is not None else {}
        self._prob_cache: OrderedDict[tuple, np.ndarray | None] = OrderedDict()
        self._local_cache_maxsize = local_cache_maxsize

    def clear_caches(self, prior: bool = False, local: bool = True) -> None:
        if local:
            self._prob_cache.clear()
        if prior:
            self._prior_cache.clear()

    def prior_cache_snapshot(self) -> dict[tuple[tuple[int, ...], int], np.ndarray | None]:
        return dict(self._prior_cache)

    def solve_grid(
        self,
        rows_clues: list[tuple[int, ...]],
        cols_clues: list[tuple[int, ...]],
        known_grid: np.ndarray | list[list[int | None]] | None = None,
        return_intermediate: bool = False,
        prev_row_grid: np.ndarray | None = None,
        prev_col_grid: np.ndarray | None = None,
        dirty_rows: set[int] | None = None,
        dirty_cols: set[int] | None = None,
    ) -> np.ndarray | None | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        height = len(rows_clues)
        width = len(cols_clues)

        if known_grid is not None:
            known_grid = np.asarray(known_grid)
            if known_grid.shape != (height, width):
                raise ValueError(
                    f"known_grid shape {known_grid.shape} does not match ({height}, {width})"
                )

        if prev_row_grid is not None and dirty_rows is not None:
            row_grid = prev_row_grid.copy()
            rows_to_compute: range | set[int] = dirty_rows
        else:
            row_grid = np.empty((height, width), dtype=np.float64)
            rows_to_compute = range(height)

        for i in rows_to_compute:
            blocks = rows_clues[i]
            known = None
            if known_grid is not None:
                known = tuple(int(x) for x in known_grid[i, :])
            probs = self.row_probabilities(blocks, length=width, known=known)
            if probs is None:
                return (None, None, None) if return_intermediate else None
            row_grid[i, :] = probs

        if prev_col_grid is not None and dirty_cols is not None:
            col_grid = prev_col_grid.copy()
            cols_to_compute: range | set[int] = dirty_cols
        else:
            col_grid = np.empty((height, width), dtype=np.float64)
            cols_to_compute = range(width)

        for j in cols_to_compute:
            blocks = cols_clues[j]
            known = None
            if known_grid is not None:
                known = tuple(int(x) for x in known_grid[:, j])
            probs = self.row_probabilities(blocks, length=height, known=known)
            if probs is None:
                return (None, None, None) if return_intermediate else None
            col_grid[:, j] = probs

        combined = self._combine(row_grid, col_grid)

        if combined is None:
            return (None, None, None) if return_intermediate else None

        if return_intermediate:
            return combined, row_grid, col_grid
        return combined

    def row_probabilities(
        self,
        blocks: tuple[int, ...],
        length: int = 5,
        known: tuple[int | None, ...] | None = None,
    ) -> np.ndarray | None:
        if any(b <= 0 for b in blocks):
            raise ValueError("All block sizes must be positive integers")
        if length < 0:
            raise ValueError("length must be non-negative")
        if length > 2000:
            raise ValueError("length > 2000 is not supported")

        known_arr = self._normalize_known(known, length)
        is_fully_unknown = not any(v != -1 for v in known_arr)

        if is_fully_unknown:
            prior_key = (blocks, length)
            if prior_key in self._prior_cache:
                return self._prior_cache[prior_key]
            probs = self._compute_probabilities(blocks, length, known_arr)
            self._prior_cache[prior_key] = probs
            return probs

        local_key = (blocks, length, tuple(known_arr))
        cached = cast(np.ndarray | None, self._prob_cache.get(local_key, _MISS))
        if cached is not _MISS:
            self._prob_cache.move_to_end(local_key)
            return cached

        probs = self._compute_probabilities(blocks, length, known_arr)
        self._prob_cache[local_key] = probs
        self._prob_cache.move_to_end(local_key)
        if len(self._prob_cache) > self._local_cache_maxsize:
            self._prob_cache.popitem(last=False)
        return probs

    def _compute_probabilities(
        self, blocks: tuple[int, ...], length: int, known_arr: list[int]
    ) -> np.ndarray | None:
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
                if forward[i][j] == 0:
                    continue
                if not no_zeros(i, b):
                    continue
                if i + b < length and known_arr[i + b] == 1:
                    continue
                if i + b == length:
                    suffix = backward[length][j + 1]
                else:
                    suffix = backward[i + b + 1][j + 1]
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

    def _normalize_known(self, known: tuple[int | None, ...] | None, length: int) -> list[int]:
        if known is None:
            return [-1] * length
        if len(known) != length:
            raise ValueError(f"known tuple length ({len(known)}) must match length ({length})")
        out: list[int] = []
        for v in known:
            if v is None or v == -1:
                out.append(-1)
            elif v == 0:
                out.append(0)
            elif v == 1:
                out.append(1)
            else:
                raise ValueError(f"known values must be -1, None, 0, or 1, got {v!r}")
        return out

    def _combine(self, p: np.ndarray, q: np.ndarray) -> np.ndarray | None:
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if p.shape != q.shape:
            raise ValueError(f"Shape mismatch: {p.shape} vs {q.shape}")

        with np.errstate(divide="ignore", invalid="ignore"):
            logit_p = np.log(p / (1.0 - p))
            logit_q = np.log(q / (1.0 - q))
            combined_logit = logit_p + logit_q
            result = 1.0 / (1.0 + np.exp(-combined_logit))

        if np.any(np.isnan(result)):
            return None

        return result
