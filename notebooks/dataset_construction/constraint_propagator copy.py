import numpy as np
from typing import Tuple, Dict, Optional, Union, List


class ConstraintPropagator:
    """
    Forward–backward DP for nonogram row probabilities, plus grid-level
    Bayesian combination of row and column evidence.
    """

    def __init__(self):
        # Cache only the final probability vectors (cheap, O(length) each).
        self._prob_cache: Dict[
            Tuple[Tuple[int, ...], int, Optional[Tuple[int, ...]]],
            Optional[np.ndarray],
        ] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear_caches(self) -> None:
        """Drop all cached probability vectors."""
        self._prob_cache.clear()

    def solve_grid(
        self,
        rows_clues: List[Tuple[int, ...]],
        cols_clues: List[Tuple[int, ...]],
        known_grid: Optional[Union[np.ndarray, List[List[Optional[int]]]]] = None,
        return_intermediate: bool = False,
    ) -> Union[
        Optional[np.ndarray],
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]],
    ]:
        """
        Build a row-evidence grid and a column-evidence grid, then fuse them
        with the Bayesian independence formula

            P(p,q) = p*q / (p*q + (1-p)(1-q))

        Parameters
        ----------
        rows_clues : list of tuples
            Block clues for every row.  len(rows_clues) = height.
        cols_clues : list of tuples
            Block clues for every column.  len(cols_clues) = width.
        known_grid : 2-D array-like, optional
            Partially known cells.  -1 / None = unknown, 0 = empty, 1 = filled.
            Shape must be (height, width).
        return_intermediate : bool
            If True, return (combined_grid, row_grid, col_grid).

        Returns
        -------
        np.ndarray or None
            Combined probability grid of shape (height, width), or None if
            the clues are contradictory.
        """
        height = len(rows_clues)
        width = len(cols_clues)

        if known_grid is not None:
            known_grid = np.asarray(known_grid)
            if known_grid.shape != (height, width):
                raise ValueError(
                    f"known_grid shape {known_grid.shape} does not match "
                    f"({height}, {width})"
                )

        # ---- row evidence grid -----------------------------------------
        row_grid = np.empty((height, width), dtype=np.float64)
        for i, blocks in enumerate(rows_clues):
            known = None
            if known_grid is not None:
                known = tuple(int(x) for x in known_grid[i, :])
            probs = self.row_probabilities(blocks, length=width, known=known)
            if probs is None:
                return (None, None, None) if return_intermediate else None
            row_grid[i, :] = probs

        # ---- column evidence grid --------------------------------------
        col_grid = np.empty((height, width), dtype=np.float64)
        for j, blocks in enumerate(cols_clues):
            known = None
            if known_grid is not None:
                known = tuple(int(x) for x in known_grid[:, j])
            probs = self.row_probabilities(blocks, length=height, known=known)
            if probs is None:
                return (None, None, None) if return_intermediate else None
            col_grid[:, j] = probs

        # ---- Bayesian fusion -------------------------------------------
        combined = self._combine(row_grid, col_grid)

        if combined is None:
            return (None, None, None) if return_intermediate else None

        if return_intermediate:
            return combined, row_grid, col_grid
        return combined

    def row_probabilities(
        self,
        blocks: Tuple[int, ...],
        length: int = 5,
        known: Optional[Tuple[Union[int, None], ...]] = None,
    ) -> Optional[np.ndarray]:
        """
        Return the probability (0..1) that each cell is filled.
        Returns None if the constraints are contradictory.
        """
        # --- validation -------------------------------------------------
        if any(b <= 0 for b in blocks):
            raise ValueError("All block sizes must be positive integers")
        if length < 0:
            raise ValueError("length must be non-negative")
        if length > 2000:
            raise ValueError("length > 2000 is not supported")

        known_arr = self._normalize_known(known, length)

        cache_key = (blocks, length, tuple(known_arr) if known is not None else None)
        if cache_key in self._prob_cache:
            return self._prob_cache[cache_key]

        k = len(blocks)

        # Trivial impossibility: blocks don't fit even with minimum gaps.
        if blocks and sum(blocks) + len(blocks) - 1 > length:
            self._prob_cache[cache_key] = None
            return None

        # --- O(1) range helper: are there any forced-empty cells in [i, i+b)? ---
        zero_prefix = [0] * (length + 1)
        for i in range(length):
            zero_prefix[i + 1] = zero_prefix[i] + (1 if known_arr[i] == 0 else 0)

        def no_zeros(i: int, b: int) -> bool:
            return zero_prefix[i + b] - zero_prefix[i] == 0

        # ==================================================================
        # 1. FORWARD DP
        # forward[i][j] = number of ways to fill cells [0, i) with blocks [0, j),
        #                   with cell i-1 being empty (or i == 0).
        # ==================================================================
        forward = [[0] * (k + 1) for _ in range(length + 1)]
        forward[0][0] = 1

        for i in range(length + 1):
            for j in range(k + 1):
                val = forward[i][j]
                if val == 0:
                    continue

                # Option A: place an empty cell at i
                if i < length and known_arr[i] != 1:
                    forward[i + 1][j] += val

                # Option B: place block j starting at i
                if j < k:
                    b = blocks[j]
                    if i + b <= length and no_zeros(i, b):
                        if i + b == length:
                            forward[length][j + 1] += val
                        elif known_arr[i + b] != 1:
                            forward[i + b + 1][j + 1] += val

        total = forward[length][k]
        if total == 0:
            self._prob_cache[cache_key] = None
            return None

        # ==================================================================
        # 2. BACKWARD DP
        # backward[p][j] = number of ways to fill cells [p, length) with blocks [j, k),
        #                  with cell p-1 being empty (or p == 0).
        # ==================================================================
        backward = [[0] * (k + 1) for _ in range(length + 2)]
        backward[length][k] = 1

        for p in range(length, -1, -1):
            for j in range(k, -1, -1):
                # Option A: place an empty cell at p
                if p < length and known_arr[p] != 1:
                    backward[p][j] += backward[p + 1][j]

                # Option B: place block j starting at p
                if j < k:
                    b = blocks[j]
                    if p + b <= length and no_zeros(p, b):
                        if p + b == length:
                            backward[p][j] += backward[length][j + 1]
                        elif known_arr[p + b] != 1:
                            backward[p][j] += backward[p + b + 1][j + 1]

        # ==================================================================
        # 3. BUILD PER-CELL COUNTS  (difference-array / prefix-sum trick)
        #
        # For every valid placement of block j at position i:
        #   contrib = forward[i][j] * suffix_ways
        #   This placement covers cells [i, i+b). Add contrib to each of them.
        #
        # Instead of a nested loop over cells, we use a difference array so the
        # whole accumulation is O(length * k).
        # ==================================================================
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

        probs = np.array([c / total for c in counts], dtype=np.float64)
        self._prob_cache[cache_key] = probs
        return probs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalize_known(
        self, known: Optional[Tuple[Union[int, None], ...]], length: int
    ) -> List[int]:
        """Normalise every entry to -1 (unknown), 0 or 1."""
        if known is None:
            return [-1] * length
        if len(known) != length:
            raise ValueError(
                f"known tuple length ({len(known)}) must match length ({length})"
            )

        out: List[int] = []
        for v in known:
            if v is None or v == -1:
                out.append(-1)
            elif v == 0:
                out.append(0)
            elif v == 1:
                out.append(1)
            else:
                raise ValueError(
                    f"known values must be -1, None, 0, or 1, got {v!r}"
                )
        return out

    def _combine(self, p: np.ndarray, q: np.ndarray) -> Optional[np.ndarray]:
        """
        Bayesian combination of two independent probability grids.
        
        Returns None if the combination produces NaN values.
        """
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        if p.shape != q.shape:
            raise ValueError(f"Shape mismatch: {p.shape} vs {q.shape}")

        with np.errstate(divide='ignore', invalid='ignore'):
            # logit(x) = log(x / (1-x))
            logit_p = np.log(p / (1.0 - p))
            logit_q = np.log(q / (1.0 - q))
            combined_logit = logit_p + logit_q
            # sigmoid
            result = 1.0 / (1.0 + np.exp(-combined_logit))

        # Check for NaN values and return None if found
        if np.any(np.isnan(result)):
            return None
            
        return result