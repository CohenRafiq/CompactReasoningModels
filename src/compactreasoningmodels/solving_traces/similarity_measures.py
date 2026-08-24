from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim


class SimilarityMeasure(ABC):
    """Base class for grid similarity measures.

    All measures return a similarity score in [0, 1]: 1 means the grids
    are identical, 0 means maximally different. Inputs are assumed to lie
    within [0, data_range]; intermediate scores are clipped accordingly.
    """

    def __call__(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        g1 = np.asarray(grid1)
        g2 = np.asarray(grid2)

        if g1.shape != g2.shape:
            raise ValueError(
                f"Grid shapes must match. Got {g1.shape} and {g2.shape}"
            )

        if g1.ndim < 2:
            raise ValueError(
                f"Grids must be at least 2D. Got shape {g1.shape}"
            )

        return self.compute_similarity(g1, g2)

    @abstractmethod
    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        ...


class MSE(SimilarityMeasure):
    """Mean squared error mapped to similarity: ``1 - MSE / data_range**2``."""

    def __init__(self, data_range: float = 1.0):
        if data_range <= 0:
            raise ValueError(f"data_range must be positive. Got {data_range}")
        self.data_range = data_range

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        diff = grid1.astype(np.float64) - grid2.astype(np.float64)
        mse = np.mean(diff ** 2, axis=(-2, -1))
        return 1.0 - np.clip(mse / self.data_range ** 2, 0.0, 1.0)


class MAE(SimilarityMeasure):
    """Mean absolute error mapped to similarity: ``1 - MAE / data_range``."""

    def __init__(self, data_range: float = 1.0):
        if data_range <= 0:
            raise ValueError(f"data_range must be positive. Got {data_range}")
        self.data_range = data_range

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        diff = grid1.astype(np.float64) - grid2.astype(np.float64)
        mae = np.mean(np.abs(diff), axis=(-2, -1))
        return 1.0 - np.clip(mae / self.data_range, 0.0, 1.0)


class HuberLoss(SimilarityMeasure):
    """Huber loss mapped to similarity via its maximal possible value."""

    def __init__(self, delta: float = 1.0, data_range: float = 1.0):
        if delta <= 0:
            raise ValueError(f"delta must be positive. Got {delta}")
        if data_range <= 0:
            raise ValueError(f"data_range must be positive. Got {data_range}")
        self.delta = delta
        self.data_range = data_range

    def _max_loss(self) -> float:
        r, d = self.data_range, self.delta
        return 0.5 * r * r if r <= d else d * (r - 0.5 * d)

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        diff = np.abs(grid1.astype(np.float64) - grid2.astype(np.float64))
        quadratic = 0.5 * diff ** 2
        linear = self.delta * (diff - 0.5 * self.delta)
        loss = np.where(diff <= self.delta, quadratic, linear)
        mean_loss = np.mean(loss, axis=(-2, -1))
        return 1.0 - np.clip(mean_loss / self._max_loss(), 0.0, 1.0)


class MeanCosineSimilarity(SimilarityMeasure):
    """Mean of row-wise and column-wise cosine similarity, clipped to [0, 1]."""

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        was_batched = grid1.ndim == 3
        if not was_batched:
            grid1 = grid1[np.newaxis, ...]
            grid2 = grid2[np.newaxis, ...]

        g1_rows = grid1.astype(np.float64)  # (B, H, W)
        g2_rows = grid2.astype(np.float64)  # (B, H, W)

        row_dot = np.sum(g1_rows * g2_rows, axis=-1)  # (B, H)
        row_norm1 = np.linalg.norm(g1_rows, axis=-1)  # (B, H)
        row_norm2 = np.linalg.norm(g2_rows, axis=-1)  # (B, H)

        # Zero-norm rows against zero-norm rows are identical (cosine 1);
        # a zero row against a non-zero one stays at 0.
        denominator = row_norm1 * row_norm2
        mask = denominator > 1e-10
        row_cos = np.ones_like(denominator)
        row_cos[mask] = row_dot[mask] / denominator[mask]

        g1_cols = grid1.transpose(0, 2, 1).astype(np.float64)  # (B, W, H)
        g2_cols = grid2.transpose(0, 2, 1).astype(np.float64)  # (B, W, H)

        col_dot = np.sum(g1_cols * g2_cols, axis=-1)  # (B, W)
        col_norm1 = np.linalg.norm(g1_cols, axis=-1)  # (B, W)
        col_norm2 = np.linalg.norm(g2_cols, axis=-1)  # (B, W)

        denominator_col = col_norm1 * col_norm2
        mask_col = denominator_col > 1e-10
        col_cos = np.ones_like(denominator_col)
        col_cos[mask_col] = col_dot[mask_col] / denominator_col[mask_col]

        mean_row = np.mean(row_cos, axis=-1)  # (B,)
        mean_col = np.mean(col_cos, axis=-1)  # (B,)

        result = np.clip((mean_row + mean_col) / 2.0, 0.0, 1.0)  # (B,)

        return result if was_batched else float(result[0])


class PearsonCorrelation(SimilarityMeasure):
    """Pearson correlation mapped to similarity.

    With ``absolute=True`` the score is ``|r|`` (anti-correlated counts as
    similar); otherwise ``(r + 1) / 2`` maps the correlation range to
    [0, 1].
    """

    def __init__(self, absolute: bool = True):
        self.absolute = absolute

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        was_batched = grid1.ndim == 3
        if not was_batched:
            grid1 = grid1[np.newaxis, ...]
            grid2 = grid2[np.newaxis, ...]

        B = grid1.shape[0]
        g1 = grid1.reshape(B, -1).astype(np.float64)
        g2 = grid2.reshape(B, -1).astype(np.float64)

        mean1 = np.mean(g1, axis=1, keepdims=True)  # (B, 1)
        mean2 = np.mean(g2, axis=1, keepdims=True)  # (B, 1)

        std1 = np.std(g1, axis=1, ddof=0)  # (B,)
        std2 = np.std(g2, axis=1, ddof=0)  # (B,)

        g1_centered = g1 - mean1
        g2_centered = g2 - mean2

        cov = np.mean(g1_centered * g2_centered, axis=1)  # (B,)

        denom = std1 * std2
        mask = denom > 1e-10
        r = np.zeros_like(denom)
        r[mask] = cov[mask] / denom[mask]

        if self.absolute:
            result = np.abs(r)
        else:
            result = (r + 1.0) / 2.0

        return result if was_batched else float(result[0])

class SSIMSimilarity(SimilarityMeasure):
    """Structural Similarity Index Measure mapped from [-1, 1] to [0, 1]."""

    def __init__(self, data_range: float = 1.0, win_size: int | None = None):
        self.data_range = data_range
        self.win_size = win_size

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        was_batched = grid1.ndim == 3
        if not was_batched:
            grid1 = grid1[np.newaxis, ...]
            grid2 = grid2[np.newaxis, ...]

        B = grid1.shape[0]
        min_dim = min(grid1.shape[1], grid1.shape[2])
        win_size = self.win_size if self.win_size is not None else min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        similarities = []

        for i in range(B):
            sim = ssim(
                grid1[i],
                grid2[i],
                data_range=self.data_range,
                win_size=win_size,
                full=False
            )
            similarities.append(sim)

        result = (np.array(similarities) + 1.0) / 2.0
        return result if was_batched else float(result[0])


class WassersteinSimilarity(SimilarityMeasure):
    """Earth Mover's Distance mapped to similarity: ``1 - W / data_range``."""

    def __init__(self, data_range: float = 1.0):
        if data_range <= 0:
            raise ValueError(f"data_range must be positive. Got {data_range}")
        self.data_range = data_range

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        was_batched = grid1.ndim == 3
        if not was_batched:
            grid1 = grid1[np.newaxis, ...]
            grid2 = grid2[np.newaxis, ...]

        B = grid1.shape[0]
        distances: list[float] = []

        for i in range(B):
            dist = wasserstein_distance(
                grid1[i].flatten(),
                grid2[i].flatten()
            )
            distances.append(dist)

        result = 1.0 - np.clip(np.asarray(distances) / self.data_range, 0.0, 1.0)

        return result if was_batched else float(result[0])

def compare_batches(
    grids_a: np.ndarray,
    grids_b: np.ndarray,
    measure: SimilarityMeasure
) -> np.ndarray:
    a = np.asarray(grids_a)
    b = np.asarray(grids_b)

    if a.ndim == 2:
        a = a[np.newaxis, ...]
    if b.ndim == 2:
        b = b[np.newaxis, ...]

    N, M = a.shape[0], b.shape[0]
    a_exp = np.repeat(a[:, np.newaxis, ...], M, axis=1)  # (N, M, H, W)
    b_exp = np.repeat(b[np.newaxis, ...], N, axis=0)     # (N, M, H, W)

    a_flat = a_exp.reshape(N * M, *a.shape[1:])
    b_flat = b_exp.reshape(N * M, *b.shape[1:])
    similarities = np.asarray(measure(a_flat, b_flat))  # (N*M,)
    return similarities.reshape(N, M)  # (N, M)
