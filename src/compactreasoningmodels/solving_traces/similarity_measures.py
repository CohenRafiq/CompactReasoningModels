from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity as ssim


class SimilarityMeasure(ABC):

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

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        diff = grid1.astype(np.float64) - grid2.astype(np.float64)
        return np.mean(diff ** 2, axis=(-2, -1))


class MAE(SimilarityMeasure):

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        diff = grid1.astype(np.float64) - grid2.astype(np.float64)
        return np.mean(np.abs(diff), axis=(-2, -1))


class HuberLoss(SimilarityMeasure):

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        diff = np.abs(grid1.astype(np.float64) - grid2.astype(np.float64))
        quadratic = 0.5 * diff ** 2
        linear = self.delta * (diff - 0.5 * self.delta)
        loss = np.where(diff <= self.delta, quadratic, linear)
        return np.mean(loss, axis=(-2, -1))


class MeanCosineSimilarity(SimilarityMeasure):

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

        # Avoid division by zero by adding small epsilon
        denominator = row_norm1 * row_norm2
        mask = denominator > 1e-10
        row_cos = np.zeros_like(denominator)
        row_cos[mask] = row_dot[mask] / denominator[mask]

        g1_cols = grid1.transpose(0, 2, 1).astype(np.float64)  # (B, W, H)
        g2_cols = grid2.transpose(0, 2, 1).astype(np.float64)  # (B, W, H)

        col_dot = np.sum(g1_cols * g2_cols, axis=-1)  # (B, W)
        col_norm1 = np.linalg.norm(g1_cols, axis=-1)  # (B, W)
        col_norm2 = np.linalg.norm(g2_cols, axis=-1)  # (B, W)

        denominator_col = col_norm1 * col_norm2
        mask_col = denominator_col > 1e-10
        col_cos = np.zeros_like(denominator_col)
        col_cos[mask_col] = col_dot[mask_col] / denominator_col[mask_col]

        mean_row = np.mean(row_cos, axis=-1)  # (B,)
        mean_col = np.mean(col_cos, axis=-1)  # (B,)

        result = 1.0 - (mean_row + mean_col) / 2.0  # (B,)

        return result if was_batched else float(result[0])


class PearsonCorrelation(SimilarityMeasure):

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
            result = 1.0 - np.abs(r)
        else:
            result = 1.0 - r

        return result if was_batched else float(result[0])

class SSIMSimilarity(SimilarityMeasure):
    """Structural Similarity Index Measure"""

    def __init__(self, data_range: float = 1.0):
        self.data_range = data_range

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        was_batched = grid1.ndim == 3
        if not was_batched:
            grid1 = grid1[np.newaxis, ...]
            grid2 = grid2[np.newaxis, ...]

        B = grid1.shape[0]
        similarities = []

        for i in range(B):
            sim = ssim(
                grid1[i],
                grid2[i],
                data_range=self.data_range,
                full=False
            )
            similarities.append(sim)

        result = np.array(similarities)
        return result if was_batched else float(result[0])


class WassersteinSimilarity(SimilarityMeasure):
    """Earth Mover's Distance converted to similarity"""

    def compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float | np.ndarray:
        was_batched = grid1.ndim == 3
        if not was_batched:
            grid1 = grid1[np.newaxis, ...]
            grid2 = grid2[np.newaxis, ...]

        B = grid1.shape[0]
        distances = []

        for i in range(B):
            dist = wasserstein_distance(
                grid1[i].flatten(),
                grid2[i].flatten()
            )
            distances.append(dist)

        distances = np.array(distances)

        result = np.clip(distances, 0, 1)

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
    similarities = measure(a_flat, b_flat)  # (N*M,)
    return similarities.reshape(N, M)  # (N, M)
