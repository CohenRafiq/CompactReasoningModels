from abc import ABC, abstractmethod
import numpy as np

class StepAlignment(ABC):
    @abstractmethod
    def align(self, trace1: np.ndarray, trace2: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        """
        Inputs:
        trace1: (B, C, steps_A)
        trace2: (B, C, steps_B)

        Output:
        aligned_trace1: (B, C, K)
        aligned_trace2: (B, C, K)
        """
        pass

class MeanPoolingAlignment(StepAlignment):
    def align(self, trace1: np.ndarray, trace2: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align two solving traces by mean pooling along the steps dimension.
        """
        aligned_trace1 = np.mean(trace1, axis=-1, keepdims=True)
        aligned_trace2 = np.mean(trace2, axis=-1, keepdims=True)
        return aligned_trace1, aligned_trace2

class LinearGradientPooling(StepAlignment):
    def align(self, trace1: np.ndarray, trace2: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        """
        Align two solving traces by applying a linear gradient pooling along the steps dimension.
        """
        gradients1 = np.apply_along_axis(lambda x1: np.gradient(x1), -1, trace1)
        gradients2 = np.apply_along_axis(lambda x2: np.gradient(x2), -1, trace2)

        return gradients1, gradients2

"""
- Dynamic Time Warping
- Autoencoder
- PCA + Procrustes
"""

class PCAProcrustesAlignment(StepAlignment):
    """
    Align two solving traces of possibly different step-lengths by:
      1) resampling both traces to a common number of steps K along the
         step dimension (linear interpolation),
      2) reducing channel dimensionality with PCA (optional, per-batch,
         fit jointly on both traces so they land in the same subspace),
      3) solving an orthogonal Procrustes problem (Kabsch algorithm) to
         find the rotation that best aligns trace2's step-trajectory onto
         trace1's, and applying that rotation to trace2.

    This corrects for step-count mismatch (via resampling) and for
    arbitrary rotations/reflections of the underlying representation
    space (via PCA + Procrustes), which is useful when comparing traces
    that live in different but related coordinate systems.
    """

    def __init__(self, K: int | None = None, n_components: int | None = None,
                 allow_reflection: bool = False, allow_scaling: bool = False):
        """
        Args:
            K: target number of steps to resample both traces to.
               Defaults to min(steps_A, steps_B).
            n_components: number of PCA components to keep along the
               channel dimension. Defaults to C (no dimensionality
               reduction, PCA is only used to whiten/orient the space).
            allow_reflection: if False (default), the Procrustes solution
               is constrained to be a proper rotation (det(R) = 1),
               matching the classic Kabsch algorithm.
            allow_scaling: if True, also solve for an isotropic scale
               factor between the two trajectories (Procrustes with
               scaling), in addition to the rotation.
        """
        self.K = K
        self.n_components = n_components
        self.allow_reflection = allow_reflection
        self.allow_scaling = allow_scaling

    # ---------- helpers ----------

    @staticmethod
    def _resample(trace: np.ndarray, K: int) -> np.ndarray:
        """Linearly resample (B, C, steps) -> (B, C, K) along the last axis."""
        B, C, steps = trace.shape
        if steps == K:
            return trace.copy()
        x_old = np.linspace(0.0, 1.0, steps)
        x_new = np.linspace(0.0, 1.0, K)
        out = np.empty((B, C, K), dtype=trace.dtype)
        for b in range(B):
            for c in range(C):
                out[b, c] = np.interp(x_new, x_old, trace[b, c])
        return out

    @staticmethod
    def _pca_fit_transform(X: np.ndarray, Y: np.ndarray, n_components: int
                            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Jointly fit a PCA basis on the concatenation of X and Y (each (K, C))
        and project both onto the top n_components. Centers on the joint mean.
        """
        joint = np.concatenate([X, Y], axis=0)  # (2K, C)
        mean = joint.mean(axis=0, keepdims=True)
        centered = joint - mean
        # SVD-based PCA
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:n_components]  # (n_components, C)
        Xc = (X - mean) @ components.T
        Yc = (Y - mean) @ components.T
        return Xc, Yc

    @staticmethod
    def _orthogonal_procrustes(X: np.ndarray, Y: np.ndarray,
                                allow_reflection: bool, allow_scaling: bool
                                ) -> tuple[np.ndarray, float]:
        """
        Solve for rotation R (and optional scale s) minimizing
        || X - s * Y @ R ||_F, i.e. align Y onto X.
        X, Y: (K, D)
        Returns R (D, D), s (float, 1.0 if allow_scaling=False).
        """
        # Center both sets (translation is handled separately in align()),
        # but we re-center here defensively for numerical stability.
        Xc = X - X.mean(axis=0, keepdims=True)
        Yc = Y - Y.mean(axis=0, keepdims=True)

        M = Yc.T @ Xc  # (D, D)
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt

        if not allow_reflection and np.linalg.det(R) < 0:
            U[:, -1] *= -1
            S[-1] *= -1
            R = U @ Vt

        s = 1.0
        if allow_scaling:
            var_Y = np.sum(Yc ** 2)
            s = S.sum() / var_Y if var_Y > 1e-12 else 1.0

        return R, s

    # ---------- main API ----------

    def align(self, trace1: np.ndarray, trace2: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
        B, C, steps_A = trace1.shape
        _, _, steps_B = trace2.shape
        K = self.K if self.K is not None else min(steps_A, steps_B)
        n_components = self.n_components if self.n_components is not None else C

        # 1) Resample both traces onto a common step grid of length K.
        t1_resampled = self._resample(trace1, K)  # (B, C, K)
        t2_resampled = self._resample(trace2, K)  # (B, C, K)

        aligned1 = np.empty((B, n_components, K), dtype=float)
        aligned2 = np.empty((B, n_components, K), dtype=float)

        for b in range(B):
            X = t1_resampled[b].T  # (K, C) -- steps as points
            Y = t2_resampled[b].T  # (K, C)

            # 2) Joint PCA to a shared, comparable subspace.
            Xp, Yp = self._pca_fit_transform(X, Y, n_components)  # (K, n_components)

            # 3) Orthogonal Procrustes: rotate Y onto X.
            R, s = self._orthogonal_procrustes(
                Xp, Yp, self.allow_reflection, self.allow_scaling
            )

            X_mean = Xp.mean(axis=0, keepdims=True)
            Y_mean = Yp.mean(axis=0, keepdims=True)

            Xc = Xp - X_mean
            Y_aligned = s * (Yp - Y_mean) @ R + X_mean  # bring Y into X's frame

            aligned1[b] = (Xc + X_mean).T  # (n_components, K)
            aligned2[b] = Y_aligned.T      # (n_components, K)

        return aligned1, aligned2
        