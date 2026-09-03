import numpy as np
import torch

from compactreasoningmodels.solvers import BaseSolver
from compactreasoningmodels.losses.clue_reconstruction import ClueReconstructionLoss


class BaseGradientDescentSolver(BaseSolver):

    def __init__(self, **kwargs):
        self.loss_fn = ClueReconstructionLoss(reduction="none")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(**kwargs)


    def _sample_grid(self, n: int, sampling_ratio: float) -> torch.Tensor:
        """Continuous or binary weight in [0, 1] used by GlobalMixin to
        blend between the previous value and the freshly-optimised value."""
        raise NotImplementedError

    def _sample_mask(self, n: int, sampling_ratio: float) -> torch.Tensor:
        """Hard {0, 1} selection mask used by coordinate-wise mixins
        (GaussSeidel, Jacobi) to decide which variables are unfrozen."""
        raise NotImplementedError

    def _reset_optimizer_state(self, n: int) -> None:
        """Called once at the start of each `_step` call."""
        raise NotImplementedError

    def _compute_update(self, grad: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        """Return the parameter delta (same shape as `grad` / `idxs`).
        Called inside a `torch.no_grad()` block by the update mixins."""
        raise NotImplementedError

    def _optimise(self, prev: torch.Tensor, n: int, sampling_ratio: float,
        tensor_clues: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _step(self, clues: np.ndarray, prev: np.ndarray,
        num_steps: int, sampling_ratio: float = 1.0) -> np.ndarray:
        tensor_clues = torch.tensor(
            np.asarray(clues).flatten(), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Clamp before logit to avoid ±inf on exact 0/1 boundaries.
        prev_t = torch.tensor(prev, dtype=torch.float32, device=self.device)
        prev_t = torch.clamp(prev_t, 1e-6, 1 - 1e-6)
        logit_grid = torch.logit(prev_t, eps=1e-6)

        steps_torch = [logit_grid]
        n = logit_grid.numel()

        self._reset_optimizer_state(n)

        for _ in range(num_steps):
            current = steps_torch[-1]
            new_logits = self._optimise(current, n, sampling_ratio, tensor_clues)
            steps_torch.append(new_logits)

        steps = [torch.sigmoid(step).detach().cpu().numpy() for step in steps_torch]
        return np.array(steps)


# =============================================================================
# Sampling mixins
# =============================================================================

class DiscreteSamplingMixin:
    def _sample_grid(self, n: int, sampling_ratio: float) -> torch.Tensor:
        return (torch.rand(n, device=self.device) < sampling_ratio).float()

    # Already a hard {0, 1} mask, so reuse it directly.
    def _sample_mask(self, n: int, sampling_ratio: float) -> torch.Tensor:
        return self._sample_grid(n, sampling_ratio)


class BetaSamplingMixin:
    def _sample_grid(
        self, n: int, sampling_ratio: float, concentration: float = 10
    ) -> torch.Tensor:
        if sampling_ratio <= 0:
            return torch.zeros(n, device=self.device)
        if sampling_ratio >= 1:
            return torch.ones(n, device=self.device)
        alpha = sampling_ratio * concentration
        beta = (1 - sampling_ratio) * concentration
        return torch.distributions.Beta(alpha, beta).sample((n,)).to(self.device)

    # A continuous Beta draw is essentially never exactly 0, so it cannot be
    # used to select a sparse subset via nonzero(). Selection needs its own
    # hard draw.
    def _sample_mask(self, n: int, sampling_ratio: float) -> torch.Tensor:
        return (torch.rand(n, device=self.device) < sampling_ratio).float()


# =============================================================================
# Optimiser mixins
# =============================================================================

class AdamOptimizerMixin:

    def __init__(
        self,
        step_size: float = 5e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        **kwargs,
    ):
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = None
        self._v = None
        self._t = None
        super().__init__(**kwargs)

    def _reset_optimizer_state(self, n: int) -> None:
        self._m = torch.zeros(n, device=self.device)
        self._v = torch.zeros(n, device=self.device)
        self._t = torch.zeros(n, dtype=torch.long, device=self.device)

    def _compute_update(self, grad: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        self._t[idxs] = self._t[idxs] + 1
        t = self._t[idxs]

        m, v = self._m[idxs], self._v[idxs]

        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad.pow(2)

        self._m[idxs] = m
        self._v[idxs] = v

        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)

        return self.step_size * m_hat / (v_hat.sqrt() + self.eps)


class SGDOptimizerMixin:

    def __init__(self, step_size: float = 1, **kwargs):
        self.step_size = step_size
        super().__init__(**kwargs)

    def _reset_optimizer_state(self, n: int) -> None:
        pass  # SGD is stateless

    def _compute_update(self, grad: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        return self.step_size * grad


# =============================================================================
# Update-strategy mixins
# =============================================================================

class GlobalMixin:

    def _optimise(
        self,
        prev: torch.Tensor,
        n: int,
        sampling_ratio: float,
        tensor_clues: torch.Tensor,
    ) -> torch.Tensor:
        weight = self._sample_grid(n, sampling_ratio)

        anchor = prev.flatten().detach()
        flat_logits = anchor.clone().requires_grad_(True)

        loss = self.loss_fn(flat_logits.unsqueeze(0), tensor_clues)
        if isinstance(loss, tuple):
            loss = loss[0]
        loss = loss.sum()

        grad = torch.autograd.grad(loss, flat_logits)[0]

        with torch.no_grad():
            all_idxs = torch.arange(anchor.numel(), device=self.device)
            update = self._compute_update(grad, all_idxs)
            flat_opt = flat_logits - update
            blended = anchor + weight * (flat_opt - anchor)

        return blended.view(prev.shape)


class GaussSeidelMixin:

    def _optimise(
        self,
        prev: torch.Tensor,
        n: int,
        sampling_ratio: float,
        tensor_clues: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._sample_mask(n, sampling_ratio)
        idxs = mask.nonzero(as_tuple=True)[0]

        grid = prev.flatten().clone().detach()
        if idxs.numel() == 0:
            return grid.view(prev.shape)

        order = idxs[torch.randperm(idxs.numel(), device=self.device)]

        for i in order:
            i = i.unsqueeze(0)  # keep everything 1-element-tensor shaped

            fixed_val = grid[i]
            free_val = fixed_val.clone().requires_grad_(True)

            candidate = grid.clone()
            candidate[i] = free_val
            loss = self.loss_fn(candidate.unsqueeze(0), tensor_clues)
            if isinstance(loss, tuple):
                loss = loss[0]
            loss = loss.sum()
            grad = torch.autograd.grad(loss, free_val)[0]

            with torch.no_grad():
                update = self._compute_update(grad, i)
                grid[i] = fixed_val - update

        return grid.view(prev.shape)


class JacobiMixin:

    def _optimise(
        self,
        prev: torch.Tensor,
        n: int,
        sampling_ratio: float,
        tensor_clues: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._sample_mask(n, sampling_ratio)
        idxs = mask.nonzero(as_tuple=True)[0]

        grid = prev.flatten().clone().detach()
        if idxs.numel() == 0:
            return grid.view(prev.shape)

        anchor = grid.clone()  # frozen starting point for all gradients
        order = idxs[torch.randperm(idxs.numel(), device=self.device)]
        deltas = []

        for i in order:
            i = i.unsqueeze(0)

            fixed_val = anchor[i]  # gradient anchored to start of pass
            free_val = fixed_val.clone().requires_grad_(True)

            candidate = anchor.clone()
            candidate[i] = free_val
            loss = self.loss_fn(candidate.unsqueeze(0), tensor_clues)
            if isinstance(loss, tuple):
                loss = loss[0]
            loss = loss.sum()
            grad = torch.autograd.grad(loss, free_val)[0]

            with torch.no_grad():
                update = self._compute_update(grad, i)
                deltas.append((i, update, fixed_val))

        # Apply all deltas simultaneously (all relative to anchor)
        for i, update, fixed_val in deltas:
            grid[i] = fixed_val - update

        return grid.view(prev.shape)


# =============================================================================
# Concrete solvers
# =============================================================================

class GDGlobalAdamSolver(DiscreteSamplingMixin, AdamOptimizerMixin, GlobalMixin, BaseGradientDescentSolver):
    pass
class GDGlobalSGDSolver(DiscreteSamplingMixin, SGDOptimizerMixin, GlobalMixin, BaseGradientDescentSolver):
    pass
class GDGaussSeidelAdamSolver(DiscreteSamplingMixin, AdamOptimizerMixin, GaussSeidelMixin, BaseGradientDescentSolver):
    pass
class GDGaussSeidelSGDSolver(DiscreteSamplingMixin, SGDOptimizerMixin, GaussSeidelMixin, BaseGradientDescentSolver):
    pass
class GDJacobiAdamSolver(DiscreteSamplingMixin, AdamOptimizerMixin, JacobiMixin, BaseGradientDescentSolver):
    pass
class GDJacobiSGDSolver(DiscreteSamplingMixin, SGDOptimizerMixin, JacobiMixin, BaseGradientDescentSolver):
    pass