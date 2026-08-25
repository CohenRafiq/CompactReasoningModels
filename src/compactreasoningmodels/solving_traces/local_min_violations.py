import math
import random

import numpy as np
import torch

from compactreasoningmodels.losses.nonogram import NonogramLoss
from compactreasoningmodels.solving_traces.base import SolvingTrace

# Greedily updates every cell based on what value
# Would minimise loss (satisfy clues)

class LocalMinViolations(SolvingTrace):

    def __init__(self, clues: np.ndarray, grid_shape: tuple[int, int],
                 initial_grid: np.ndarray | None = None, hit_rate: float = 0.5):
        super().__init__(clues, grid_shape, initial_grid)
        self.loss_fn = NonogramLoss(reduction="mean")
        self.tensor_clues = torch.tensor(clues.flatten(), dtype=torch.float32).unsqueeze(0)
        self.rows, self.cols = grid_shape
        self.hit_rate = hit_rate

    def min_loss_cell(self, logit_grid: torch.Tensor, r: int, c: int) -> torch.Tensor:
        flat_idx = r * self.cols + c
        low, high = self.coarse_sampling(logit_grid, flat_idx)
        optimal_value = self.refine_with_golden_section(logit_grid, flat_idx, low, high)
        return optimal_value

    def coarse_sampling(self, logit_grid: torch.Tensor, flat_idx: int) -> tuple[float, float]:
        coarse_values = torch.linspace(-8, 8, 50, dtype=torch.float32)
        losses = []
        grid_tensor_base = logit_grid.flatten()

        for val in coarse_values:
            test_grid = grid_tensor_base.clone()
            test_grid[flat_idx] = val
            test_grid = test_grid.unsqueeze(0)
            loss, _, _, _ = self.loss_fn(test_grid, self.tensor_clues)
            losses.append(loss.item())

        losses_tensor = torch.tensor(losses)
        min_idx = int(torch.argmin(losses_tensor))

        if min_idx > 0 and min_idx < len(coarse_values) - 1:
            low = coarse_values[min_idx - 1].item()
            high = coarse_values[min_idx + 1].item()
        else:
            low = coarse_values[max(0, min_idx - 2)].item()
            high = coarse_values[min(min_idx + 2, len(coarse_values) - 1)].item()

        return low, high

    def refine_with_golden_section(self, logit_grid: torch.Tensor, flat_idx: int,
                                    low: float, high: float,
                                    tol: float = 1e-4, max_iter: int = 50) -> torch.Tensor:
        gr = (math.sqrt(5) + 1) / 2
        grid_tensor_base = logit_grid.flatten()
        original = grid_tensor_base[flat_idx].item()

        for _ in range(max_iter):
            if abs(high - low) < tol:
                break

            mid1 = low + (high - low) / gr
            mid2 = high - (high - low) / gr

            grid_tensor_base[flat_idx] = mid1
            loss1, _, _, _ = self.loss_fn(grid_tensor_base.unsqueeze(0), self.tensor_clues)

            grid_tensor_base[flat_idx] = mid2
            loss2, _, _, _ = self.loss_fn(grid_tensor_base.unsqueeze(0), self.tensor_clues)

            if loss1.item() < loss2.item():
                low = mid2
            else:
                high = mid1

        grid_tensor_base[flat_idx] = original
        return torch.tensor((low + high) / 2, dtype=torch.float32)

    def heatmap_step(self, grid: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logit_grid = torch.logit(torch.tensor(grid, dtype=torch.float32), eps=1e-6)
            flat_logits = logit_grid.flatten()
            snapshot = flat_logits.clone()          # frozen reference for this whole pass
            new_logits = flat_logits.clone()

            for r in range(self.rows):
                for c in range(self.cols):
                    if random.random() < self.hit_rate:  # Only update a fraction of cells
                        flat_idx = r * self.cols + c
                        new_logits[flat_idx] = self.min_loss_cell(snapshot, r, c)

        new_grid = torch.sigmoid(new_logits.view(self.rows, self.cols)).numpy()
        return new_grid


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
    solver = LocalMinViolations(clues, (5, 5))
    grid = np.full((5, 5), 0.5)
    for _ in range(5):
        grid = solver.heatmap_step(grid)
        print(grid)
