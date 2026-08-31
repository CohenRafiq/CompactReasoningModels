import numpy as np
import torch

from compactreasoningmodels.losses.nonogram import NonogramLoss
from compactreasoningmodels.solving_traces.base import SolvingTrace

# Updates the entire grid using gradient descent
# to minimize the loss (satisfy clues)


class GlobalMinViolations(SolvingTrace):
    def __init__(
        self,
        clues: np.ndarray | torch.Tensor,
        grid_shape: tuple[int, int],
        initial_grid: np.ndarray | torch.Tensor | None = None,
    ):
        super().__init__(clues, grid_shape, initial_grid)
        self.loss_fn = NonogramLoss(reduction="mean")
        self.tensor_clues = torch.tensor(self.clues.flatten(), dtype=torch.float32).unsqueeze(0)
        self.rows, self.cols = grid_shape

    def heatmap_step(
        self,
        grid: np.ndarray,
        lr: float = 0.01,
        inner_steps: int = 50,
        tol: float = 1e-7,
    ) -> np.ndarray:
        with torch.no_grad():
            logit_grid = torch.logit(torch.tensor(grid, dtype=torch.float32), eps=1e-6)

        flat_logits = logit_grid.flatten().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([flat_logits], lr=lr)

        prev_loss = None
        for _ in range(inner_steps):
            optimizer.zero_grad()
            loss, _, _, _ = self.loss_fn(flat_logits.unsqueeze(0), self.tensor_clues)
            loss.backward()
            optimizer.step()

            cur_loss = loss.item()
            if prev_loss is not None and abs(prev_loss - cur_loss) < tol:
                break
            prev_loss = cur_loss

        with torch.no_grad():
            new_grid = torch.sigmoid(flat_logits.view(self.rows, self.cols)).numpy()

        return new_grid


if __name__ == "__main__":
    clues = np.array(
        [
            [[0, 0, 0], [3, 0, 0], [1, 1, 0], [3, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [3, 0, 0], [1, 1, 0], [3, 0, 0], [0, 0, 0]],
        ]
    )
    solver = GlobalMinViolations(clues, (5, 5))
    grid = np.full((5, 5), 0.5)
    for _ in range(5):
        grid = solver.heatmap_step(grid)
        print(grid)
