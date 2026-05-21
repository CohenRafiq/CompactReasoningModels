import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def grid_to_row_clues(grid, K):
    """
    grid: Tensor of shape [batch, num_lines, line_length]
    K: Number of clues to extract per row
    Returns: Tensor of shape [batch, num_lines, K]
    """

    shifted_right = torch.cat([torch.zeros_like(grid[..., :1]), grid[..., :-1]], dim=-1)
    left_run_ends = grid * (1.0 - shifted_right)
    cumulative_run_ends = torch.cumsum(left_run_ends, dim=-1)
    

    clue_indices = torch.arange(1, K + 1, dtype=grid.dtype, device=grid.device)
    
    # 5. Build weights matrix
    # cumulative_run_ends: [B, N, L] -> [B, N, L, 1]
    # k: [K]       -> broadcasts to [B, N, L, K]
    weights = torch.clamp(1.0 - torch.abs(cumulative_run_ends.unsqueeze(-1) - clue_indices), min=0.0)
    
    # 6. Apply weights and sum over the line_length dimension
    # grid: [B, N, L] -> [B, N, L, 1]
    # Result shape: [B, N, K]
    return (grid.unsqueeze(-1) * weights).sum(dim=-2)

def _nonogram_loss(grid, row_clues, col_clues, tau=0.3):
    side = grid.shape[-1]  # 5 for a 5×5 grid

    max_row_runs = row_clues.shape[2]
    max_col_runs = col_clues.shape[2]

    predicted_row_lengths = grid_to_row_clues(grid, max_row_runs)
    row_loss = F.mse_loss(
        predicted_row_lengths / side,   # normalise
        row_clues.float() / side,       # normalise
        reduction='none'
    )

    predicted_col_lengths = grid_to_row_clues(grid.transpose(1, 2), max_col_runs)
    col_loss = F.mse_loss(
        predicted_col_lengths / side,
        col_clues.float() / side,
        reduction='none'
    )

    per_sample = row_loss.mean(dim=[1, 2]) + col_loss.mean(dim=[1, 2])
    return per_sample, row_loss.mean(dim=[1, 2]), col_loss.mean(dim=[1, 2])


class NonogramLoss(nn.Module):
    """
    A drop‑in replacement for a standard criterion (e.g. BCEWithLogitsLoss).

    Parameters
    ----------
    tau : float, default 0.3
        Temperature for the soft‑rank mask. Lower → sharper rank assignment.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        How to aggregate the per‑sample losses.
    """
    def __init__(self, tau: float = 0.3, reduction: str = 'mean'):
        super().__init__()
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError(f"Invalid reduction: {reduction}")
        self.tau = tau
        self.reduction = reduction

    def forward(self,
                grid: torch.Tensor,
                clues: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        grid = grid.reshape(grid.shape[0], math.isqrt(grid.shape[1]), -1)
        # Ensure grid is a probability map; if you feed logits, apply sigmoid first

        clues_reshaped = clues.reshape(
            clues.shape[0],          # N
            2,                       # split into two groups (row / col)
            grid.shape[1],      # e.g., number of columns W
            -1                       # infer the last dimension (must be integer)
        )

        row_clues = clues_reshaped[:, 0, :, :]   # shape: (N, grid.shape[1], X)
        col_clues = clues_reshaped[:, 1, :, :]   # shape: (N, grid.shape[1], X)
        per_sample, row_loss, col_loss = _nonogram_loss(grid, row_clues, col_clues, self.tau)

        if self.reduction == 'none':
            return per_sample, row_loss, col_loss
        elif self.reduction == 'mean':
            return per_sample.mean(), row_loss.mean(), col_loss.mean()
        else:  # 'sum'
            return per_sample.sum(), row_loss.sum(), col_loss.sum()