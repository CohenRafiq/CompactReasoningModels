import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def soft_run_lengths(
    grid: torch.Tensor,       # (B, num_lines, line_len), values in [0, 1]
    max_runs: int,
    tau: float = 0.5,
) -> torch.Tensor:
    """
    Differentiable extraction of soft run lengths.
    
    Key fix vs. original: run_ends uses grid * (1 - right_neighbor) instead
    of ReLU, which keeps gradients alive everywhere in [0,1]^2.
    """
    B, L, N = grid.shape

    # Pad right with 0 so the last cell always terminates a run
    padded = F.pad(grid, (0, 1))                        # (B, L, N+1)
    right_neighbor = padded[:, :, 1:]                   # (B, L, N)  grid[i+1], 0 at boundary

    # Soft run-end indicator: high when cell is "on" and next cell is "off"
    # Fully differentiable — no ReLU dead zones
    run_ends = grid * (1.0 - right_neighbor)            # (B, L, N)

    # Soft prefix count of filled cells
    prefix_sums = grid.cumsum(dim=2)                    # (B, L, N)

    # Weight prefix sums by where runs end
    prefix_at_end = prefix_sums * run_ends              # (B, L, N)

    # Soft rank: roughly "how many run-ends have occurred up to position m"
    # Adding epsilon prevents the cumsum from assigning rank 0 to early ends
    end_cumsum = run_ends.cumsum(dim=2)                 # (B, L, N)
    # Shift by 0.5 so the first end gets rank ≈0.5, not ≈0 (avoids bucket collapse)
    end_rank = (end_cumsum - 0.5 * run_ends) * run_ends # (B, L, N)

    # Gaussian soft-assignment to run slots k = 1..max_runs
    k_values = torch.arange(1, max_runs + 1, device=grid.device, dtype=grid.dtype)
    # (B, L, N, max_runs)
    rank_mask = torch.exp(
        -(end_rank.unsqueeze(-1) - k_values) ** 2 / (2 * tau ** 2)
    )
    # Normalise per position so weights sum to 1 across slots (softmax-style)
    rank_mask = rank_mask / (rank_mask.sum(dim=-1, keepdim=True) + 1e-8)

    # Accumulate prefix lengths into each run slot
    cumulative_lengths = torch.einsum("blm, blmk -> blk", prefix_at_end, rank_mask)

    # Convert cumulative → per-run lengths
    run_lengths = torch.diff(
        cumulative_lengths,
        prepend=torch.zeros(B, L, 1, device=grid.device, dtype=grid.dtype),
        dim=-1,
    )
    return run_lengths          # (B, num_lines, max_runs)


def _nonogram_loss(grid: torch.Tensor,
                   row_clues: torch.Tensor,
                   col_clues: torch.Tensor,
                   tau: float = 0.3) -> torch.Tensor:

    max_row_runs = row_clues.shape[2] 
    max_col_runs = col_clues.shape[2]

    predicted_row_lengths = soft_run_lengths(grid, max_row_runs, tau)
    row_loss = F.mse_loss(predicted_row_lengths, row_clues.float(),
                          reduction='none')

    predicted_col_lengths = soft_run_lengths(grid.transpose(1, 2),
                                             max_col_runs, tau)
    col_loss = F.mse_loss(predicted_col_lengths, col_clues.float(),
                          reduction='none')

    return row_loss.mean(dim=[1, 2]) + col_loss.mean(dim=[1, 2])


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
        per_sample = _nonogram_loss(grid, row_clues, col_clues, self.tau)

        if self.reduction == 'none':
            return per_sample
        elif self.reduction == 'mean':
            return per_sample.mean()
        else:  # 'sum'
            return per_sample.sum()