import pytest
import torch

from compactreasoningmodels.losses.nonogram import NonogramLoss


def _integer_clues(grid: torch.Tensor, K: int) -> torch.Tensor:
    """Compute exact run-length clues as integers padded to length K."""
    H, W = grid.shape
    row_clues = torch.zeros(H, K)
    col_clues = torch.zeros(W, K)

    def runs(line: torch.Tensor) -> torch.Tensor:
        lengths = []
        count = 0
        for v in line.tolist():
            if v == 1:
                count += 1
            elif count:
                lengths.append(count)
                count = 0
        if count:
            lengths.append(count)
        return torch.tensor(lengths[:K], dtype=torch.float32)

    for i in range(H):
        r = runs(grid[i, :])
        row_clues[i, : r.numel()] = r
    for j in range(W):
        c = runs(grid[:, j])
        col_clues[j, : c.numel()] = c
    return row_clues, col_clues


def test_perfect_prediction_gives_zero_loss():
    grid = torch.tensor(
        [[1, 1, 0, 1], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]],
        dtype=torch.float32,
    )
    flat = torch.where(grid.reshape(1, -1) > 0.5, torch.tensor(6.0), torch.tensor(-6.0))
    row_clues, col_clues = _integer_clues(grid, K=2)
    clues = torch.cat([row_clues.reshape(1, -1), col_clues.reshape(1, -1)], dim=-1)

    loss = NonogramLoss(reduction="mean")
    per_sample, _, _, match = loss(flat, clues)
    assert per_sample < 1e-3
    assert match > 0.99


def test_wrong_prediction_gives_larger_loss():
    loss = NonogramLoss(reduction="mean")

    ones = torch.full((1, 16), 6.0)
    zeros = torch.full((1, 16), -6.0)
    row, col = _integer_clues(torch.ones(4, 4), K=2)
    clues = torch.cat([row.reshape(1, -1), col.reshape(1, -1)], dim=-1)

    loss_ones = loss(ones, clues)[0]
    loss_zeros = loss(zeros, clues)[0]
    assert loss_zeros > loss_ones


def test_reduction_modes():
    grid = torch.full((2, 16), 6.0)
    row, col = _integer_clues(torch.ones(4, 4), K=2)
    clues = torch.cat([row.reshape(1, -1), col.reshape(1, -1)], dim=-1).repeat(2, 1)

    mean = NonogramLoss(reduction="mean")
    none = NonogramLoss(reduction="none")
    sum_loss = NonogramLoss(reduction="sum")

    mean_out = mean(grid, clues)
    none_out = none(grid, clues)
    sum_out = sum_loss(grid, clues)

    assert mean_out[0].shape == ()
    assert none_out[0].shape == (2,)
    assert torch.isclose(sum_out[0], none_out[0].sum())


def test_invalid_reduction_raises():
    with pytest.raises(ValueError):
        NonogramLoss(reduction="bogus")
