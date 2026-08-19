import torch
import numpy as np

def _clues_torch(grid):
    if not isinstance(grid, torch.Tensor):
        grid = torch.tensor(grid, dtype=torch.float32)
    H, W = grid.shape
    K_row = (W + 1) // 2
    K_col = (H + 1) // 2
    row_clues = torch.stack([_row_torch(grid[i,  :], K_row) for i in range(H)])
    col_clues = torch.stack([_row_torch(grid[:, j ], K_col) for j in range(W)])
    return row_clues, col_clues

def _row_torch(x, K, alpha=10.0):
    on      = torch.sigmoid(alpha * (x - 0.5))
    on_prev = torch.cat([x.new_zeros(1), on[:-1]])
    s       = on * (1.0 - on_prev)
    c       = torch.cumsum(s, dim=0)
    k       = torch.arange(1, K + 1, dtype=x.dtype, device=x.device)
    weights = torch.clamp(1.0 - torch.abs(c.unsqueeze(1) - k.unsqueeze(0)), min=0.0)
    return (x.unsqueeze(1) * weights).sum(dim=0)


def nonogram_clues_smooth(grid, alpha=10.0):
    grid = np.asarray(grid, dtype=float)
    H, W = grid.shape
    K_row = (W + 1) // 2
    K_col = (H + 1) // 2

    def row(x, K):
        on      = 1.0 / (1.0 + np.exp(alpha * 0.5 - alpha * x))
        on_prev = np.concatenate([[0.0], on[:-1]])
        s       = on * (1.0 - on_prev)
        c       = np.cumsum(s)
        k       = np.arange(1, K + 1)
        weights = np.maximum(0.0, 1.0 - np.abs(c[:, None] - k[None, :]))
        return (x[:, None] * weights).sum(axis=0)

    row_clues = np.stack([row(grid[i,  :], K_row) for i in range(H)])
    col_clues = np.stack([row(grid[:, j ], K_col) for j in range(W)])
    return row_clues, col_clues


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    sep = "-" * 50

    print(sep)
    print("Example 1: integer-valued grid")
    print(sep)
    grid1 = np.array([[0, 0, 3],
                      [0, 2, 0],
                      [1, 0, 1]], dtype=float)
    row_clues, col_clues = nonogram_clues_smooth(grid1)
    print("Grid:\n", grid1)
    print("\nRow clues:\n", row_clues)
    print("\nCol clues:\n", col_clues)

    print("\n" + sep)
    print("Example 2: binary 4x5 nonogram")
    print(sep)
    grid2 = np.array([[1, 1, 0, 1, 1],
                      [0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 1],
                      [0, 0, 1, 0, 0]], dtype=float)
    row_clues, col_clues = nonogram_clues_smooth(grid2)
    print("Grid:\n", grid2)
    print("\nRow clues:\n", row_clues)
    print("\nCol clues:\n", col_clues)

    print("\n" + sep)
    print("Example 3: continuous-valued grid")
    print(sep)
    grid3 = np.array([[0.9, 0.8, 0.0, 0.7],
                      [0.0, 0.6, 0.0, 0.0],
                      [0.5, 0.0, 0.3, 0.4]])
    row_clues, _ = nonogram_clues_smooth(grid3)
    print("Grid:\n", grid3)
    print("\nRow clues:\n", row_clues)

    print("\n" + sep)
    print("Example 4: PyTorch autograd")
    print(sep)
    grid_t = torch.tensor([[0., 0., 3.],
                           [0., 2., 0.],
                           [1., 0., 1.]], requires_grad=True)
    row_clues_t, _ = _clues_torch(grid_t)
    print("Row clues:\n", row_clues_t.detach().numpy())
    row_clues_t[0, 0].backward()
    print("\nGradient:")
    print(grid_t.grad.numpy())

    print("\n" + sep)
    print("Example 5: smooth sigmoid vs clip variant")
    print(sep)
    grid4 = np.array([[1, 0, 1, 1, 0, 1]])
    rc_soft, _ = nonogram_clues_smooth(grid4, alpha=10.0)
    print("Grid:          ", grid4[0])
    print("Smooth variant:", rc_soft[0])
