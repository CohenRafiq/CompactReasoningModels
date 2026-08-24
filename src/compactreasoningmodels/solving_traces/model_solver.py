
import numpy as np
import torch

from compactreasoningmodels.models.base import BaseModel
from compactreasoningmodels.models.recursive_gridmlp import RecursiveGridMLP
from compactreasoningmodels.solving_traces.base import SolvingTrace
from compactreasoningmodels.utils.load_model import load_model


class ModelSolver(SolvingTrace):

    def __init__(self, clues: np.ndarray, grid_shape: tuple[int, int],
                 model: BaseModel | None = None, initial_grid: np.ndarray | None = None):
        super().__init__(clues, grid_shape, initial_grid)
        if model is None:
            model = load_model(
                RecursiveGridMLP,
                "./models/jsonldataset/recursivegridmlp/06.pt",
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                input_size=30,
                output_size=75,
                hidden_size=256,
                num_layers=9,
                dropout=0.3
            )
        self.model = model
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_clues = (
            torch.from_numpy(clues)
            .flatten()
            .unsqueeze(0)
            .to(device)
            )

    def compress_categorical_abstain(self, grid: torch.Tensor) -> np.ndarray:
        # Input (3, H, W) → Output (H, W)
        smooth = torch.softmax(grid, dim=0)
        compressed = (smooth[1] + 0.5 * smooth[2])/3
        return compressed.cpu().detach().numpy()



    def heatmap_step(self, grid: np.ndarray, num_steps: int = 6) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(
                self.tensor_clues, layer_num=num_steps
                ).cpu().detach()[0].reshape(3, 5, 5)
            if logits.ndim == 3 or logits.shape[0] == 3:
                logits = self.compress_categorical_abstain(logits)
        return logits


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
    solver = ModelSolver(clues, (5, 5))
    grid = np.full((5, 5), 0.5)
    for _ in range(5):
        grid = solver.heatmap_step(grid)
        print(grid)
