import numpy as np
import torch
from torch.utils.data import DataLoader

from compactreasoningmodels.losses.nonogram import NonogramLoss
from compactreasoningmodels.solving_traces.arc_consistency import ArcConsistency
from compactreasoningmodels.solving_traces.discrete_genetic import DiscreteGeneticAlgorithm
from compactreasoningmodels.solving_traces.global_min_violations import GlobalMinViolations
from compactreasoningmodels.solving_traces.local_min_violations import LocalMinViolations
from compactreasoningmodels.solving_traces.model_solver import ModelSolver
from compactreasoningmodels.utils.types import SolverProfile


class HeatmapStore:
    _solvers = {
        "arc_consistency": ArcConsistency,
        "local_min_violations": LocalMinViolations,
        "global_min_violations": GlobalMinViolations,
        "discrete_genetic": DiscreteGeneticAlgorithm,
        "model_solver": ModelSolver
    }

    def __init__(self, dataloader: DataLoader, initial_grid_type: str = "random"):
        self.dataloader = dataloader
        self.heatmaps = {}
        self.initial_grid = HeatmapStore._gen_inital_grid(
            initial_grid_type,
            self.dataloader.dataset.target_shape)

    @staticmethod
    def _gen_inital_grid(gen_type: str, shape: tuple[int, int]) -> np.ndarray:
        match gen_type:
            case "random":
                return np.random.rand(*shape)
            case "zeros":
                return np.zeros(shape)
            case _:
                raise ValueError(f"Unknown initial grid type: {gen_type}")

    def add_heatmaps(self, solver_profiles: list[SolverProfile] | list[tuple[str, int]]
                     ) -> dict[SolverProfile, list[tuple[np.ndarray, np.ndarray]]]:
        solver_profiles = [
            p if isinstance(p, SolverProfile) else SolverProfile(*p)
            for p in solver_profiles
        ]
        target_shape = self.dataloader.dataset.target_shape
        for solver_profile in solver_profiles:
            if solver_profile in self.heatmaps:
                continue  # Skip if heatmap already exists
            results = []

            for input_tensor, target_grid in self.dataloader:
                solver_class = self._solvers[solver_profile.name]
                solver_instance = solver_class(
                    input_tensor, target_shape,
                    initial_grid=self.initial_grid)
                heatmap = solver_instance.heatmap(num_steps=solver_profile.num_steps)
                results.append((heatmap, input_tensor, target_grid.squeeze(0).numpy()))

            self.heatmaps[solver_profile] = results
        return self.heatmaps

    def ascii_distribution(self, bins: int = 20, width: int = 50) -> None:
        for solver_profile, results in self.heatmaps.items():
            heatmaps = np.array([hm for hm, _, _ in results])
            values = heatmaps.flatten()
            hist, bin_edges = np.histogram(values, bins=bins, range=(0, 1))
            max_count = hist.max()
            print(f"\n{solver_profile.name} (steps={solver_profile.num_steps})")
            print("-" * 70)
            print(f"  Count: {len(values)} values")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Std: {np.std(values):.4f}")
            print(f"  Min: {np.min(values):.4f}")
            print(f"  Max: {np.max(values):.4f}")
            print()
            for i in range(len(hist)):
                if hist[i] > 0:
                    bar_length = int((hist[i] / max_count) * width)
                    bar = "█" * bar_length
                    percentage = (hist[i] / len(values)) * 100
                    print(f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}: {bar} {percentage:.1f}%")
                else:
                    print(f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}:")

            print("-" * 70)

    def display_accuracy(self) -> None:
        for (solver_name, num_steps), results in self.heatmaps.items():
            correct_count = sum(
                1 for heatmap, _, target in results
                if np.allclose(heatmap, target, atol=1e-2)
            )
            accuracy = correct_count / len(results) * 100
            print(f"{solver_name} (steps={num_steps}): Accuracy = {accuracy:.2f}%")

    def display_mse_loss(self) -> None:
        for (solver_name, num_steps), results in self.heatmaps.items():
            heatmap_tensors = torch.tensor(np.array([hm for hm, _, _ in results]), dtype=torch.float32).flatten(start_dim=1)
            target_tensors = torch.tensor(np.array([target for _, _, target in results]), dtype=torch.float32).flatten(start_dim=1)
            total_loss = torch.nn.functional.mse_loss(heatmap_tensors, target_tensors)
            average_loss = total_loss / len(results)
            print(f"{solver_name} (steps={num_steps}): Average Loss = {average_loss:.4f}")

    def display_clue_loss(self) -> None:
        for (solver_name, num_steps), results in self.heatmaps.items():
            heatmap_tensors = torch.tensor(np.array([hm for hm, _, _ in results]), dtype=torch.float32).flatten(start_dim=1)
            clue_tensors = torch.tensor(np.array([clue for _, clue, _ in results]), dtype=torch.float32).flatten(start_dim=1)
            total_loss = NonogramLoss()(heatmap_tensors, clue_tensors)[0]
            average_loss = total_loss / len(results)
            print(f"{solver_name} (steps={num_steps}): Average Loss = {average_loss:.4f}")

