from collections.abc import Generator

import pandas as pd
import torch

from compactreasoningmodels.experiments.base import BaseExperiment


class NoisySimilarityExperiment(BaseExperiment):

    def _add_noise_to_grid(self, grid: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
        noise = torch.rand_like(grid) * 2 * noise_level - noise_level
        return (1 - noise_level) * grid + noise

    def next_solver(self, solver_steps: dict[str, int] | int | None = None
                    ) -> Generator[tuple[str, int], None, None]:
        match solver_steps:
            case dict():
                steps = solver_steps
            case int():
                steps = dict.fromkeys(self.solvers, solver_steps)
            case None:
                steps = dict.fromkeys(self.solvers, 1)
            case _:
                raise TypeError(
                    f"solver_steps must be dict, int, or None, got {type(solver_steps)}"
                    )
        for solver_name in self.solvers:
            yield solver_name, steps.get(solver_name, 1)

    def compute_clean_heatmaps(self, num_samples: int = 100, initial_grid: str = "random",
                                solver_steps: dict[str, int] | int | None = None
                                ) -> dict[str, torch.Tensor]:
        clean_heatmaps: dict[str, list[torch.Tensor]] = {name: [] for name in self.solvers}
        for name, _, heatmap, _ in self.generate_heatmaps(
                num_samples=num_samples,
                initial_grid=initial_grid,
                solver_steps=solver_steps):
            clean_heatmaps[name].append(torch.tensor(heatmap))

        return {name: torch.stack(grids) for name, grids in clean_heatmaps.items()}

    def run_experiment(self, num_samples: int = 100, initial_grid: str = "random",
                        solver_steps: dict[str, int] | int | None = None,
                        noise_level: float = 0.1,
                        ) -> dict[str, pd.DataFrame]:

        clean_heatmaps = self.compute_clean_heatmaps(
            num_samples=num_samples,
            initial_grid=initial_grid,
            solver_steps=solver_steps)

        noisy_heatmaps = {
            name: self._add_noise_to_grid(grids, noise_level=noise_level)
            for name, grids in clean_heatmaps.items()
        }

        matrices = {}
        for metric_name, metric in self.metrics.items():
            scores = {
                noisy_name: {
                    clean_name: metric(noisy_heatmaps[noisy_name],
                                       clean_heatmaps[clean_name]).mean().item()
                    for clean_name in self.solvers
                }
                for noisy_name in self.solvers
            }
            matrices[metric_name] = pd.DataFrame(scores)

        self.print_results(matrices)
        return matrices

    def print_results(self, matrices: dict[str, pd.DataFrame]):
        for metric_name, matrix in matrices.items():
            matrix = matrix.round(4)
            matrices[metric_name] = matrix
            print(f"{metric_name} Matrix (rows=clean, cols=noisy):")
            print(matrix)
            print()
