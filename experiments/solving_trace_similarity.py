import torch
import numpy as np
import itertools
import pandas as pd

from compactreasoningmodels.solving_traces.discrete_genetic import DiscreteGeneticAlgorithm
from compactreasoningmodels.solving_traces.global_min_violations import GlobalMinViolations
from compactreasoningmodels.solving_traces.local_min_violations import LocalMinViolations
from compactreasoningmodels.solving_traces.arc_consistency import ArcConsistency
from compactreasoningmodels.solving_traces.similarity_measures import MSE, PearsonCorrelation
from compactreasoningmodels.datasets.jsonl import JsonlDataset

SOLVERS = {
    "arc_consistency": ArcConsistency, 
    "local_min_violations": LocalMinViolations, 
    "global_min_violations": GlobalMinViolations, 
    "discrete_genetic": DiscreteGeneticAlgorithm
}

METRICS = {
    "MSE": MSE(),
    "Pearson Correlation": PearsonCorrelation()
}

def add_noise_to_grid(grid: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    noise = np.random.uniform(-noise_level, noise_level, size=grid.shape)
    noisy_grid = (1-noise_level) * grid + noise
    return noisy_grid

def main(number_samples=100, print_every=None):
    dataset = JsonlDataset("raw/nonogram_5x5_small.jsonl", target_shape=(5, 5))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    heatmaps = {name: [] for name in SOLVERS.keys()}
    
    for i, (input_tensor, target_tensor) in enumerate(dataloader):
        if i >= number_samples:
            break
        clues = input_tensor.squeeze(0).numpy()
        clues = clues.reshape(2, 5, 3)
        target = target_tensor.squeeze(0).numpy()
        initial_grid = np.random.uniform(0.1, 0.9, size=target.shape)
        
        solvers = {name: solver_cls(clues, target.shape)
                   for name, solver_cls in SOLVERS.items()}
        
        for name, solver in solvers.items():
            hm = solver.heatmap_step(initial_grid)
            heatmaps[name].append(hm)

        if print_every is not None and (i + 1) % print_every == 0:
            print(f"Processed {i + 1}/{number_samples} samples.")

    solver_names = list(SOLVERS.keys())


    matrices = {
        name: pd.DataFrame(index=solver_names, columns=solver_names, dtype=float)
        for name in METRICS
    }

    for noisy, clean in itertools.product(solver_names, repeat=2):
        noisy_grids = add_noise_to_grid(np.array(heatmaps[noisy]))
        clean_grids = np.array(heatmaps[clean])

        for metric_name, metric in METRICS.items():
            result = np.mean(metric(noisy_grids, clean_grids))
            matrices[metric_name].loc[noisy, clean] = result

    for metric_name, matrix in matrices.items():
        matrix = matrix.round(4)
        matrices[metric_name] = matrix
        print(f"{metric_name} Matrix (rows=noisy, cols=clean):")
        print(matrix)
        print()

if __name__ == "__main__":
    main(number_samples=20, print_every=5)