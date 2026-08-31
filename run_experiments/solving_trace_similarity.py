import torch
import numpy as np
import itertools
import pandas as pd

from compactreasoningmodels.solving_traces.discrete_genetic import DiscreteGeneticAlgorithm
from compactreasoningmodels.solving_traces.global_min_violations import GlobalMinViolations
from compactreasoningmodels.solving_traces.local_min_violations import LocalMinViolations
from compactreasoningmodels.solving_traces.arc_consistency import ArcConsistency
from compactreasoningmodels.solving_traces.model_solver import ModelSolver
from compactreasoningmodels.solving_traces.similarity_measures import MSE, PearsonCorrelation, SSIMSimilarity
from compactreasoningmodels.datasets.nonogram_dataset import NonogramDataset

SOLVERS = {
    "arc_consistency": ArcConsistency, 
    "local_min_violations": LocalMinViolations, 
    "global_min_violations": GlobalMinViolations, 
    "discrete_genetic": DiscreteGeneticAlgorithm,
    "model_solver": ModelSolver
}

METRICS = {
    "MSE": MSE(),
    "Pearson Correlation": PearsonCorrelation(),
    "SSIM": SSIMSimilarity()
}

def print_distribution_ascii(data, solver_name, bins=20, width=50):
    values = np.array(data).flatten()
    values = values[(values >= 0) & (values <= 1)]
    hist, bin_edges = np.histogram(values, bins=bins, range=(0, 1))
    max_count = hist.max()
    print(f"\n{solver_name}")
    print("-" * (width + 20))
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
    
    print("-" * (width + 20))

def add_noise_to_grid(grid: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    noise = np.random.uniform(-noise_level, noise_level, size=grid.shape)
    noisy_grid = (1-noise_level) * grid + noise
    return noisy_grid

def main(number_samples=100, print_every=None, noise_level=0.1):
    dataset = NonogramDataset("data/raw/nonogram_5x5_small.jsonl")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    heatmaps = {name: [] for name in SOLVERS.keys()}
    
    for i, (input_tensor, target_tensor, _, _) in enumerate(dataloader):
        if i >= number_samples:
            break
        target = target_tensor.squeeze(0).numpy()
        initial_grid = np.random.uniform(0.1, 0.9, size=target.shape)

        solvers = {name: solver_cls(input_tensor, target.shape, initial_grid=initial_grid)
                   for name, solver_cls in SOLVERS.items()}
        
        for name, solver in solvers.items():
            hm = solver.heatmap(num_steps=1)
            heatmaps[name].append(hm)

        if print_every is not None and (i + 1) % print_every == 0:
            print(f"Processed {i + 1}/{number_samples} samples.")

    solver_names = list(SOLVERS.keys())


    matrices = {
        name: pd.DataFrame(index=solver_names, columns=solver_names, dtype=float)
        for name in METRICS
    }

    clean_grids = {solver: np.array(heatmaps[solver]) for solver in solver_names}
    noisy_grids = {
        solver: add_noise_to_grid(clean_grids[solver], noise_level=noise_level)
        for solver in solver_names
    }

    for noisy, clean in itertools.product(solver_names, repeat=2):
        for metric_name, metric in METRICS.items():
            result = np.mean(metric(noisy_grids[noisy], clean_grids[clean]))
            matrices[metric_name].loc[noisy, clean] = result

    for metric_name, matrix in matrices.items():
        matrix = matrix.round(4)
        matrices[metric_name] = matrix
        print(f"{metric_name} Matrix (rows=noisy, cols=clean):")
        print(matrix)
        print()

    for solver in solver_names:
        print_distribution_ascii(heatmaps[solver], solver)

if __name__ == "__main__":
    main(number_samples=20, print_every=5, noise_level=0.1)