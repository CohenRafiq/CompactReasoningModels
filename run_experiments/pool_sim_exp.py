import itertools
import numpy as np
from torch.utils.data import DataLoader

from compactreasoningmodels.solving_traces import ArcConsistency, DiscreteGeneticAlgorithm, LocalMinViolations, GlobalMinViolations, ModelSolver
from compactreasoningmodels.datasets.nonogram_dataset import NonogramDataset
from compactreasoningmodels.trace_similarity.step_alignment import LinearGradientPooling, PCAProcrustesAlignment, MeanPoolingAlignment

GRID_SHAPE = (5, 5)

SOLVER_CONFIGS = [
    ("ArcConsistency hit_rate=0.5", ArcConsistency, dict(hit_rate=0.5), 8),
    ("ArcConsistency hit_rate=0.7", ArcConsistency, dict(hit_rate=0.7), 6),
    ("ArcConsistency hit_rate=1.0", ArcConsistency, dict(hit_rate=1.0), 4),
    ("LocalMinViolations", LocalMinViolations, dict(hit_rate=0.7), 20),
    ("LocalMinViolations2", LocalMinViolations, dict(hit_rate=0.7), 20),
    ("GlobalMinViolations", GlobalMinViolations, dict(), 15),
    ("ModelSolver", ModelSolver, dict(), 8),
    ("DiscreteGeneticAlgorithm population_size=200", DiscreteGeneticAlgorithm, dict(population_size=200), 40),
    ("DiscreteGeneticAlgorithm population_size=50", DiscreteGeneticAlgorithm, dict(population_size=50), 40),
]


def run_solver(solver_cls, input_tensor, kwargs, n_steps):
    """Run a solver for n_steps, returning the list of heatmaps produced."""
    solver = solver_cls(input_tensor, GRID_SHAPE, **kwargs)
    heatmap = solver.initial_grid
    trace = []
    for _ in range(n_steps):
        heatmap = solver.heatmap_step(heatmap)
        trace.append(heatmap)
    return trace


def main():
    dataset = NonogramDataset("data/raw/nonogram_5x5.jsonl", max_size=5)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    traces = {name: [] for name, *_ in SOLVER_CONFIGS}

    for input_tensor, target_grid, _, _ in dataloader:
        for name, solver_cls, kwargs, n_steps in SOLVER_CONFIGS:
            traces[name].append(run_solver(solver_cls, input_tensor, kwargs, n_steps))

    alignment = PCAProcrustesAlignment(K=20, n_components=5)
    for (name1, name2) in itertools.combinations(traces, 2):
        heatmaps1 = np.array(traces[name1])[:, [0, 2, 3, 1]]
        heatmaps2 = np.array(traces[name2])[:, [0, 2, 3, 1]]

        heatmaps1 = heatmaps1.reshape(heatmaps1.shape[0], -1, heatmaps1.shape[-1])
        heatmaps2 = heatmaps2.reshape(heatmaps2.shape[0], -1, heatmaps2.shape[-1])
        trace1, trace2 = alignment.align(heatmaps1, heatmaps2)
        mse = np.mean((trace1 - trace2) ** 2)

        print(f"Comparing {name1} and {name2}:")
        print(f"    MSE = {mse:.4f}")


if __name__ == "__main__":
    main()