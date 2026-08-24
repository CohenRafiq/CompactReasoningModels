from pathlib import Path

import numpy as np
import pytest

from compactreasoningmodels.datasets.jsonl import JsonlDataset
from compactreasoningmodels.solving_traces.arc_consistency import ArcConsistency
from compactreasoningmodels.solving_traces.base import SolvingTrace
from compactreasoningmodels.solving_traces.discrete_genetic import DiscreteGeneticAlgorithm
from compactreasoningmodels.solving_traces.global_min_violations import GlobalMinViolations
from compactreasoningmodels.solving_traces.local_min_violations import LocalMinViolations
from compactreasoningmodels.solving_traces.similarity_measures import (
    MAE,
    MSE,
    HuberLoss,
    MeanCosineSimilarity,
    PearsonCorrelation,
    SSIMSimilarity,
    WassersteinSimilarity,
    compare_batches,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TINY_JSONL = FIXTURES_DIR / "nonogram_5x5_tiny.jsonl"

GRID_SIZE = 5
MAX_CLUE_LEN = 3
N_RECORDS = 5

# All similarity measures share one convention: 1 = identical, 0 = maximal
# difference, values in [0, 1].
METRICS = {
    "mse": MSE(),
    "mae": MAE(),
    "huber": HuberLoss(),
    "wasserstein": WassersteinSimilarity(),
    "mean_cosine": MeanCosineSimilarity(),
    "pearson": PearsonCorrelation(),
    "ssim": SSIMSimilarity(),
}

SOLVER_FACTORIES = {
    "arc_consistency": lambda clues, shape, initial_grid=None: ArcConsistency(
        clues, shape, initial_grid
    ),
    "global_min_violations": lambda clues, shape, initial_grid=None: GlobalMinViolations(
        clues, shape, initial_grid
    ),
    "local_min_violations": lambda clues, shape, initial_grid=None: LocalMinViolations(
        clues, shape, initial_grid
    ),
    "discrete_genetic": lambda clues, shape, initial_grid=None: DiscreteGeneticAlgorithm(
        clues, shape, initial_grid, population_size=50, concurrent_samples=50
    ),
}


def _make_solver(name: str, clues: np.ndarray, shape: tuple[int, int], seed: int = 0,
                 initial_grid: np.ndarray | None = None):
    np.random.seed(seed)  # legacy global RNG is what the solvers consume
    solver: SolvingTrace = SOLVER_FACTORIES[name](clues, shape, initial_grid)
    return solver


def _initial_grid(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.25, 0.75, size=(GRID_SIZE, GRID_SIZE))


@pytest.fixture(scope="module")
def dataset() -> JsonlDataset:
    return JsonlDataset(TINY_JSONL, target_shape=(GRID_SIZE, GRID_SIZE))


@pytest.fixture(scope="module")
def clues_for(dataset: JsonlDataset):
    def _clues(idx: int) -> np.ndarray:
        x, _ = dataset[idx]
        return x.numpy().reshape(2, GRID_SIZE, MAX_CLUE_LEN)

    return _clues


@pytest.fixture(scope="module")
def fast_heatmaps(clues_for) -> dict[str, np.ndarray]:
    start = _initial_grid()
    return {
        name: _make_solver(name, clues_for(0), (GRID_SIZE, GRID_SIZE)).heatmap_step(start)
        for name in ("arc_consistency", "global_min_violations")
    }


def test_fixture_dataset_loads(dataset: JsonlDataset):
    assert len(dataset) == N_RECORDS
    x, y = dataset[0]
    assert x.shape == (2 * GRID_SIZE * MAX_CLUE_LEN,)
    assert y.shape == (GRID_SIZE, GRID_SIZE)
    assert set(np.unique(y.numpy())).issubset({0.0, 1.0})


@pytest.mark.parametrize("name", SOLVER_FACTORIES)
def test_heatmap_step_output_valid(name, clues_for):
    solver = _make_solver(name, clues_for(0), (GRID_SIZE, GRID_SIZE))
    heatmap = solver.heatmap_step(_initial_grid())

    assert heatmap.shape == (GRID_SIZE, GRID_SIZE)
    assert np.isfinite(heatmap).all()
    assert (heatmap >= 0.0).all() and (heatmap <= 1.0).all()


@pytest.mark.parametrize("name", SOLVER_FACTORIES)
def test_deterministic_given_same_start(name, clues_for):
    clues = clues_for(0)
    start = _initial_grid()

    first = _make_solver(name, clues, (GRID_SIZE, GRID_SIZE), seed=7).heatmap_step(start)
    second = _make_solver(name, clues, (GRID_SIZE, GRID_SIZE), seed=7).heatmap_step(start)

    np.testing.assert_allclose(first, second)


@pytest.mark.parametrize("name", SOLVER_FACTORIES)
def test_try_solve_trace_structure(name, clues_for):
    start = _initial_grid()
    solver = _make_solver(name, clues_for(1), (GRID_SIZE, GRID_SIZE), initial_grid=start)

    solved, traces = solver.try_solve(max_steps=2)

    assert isinstance(solved, bool)
    assert len(traces) >= 2
    np.testing.assert_array_equal(traces[0], start)
    assert all(t.shape == (GRID_SIZE, GRID_SIZE) for t in traces)


def test_arc_consistency_solves_unique_puzzle():
    # Unique solution [[1, 1], [1, 0]]: rows [2],[1]; cols [2],[1].
    clues = np.array([[[2.0], [1.0]], [[2.0], [1.0]]])
    expected = np.array([[1.0, 1.0], [1.0, 0.0]])

    solver = ArcConsistency(clues, (2, 2), initial_grid=np.full((2, 2), 0.5))
    solved, traces = solver.try_solve(max_steps=10)

    assert solved
    np.testing.assert_allclose(traces[-1], expected, atol=1e-9)


@pytest.mark.parametrize("metric_name", METRICS)
def test_cross_solver_similarity_matrix(metric_name, fast_heatmaps):
    metric = METRICS[metric_name]
    names = list(fast_heatmaps)
    batch = np.stack([fast_heatmaps[name] for name in names])

    matrix = compare_batches(batch, batch, metric)

    assert matrix.shape == (len(names), len(names))
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-6)

    pairwise = metric(fast_heatmaps[names[0]], fast_heatmaps[names[1]])
    assert isinstance(pairwise, float)
    assert np.isfinite(pairwise)


@pytest.mark.parametrize(
    "solver_name", ["arc_consistency", "global_min_violations"]
)
@pytest.mark.parametrize("metric_name", list(METRICS))
def test_noisy_trace_farther_scores_lower(solver_name, metric_name, fast_heatmaps):
    base = fast_heatmaps[solver_name]
    rng = np.random.default_rng(123)
    small_noise = base + rng.normal(0.0, 0.02, size=base.shape)
    large_noise = base + rng.normal(0.0, 0.30, size=base.shape)

    metric = METRICS[metric_name]
    assert metric(base, small_noise) > metric(base, large_noise)
