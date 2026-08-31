import numpy as np
import pytest

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

ALL_MEASURES = {
    "mse": MSE(),
    "mae": MAE(),
    "huber": HuberLoss(),
    "wasserstein": WassersteinSimilarity(),
    "mean_cosine": MeanCosineSimilarity(),
    "pearson": PearsonCorrelation(),
    "ssim": SSIMSimilarity(),
}


def _checkerboard(size: int = 8) -> np.ndarray:
    y, x = np.indices((size, size))
    return ((x + y) % 2).astype(np.float64)


def _half_filled(size: int = 8) -> np.ndarray:
    grid = np.zeros((size, size))
    grid[:, : size // 2] = 1.0
    return grid


def _quarter_filled(size: int = 8) -> np.ndarray:
    grid = np.zeros((size, size))
    grid[: size // 2, : size // 2] = 1.0
    return grid


@pytest.mark.parametrize("name", ALL_MEASURES)
def test_identical_grids_score_one(name):
    grid = _checkerboard()
    assert ALL_MEASURES[name](grid, grid.copy()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", ALL_MEASURES)
def test_different_grids_score_below_one(name):
    score = ALL_MEASURES[name](_half_filled(), _quarter_filled())
    assert score < 1.0 - 1e-6
    assert score >= 0.0


@pytest.mark.parametrize("name", ALL_MEASURES)
def test_scores_stay_within_unit_range(name):
    pairs = [
        (_half_filled(), _quarter_filled()),
        (_checkerboard(), 1.0 - _checkerboard()),
        (_half_filled(), _half_filled() + 0.5),
    ]
    for g1, g2 in pairs:
        single = float(ALL_MEASURES[name](g1, g2))
        assert 0.0 <= single <= 1.0
        batched = ALL_MEASURES[name](np.stack([g1, g2]), np.stack([g2, g1]))
        assert ((batched >= 0.0) & (batched <= 1.0)).all()


@pytest.mark.parametrize("name", ALL_MEASURES)
def test_shape_mismatch_raises(name):
    with pytest.raises(ValueError):
        ALL_MEASURES[name](_half_filled(8), _half_filled(9))


@pytest.mark.parametrize("name", ALL_MEASURES)
def test_below_two_dimensions_raises(name):
    line = np.zeros(8)
    with pytest.raises(ValueError):
        ALL_MEASURES[name](line, line.copy())


@pytest.mark.parametrize("name", ALL_MEASURES)
def test_batch_matches_single(name):
    measure = ALL_MEASURES[name]
    grids = [_checkerboard(), _half_filled(), _quarter_filled()]
    shifted = [grids[1], grids[2], grids[0]]

    batch = measure(np.stack(grids), np.stack(shifted))
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (3,)
    for i in range(3):
        assert batch[i] == pytest.approx(measure(grids[i], shifted[i]), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("name", ALL_MEASURES)
def test_single_input_returns_float(name):
    score = ALL_MEASURES[name](_half_filled(), _quarter_filled())
    assert isinstance(score, float)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MSE(data_range=0),
        lambda: MAE(data_range=-1),
        lambda: HuberLoss(delta=0),
        lambda: WassersteinSimilarity(data_range=0),
    ],
    ids=["mse", "mae", "huber_delta", "wasserstein"],
)
def test_invalid_parameters_raise(factory):
    with pytest.raises(ValueError):
        factory()


def test_mse_known_values():
    g1 = np.zeros((2, 2))
    g2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert MSE(data_range=1)(g1, g2) == pytest.approx(0.0)  # 7.5 clipped
    assert MSE(data_range=10)(g1, g2) == pytest.approx(1 - 7.5 / 100)


def test_mae_known_values():
    g1 = np.zeros((2, 2))
    g2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert MAE(data_range=1)(g1, g2) == pytest.approx(0.0)  # 2.5 clipped
    assert MAE(data_range=10)(g1, g2) == pytest.approx(1 - 2.5 / 10)


def test_huber_known_values():
    g1 = np.zeros((2, 2))
    g2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    # delta=1, data_range=4: max loss 3.5, mean loss 2.0
    assert HuberLoss(delta=1.0, data_range=4.0)(g1, g2) == pytest.approx(1 - 2.0 / 3.5)
    # delta=10, data_range=4: max loss 8.0, mean loss 3.75 (quadratic region)
    assert HuberLoss(delta=10.0, data_range=4.0)(g1, g2) == pytest.approx(1 - 3.75 / 8)


def test_pearson_absolute_flag():
    grid = _checkerboard()
    inverted = 1.0 - grid
    assert PearsonCorrelation(absolute=True)(grid, inverted) == pytest.approx(1.0)
    assert PearsonCorrelation(absolute=False)(grid, inverted) == pytest.approx(0.0)
    assert PearsonCorrelation(absolute=False)(grid, grid.copy()) == pytest.approx(1.0)


def test_mean_cosine_orthogonal_grids_score_zero():
    grid = _checkerboard()
    assert MeanCosineSimilarity()(grid, 1.0 - grid) == pytest.approx(0.0)


def test_wasserstein_known_values():
    zeros = np.zeros((4, 4))
    ones = np.ones((4, 4))
    assert WassersteinSimilarity(data_range=1)(zeros, ones) == pytest.approx(0.0)
    assert WassersteinSimilarity(data_range=2)(zeros, ones) == pytest.approx(0.5)
    assert WassersteinSimilarity(data_range=1)(zeros, zeros.copy()) == pytest.approx(1.0)


@pytest.mark.parametrize("name", ALL_MEASURES)
def test_noise_monotonicity(name):
    measure = ALL_MEASURES[name]
    base = _half_filled()
    rng = np.random.default_rng(0)
    small_noise = base + rng.normal(0.0, 0.02, size=base.shape)
    large_noise = base + rng.normal(0.0, 0.30, size=base.shape)

    assert measure(base, small_noise) > measure(base, large_noise)


def test_compare_batches_matrix_properties():
    grids = np.stack([_checkerboard(), _half_filled()])
    matrix = compare_batches(grids, grids, MSE())

    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-6)
    np.testing.assert_allclose(matrix, matrix.T)
    assert matrix[0, 1] < matrix[0, 0]
