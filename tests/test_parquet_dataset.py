from pathlib import Path

import pytest
import torch

from compactreasoningmodels.data_generation.parquet_reader import ParquetReader
from compactreasoningmodels.datasets.parquet import ParquetPuzzleDataset

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def small_parquet() -> str:
    return str(FIXTURES_DIR / "nng_5x5_tiny.parquet")


def test_dataset_shapes(small_parquet):
    dataset = ParquetPuzzleDataset(small_parquet)
    X, y = dataset[0]
    assert X.shape == (2, 5, 3)
    assert y.shape == (5, 5)
    assert dataset.X.shape[0] == len(dataset)


def test_target_shape_match(small_parquet):
    dataset = ParquetPuzzleDataset(small_parquet, target_shape=(5, 5))
    assert dataset.y.shape[1:] == (5, 5)


def test_split_categories(small_parquet):
    dataset = ParquetPuzzleDataset(small_parquet, split_categories=True)
    X, y = dataset[0]
    assert y.shape[-1] == 2
    assert torch.allclose(y[..., 0] + y[..., 1], torch.ones_like(y[..., 0]))


def test_empty_query_raises(small_parquet):
    reader = ParquetReader(small_parquet)
    with pytest.raises(ValueError):
        reader._apply_query("puzzle_id < 0")


def test_reader_dataloaders(small_parquet):
    reader = ParquetReader(small_parquet)
    train_loader, test_loader, _, _ = reader.create_dataloaders(
        train_ratio=0.7, batch_size=16, random_seed=1
    )
    X, y = next(iter(train_loader))
    assert X.dim() >= 2
    assert X.shape[0] <= 16
