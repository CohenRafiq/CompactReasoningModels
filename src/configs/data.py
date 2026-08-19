from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PuzzleDatasetConfig:
    _target_: str = "src.data.puzzle_dataset.PuzzleDataset"
    input_path: str = "processed/nonogram_5x5_x.npy"
    target_path: Optional[str] = "processed/nonogram_5x5_y.npy"
    target_shape: Optional[List[int]] = None
    split_categories: bool = False


@dataclass
class ParquetPuzzleDatasetConfig:
    _target_: str = "src.data.parquet_puzzle_dataset.ParquetPuzzleDataset"
    parquet_path: str = "raw/nng_5x5_small.parquet"
    target_shape: Optional[List[int]] = None
    filter_query: Optional[str] = None
    split_categories: bool = False
