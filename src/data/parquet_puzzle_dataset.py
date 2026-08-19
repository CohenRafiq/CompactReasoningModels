from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.data.base_dataset import BaseDataset


class ParquetPuzzleDataset(BaseDataset):
    """
    A Dataset that loads from Parquet files with full metadata access.

    Extends BaseDataset to support Parquet files and expose additional
    metadata like intermediate solutions, solving methods, and difficulty metrics.

    Parquet columns:
        - puzzle_id, row_clues, col_clues, solution
        - intermediate_solutions, intermediate_methods
        - grid_density, grid_height, grid_width, steps
        - requires_search, one_step_rounding
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        target_shape: Optional[Tuple[int, ...]] = None,
        filter_query: Optional[str] = None,
        split_categories: bool = False,
    ):
        super().__init__()

        import os
        base_dir = Path(os.environ.get("DATA_DIR", "data"))
        self.parquet_path = base_dir / parquet_path
        self.filter_query = filter_query

        self._df = pd.read_parquet(self.parquet_path, engine="pyarrow")

        if filter_query:
            self._df = self._df.query(filter_query)
            if len(self._df) == 0:
                raise ValueError(f"Filter query '{filter_query}' resulted in empty dataset")

        self._prepare_tensors(target_shape)

        if split_categories:
            self._split_categories()

    def _prepare_tensors(self, target_shape: Optional[Tuple[int, ...]]):
        df = self._df
        
        grid_heights = df["grid_height"].values
        grid_widths = df["grid_width"].values
        
        max_h = int(grid_heights.max())
        max_w = int(grid_widths.max())
        max_clue_len_rows = max((gw + 1) // 2 for gw in grid_widths)
        max_clue_len_cols = max((gh + 1) // 2 for gh in grid_heights)
        max_clue_len = max(max_clue_len_rows, max_clue_len_cols)
        
        n = len(df)
        grid_size = max(max_h, max_w)
        
        self.X = torch.zeros((n, 2, grid_size, max_clue_len), dtype=torch.float32)
        
        for i, (_, row) in enumerate(df.iterrows()):
            row_clues = row["row_clues"]
            col_clues = row["col_clues"]
            
            if isinstance(row_clues, np.ndarray):
                row_clues = row_clues.tolist()
            if isinstance(col_clues, np.ndarray):
                col_clues = col_clues.tolist()
            
            for r_idx, clue in enumerate(row_clues):
                clue_list = list(clue) if not isinstance(clue, list) else clue
                self.X[i, 0, r_idx, :len(clue_list)] = torch.tensor(clue_list, dtype=torch.float32)
            
            for c_idx, clue in enumerate(col_clues):
                clue_list = list(clue) if not isinstance(clue, list) else clue
                self.X[i, 1, c_idx, :len(clue_list)] = torch.tensor(clue_list, dtype=torch.float32)
        
        solutions = df["solution"].tolist()
        self.y = torch.zeros((n, max_h, max_w), dtype=torch.float32)
        for i, sol in enumerate(solutions):
            if isinstance(sol, np.ndarray):
                sol_array = np.stack(sol.tolist() if sol.dtype == object else [r for r in sol]).astype(np.float32)
            else:
                sol_array = np.array(sol, dtype=np.float32)
            h, w = sol_array.shape
            self.y[i, :h, :w] = torch.from_numpy(sol_array)
        
        self._input_shape = tuple(self.X.shape[1:])
        self._target_shape = target_shape if target_shape else (max_h, max_w)
        
        self._grid_heights = grid_heights
        self._grid_widths = grid_widths

    def _split_categories(self):
        self.y = torch.stack([self.y, 1 - self.y], dim=-1)
        self._target_shape = self.y.shape[1:]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def intermediate_solutions(self) -> List[np.ndarray]:
        grids = self._df["intermediate_solutions"].values
        return [
            np.array(g.tolist() if hasattr(g, "tolist") else g, dtype=np.float64)
            for g in grids
        ]

    @property
    def intermediate_methods(self) -> List[List[str]]:
        methods = self._df["intermediate_methods"].values
        return [list(m) if not isinstance(m, list) else m for m in methods]

    @property
    def requires_search(self) -> np.ndarray:
        return self._df["requires_search"].values

    @property
    def grid_density(self) -> np.ndarray:
        return self._df["grid_density"].values

    @property
    def steps(self) -> np.ndarray:
        return self._df["steps"].values

    @property
    def one_step_rounding(self) -> np.ndarray:
        return self._df["one_step_rounding"].values

    @property
    def puzzle_ids(self) -> np.ndarray:
        return self._df["puzzle_id"].values

    def get_intermediate_grids(self, idx: int) -> List[np.ndarray]:
        grids = self._df.iloc[idx]["intermediate_solutions"]
        return [np.array(g.tolist() if hasattr(g, "tolist") else g, dtype=np.float64) for g in grids]

    def get_intermediate_methods(self, idx: int) -> List[str]:
        methods = self._df.iloc[idx]["intermediate_methods"]
        return list(methods) if not isinstance(methods, list) else methods

    def get_solution_steps(self, idx: int) -> List[Tuple[np.ndarray, str]]:
        grids = self.get_intermediate_grids(idx)
        methods = self.get_intermediate_methods(idx)
        return list(zip(grids, methods))

    def filter_by_search(self, requires_search: bool = True) -> "ParquetPuzzleDataset":
        mask = self._df["requires_search"] == requires_search
        new_df = self._df[mask].reset_index(drop=True)
        new_dataset = object.__new__(ParquetPuzzleDataset)
        BaseDataset.__init__(new_dataset)
        new_dataset.parquet_path = self.parquet_path
        new_dataset.filter_query = self.filter_query
        new_dataset._df = new_df
        new_dataset._prepare_tensors(None)
        return new_dataset

    def filter_by_query(self, query: str) -> "ParquetPuzzleDataset":
        new_df = self._df.query(query).reset_index(drop=True)
        new_dataset = object.__new__(ParquetPuzzleDataset)
        BaseDataset.__init__(new_dataset)
        new_dataset.parquet_path = self.parquet_path
        new_dataset.filter_query = self.filter_query
        new_dataset._df = new_df
        new_dataset._prepare_tensors(None)
        return new_dataset
