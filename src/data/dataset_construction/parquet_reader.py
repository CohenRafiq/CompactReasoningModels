import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from src.data.puzzle_dataset import PuzzleDataset


class ParquetReader:
    def __init__(self, parquet_file: str, random_seed: int = 42, target_shape: tuple[int, int] | None = None):
        self.parquet_file = parquet_file
        self.target_shape = target_shape
        self.random_seed = random_seed

        df = pd.read_parquet(parquet_file, engine="pyarrow")

        df["row_clues"] = [
            self.convert_clue(rc, max_len=(gw + 1) // 2)
            for rc, gw in zip(df["row_clues"], df["grid_width"])
        ]
        df["col_clues"] = [
            self.convert_clue(cc, max_len=(gh + 1) // 2)
            for cc, gh in zip(df["col_clues"], df["grid_height"])
        ]

        X = torch.from_numpy(np.stack(
            [np.stack([rc, cc]) for rc, cc in zip(df["row_clues"], df["col_clues"])],
            axis=0
        ))

        y = torch.stack([
            torch.from_numpy(sol.astype(np.int8)).float()
            if hasattr(sol, "dtype") and sol.dtype != object else
            torch.from_numpy(np.array(sol.tolist(), dtype=np.int8)).float()
            for sol in df["solution"]
        ])

        df["X"] = list(X)
        df["y"] = list(y)

        self.dataframe = df

    def convert_clue(self, clue, max_len: int) -> np.ndarray:
        if isinstance(clue, np.ndarray):
            clue = clue.tolist()

        return np.array([
            list(inner) + [0] * max(0, max_len - len(inner))
            for inner in clue
        ], dtype=np.float32)

    def _apply_query(self, query: str | pd.Series | None) -> pd.DataFrame:
        if query is None:
            filtered_df = self.dataframe
        elif isinstance(query, str):
            filtered_df = self.dataframe.query(query)
        else:
            mask = query.reindex(self.dataframe.index, fill_value=False) if isinstance(query, pd.Series) else query
            filtered_df = self.dataframe[mask]

        if len(filtered_df) == 0:
            raise ValueError("The provided query resulted in an empty dataset.")
        return filtered_df

    def _split_indices(self, n: int, train_ratio: float, random_seed: int | None) -> tuple[list[int], list[int]]:
        train_size = int(n * train_ratio)
        test_size = n - train_size
        seed = self.random_seed if random_seed is None else random_seed
        train_idx, test_idx = random_split(
            range(n), [train_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )
        return list(train_idx), list(test_idx)

    def _build_dataset(self, df: pd.DataFrame, flat: bool) -> PuzzleDataset:
        X = torch.stack(df["X"].tolist())
        y = torch.stack(df["y"].tolist())

        target_shape = self.target_shape or (
            int(df.iloc[0]["grid_height"]),
            int(df.iloc[0]["grid_width"])
        )

        dataset = PuzzleDataset(X, y, target_shape=target_shape)
        if flat:
            dataset.flatten()
        if len(target_shape) == 3:
            dataset.y = dataset.y.long()
        return dataset

    def create_dataloaders(
        self,
        train_ratio: float = 0.8,
        batch_size: int = 32,
        query: str | pd.Series | None = None,
        flat: bool = True,
        random_seed: int | None = None,
    ):
        filtered_df = self._apply_query(query)
        train_idx, test_idx = self._split_indices(len(filtered_df), train_ratio, random_seed)

        train_df = filtered_df.iloc[train_idx].reset_index(drop=True)
        test_df = filtered_df.iloc[test_idx].reset_index(drop=True)

        train_dataset = self._build_dataset(train_df, flat)
        test_dataset = self._build_dataset(test_df, flat)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, train_df, test_df

    def create_single_dataloader(
        self,
        batch_size: int = 32,
        query: str | pd.Series | None = None,
        flat: bool = True,
        shuffle: bool = True,
        split: str = "all",
        train_ratio: float = 0.8,
        random_seed: int | None = None,
    ):
        filtered_df = self._apply_query(query)

        if split != "all":
            train_idx, test_idx = self._split_indices(len(filtered_df), train_ratio, random_seed)
            idx = train_idx if split == "train" else test_idx
            filtered_df = filtered_df.iloc[idx]

        filtered_df = filtered_df.reset_index(drop=True)
        dataset = self._build_dataset(filtered_df, flat)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return data_loader, filtered_df

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.iloc[idx]

    def get_intermediate_grids(self, idx: int) -> list[np.ndarray]:
        grids = self.dataframe.iloc[idx]["intermediate_solutions"]
        return [np.array(g.tolist() if hasattr(g, "tolist") else g, dtype=np.int8) for g in grids]

    def get_intermediate_methods(self, idx: int) -> list[str]:
        return list(self.dataframe.iloc[idx]["intermediate_methods"])

    def get_solution_steps(self, idx: int) -> list[tuple[np.ndarray, str]]:
        grids = self.get_intermediate_grids(idx)
        methods = self.get_intermediate_methods(idx)
        return list(zip(grids, methods))

    def get_solution_tensor(self, idx: int) -> torch.Tensor:
        sol = self.dataframe.iloc[idx]["solution"]
        return torch.tensor(sol.tolist() if hasattr(sol, "tolist") else sol, dtype=torch.float32)

    @property
    def requires_search_mask(self) -> pd.Series:
        return self.dataframe["requires_search"]
