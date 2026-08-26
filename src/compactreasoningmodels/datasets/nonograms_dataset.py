import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class Metadata:
    pass

class NonogramsDataset(Dataset):
    def __init__(self, data: Any, max_rows: int|None=None, flat=True, padding="square") -> None:
        split, data_type = self._get_data_format(data)
        if split:
            self.X, self.y, self.metadata = self._load_split_data(data, data_type, max_rows)
        else:
            self.X, self.y, self.metadata = self._load_data(data, data_type, max_rows)

    def _get_data_format(self, data: Any) -> tuple[int, ...]:
        match type(data):
            case tuple() | list():
                if len(data) != 2 or type(data[0]) != type(data[1]):
                    raise ValueError("Invalid data format.")
                split = True
                inner_data = data[0]
            case np.ndarray() | torch.Tensor() | str():
                split = False
                inner_data = data
            case _:
                raise ValueError("Invalid data format.")

        match type(inner_data):
            case np.ndarray():
                data_type = "ndarray"
            case torch.Tensor():
                data_type = "tensor"
            case str():
                match = re.search(r'\.(jsonl|parquet|npy)$', inner_data)
                if match:
                    data_type = match.group(1)
                else:
                    raise ValueError("Invalid data format.")
            case _:
                raise ValueError("Invalid data format.")

        return split, data_type

    def _load_split_data(self, data: Any, data_type: str, max_rows: int|None=None) -> tuple[Any, Any]:
        pass

    def _load_data(self, data: Any, data_type: str, max_rows: int | None=None) -> Any:
        pass

    def _reshape_data(self, data: torch.tensor) -> torch.tensor:
        match self.data_shape:
            case "square":
                pass
            case "max":
                pass
            case "transpose":
                pass
            case "sequence":
                pass
            case "sequence_no_padding":
                pass
            case _:
                raise ValueError(f"Invalid data_shape: {self.data_shape}")

    def __getitem__(self, idx: int):
        return (self.X[idx], self.y[idx]) if self.y is not None else (self.X[idx],)

    def __len__(self) -> int:
        return len(self.X)
