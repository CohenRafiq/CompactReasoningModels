import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

class NonogramsDataset(Dataset):
    def __init__(self, data: Any, max_rows: int|None=None, flat=True, padding="square") -> None:
        self.X, self.y, self.metadata = self._load_data(data, max_rows)

    def _load_from_list_dictionary(self, data: list[dict], max_rows: int|None=None
                                   ) -> tuple[list[list[list[int]]], torch.Tensor, list[dict]]:
        clues =[]
        grids = []
        meta = []
        for entry in data[:max_rows] if max_rows is not None else data:
            if not isinstance(entry, dict):
                raise ValueError("Each entry in the list must be a dictionary.")
            clues.append([entry.pop("rows"), entry.pop("cols")])
            grids.append(entry.pop("grid"))
            meta.append(entry)

        return clues, torch.Tensor(grids, dtype=torch.float32), meta

    def _make_easy_metadata(self, clues: list[list[int]], grid: torch.Tensor) -> dict:
        grid_density = torch.sum(grid) / (grid.shape[0] * grid.shape[1])
        mean_clue_runs = np.mean([len(row) for row in clues[0]])
        return {
            "shape": grid.shape,
            "density": grid_density,
            "mean_clue_runs": mean_clue_runs
        }

    def _reformat_clue_grid_pair(self, clues, grid) -> tuple[list[list[list[int]]], torch.Tensor]:
        grid_tensor = torch.tensor(grid, dtype=torch.float32)
        if type(clues) is dict:
            new_clues = [[clues.pop("rows"), clues.pop("cols")]]
            meta = clues | self._make_easy_metadata(new_clues, grid_tensor)
        elif type(clues) is list or type(clues) is tuple and len(clues) == 2:
            new_clues = [list(clues)]
            meta = self._make_easy_metadata(new_clues, grid_tensor)
        else:
            raise ValueError("Invalid clues format.")
        return new_clues, grid_tensor, meta

    def _load_data(self, data: list | tuple | str, max_rows: int | None=None) -> Any:
        if type(data) is str:        # load from file
            pass
        elif type(data) is list | tuple and len(data) == 2:         # split X, y
            clues, grids, meta = [], [], []
            for clue_entry, grid_entry in data[:max_rows] if max_rows is not None else data:
                new_clues, grid_tensor, meta_entry = self._reformat_clue_grid_pair(clue_entry, grid_entry)
                clues.append(new_clues)
                grids.append(grid_tensor)
                meta.append(meta_entry)
            return clues, torch.stack(grids), meta
        else:
            clues, grids, meta = [], [], []
            for entry in data[:max_rows] if max_rows is not None else data:
                if isinstance(entry, dict):
                    clues.append([entry.pop("rows"), entry.pop("cols")])
                    grids.append(entry.pop("grid"))
                    meta.append(entry)
                elif (isinstance(entry, tuple) or 
                      isinstance(entry, list)) and len(entry) == 2:
                    if isinstance(entry[0], list):
                        clues.append(entry[0])
                    elif isinstance(entry[0], dict):
                        clues.append([entry[0].pop("rows"), entry[0].pop("cols")])
                    elif isinstance(entry[0], tuple):
                        clues.append(list(entry[0]))
                    else:
                        raise ValueError("Invalid clues format.")
                    grids.append(entry[1])
                    meta.append(self._make_easy_metadata(clues[-1], grids[-1]))
                else:
                    raise ValueError("Invalid data format.")
            return clues, torch.Tensor(grids, dtype=torch.float32), meta


    def _load_data(self, data: Any, max_rows: int | None=None) -> Any:
        datatype = type(data)
        if datatype is list and all(isinstance(entry, dict) for entry in data):
            return self._load_from_list_dictionary(data, max_rows)
        elif datatype is list | tuple and len(data) == 2:
            clues = data[0]
            grids = data[1]
            return data[0], data[1], None
        match type(data):
            case tuple():
                raise NotImplementedError("Loading from tuple is not implemented yet.")
            case list():
                if all(isinstance(entry, dict) for entry in data):
                    return self._load_from_list_dictionary(data, max_rows)
                
            case np.ndarray():
                raise NotImplementedError("Loading from numpy array is not implemented yet.")
            case torch.Tensor():
                raise NotImplementedError("Loading from torch tensor is not implemented yet.")
            case str():
                match = re.search(r'\.(jsonl|parquet|npy)$', data)
                if match:
                    data_type = match.group(1)
                else:
                    raise ValueError("Invalid file format.")
            case _:
                raise ValueError("Invalid data format.")

        return data_type

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
