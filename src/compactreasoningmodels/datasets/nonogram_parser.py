import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

import compactreasoningmodels.utils.puzzle_types as t
from compactreasoningmodels.utils.grid import derive_clues_from_grid


class NonogramParser:

    def __init__(self, max_size: int | None = None, batch_size: int = 256) -> None:
        self.max_size = max_size
        self.batch_size = batch_size

    def parse(self, data: str | Path | t.Dataset) -> t.ReformattedData:
        if isinstance(data, (str, Path)):
            return self._parse_file(data)
        elif isinstance(data, t.GridDataset):
            return self._parse_grid_dataset(data)
        elif isinstance(data, list):
            if len(data) == 2 and self._is_split_data(data):
                return self._parse_split_data(data)
            else:
                return self._parse_entry_list(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    # ------------------------------------------------------------------ #
    #  File loaders — lazy, streaming, metadata-preserving
    # ------------------------------------------------------------------ #

    def _parse_file(self, file_path: str | Path) -> t.ReformattedData:
        if isinstance(file_path, Path):
            file_path = str(file_path)
        if file_path.endswith(".jsonl"):
            return self._parse_jsonl(file_path)
        elif file_path.endswith(".parquet"):
            return self._parse_parquet(file_path)
        elif file_path.endswith(".npy"):
            return self._parse_npy(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    def _parse_jsonl(self, path: str) -> Iterator[t.ReformattedData]:
        """Lazy-load JSONL: stream entries, batch into ReformattedData chunks."""
        batch_clues, batch_grids, batch_meta = [], [], []
        count = 0

        with open(path) as f:
            for line in f:
                if self.max_size is not None and count >= self.max_size:
                    break

                raw = json.loads(line)

                # Preserve ALL metadata, only extract known fields
                meta = dict(raw)
                rows = meta.pop("rows")
                cols = meta.pop("cols")
                grid_raw = meta.pop("grid")

                batch_clues.append([rows, cols])
                batch_grids.append(self._grid_to_tensor(grid_raw))
                batch_meta.append(meta)
                count += 1

                # Yield full batches to bound memory
                if len(batch_clues) >= self.batch_size:
                    yield self._pack_batch(batch_clues, batch_grids, batch_meta)
                    batch_clues, batch_grids, batch_meta = [], [], []

        # Yield remainder
        if batch_clues:
            yield self._pack_batch(batch_clues, batch_grids, batch_meta)

    def _parse_parquet(self, path: str) -> Iterator[t.ReformattedData]:
        """Lazy-load Parquet: use pyarrow/parquet chunk iteration."""
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        count = 0

        for batch in pf.iter_batches(batch_size=self.batch_size):
            batch_clues, batch_grids, batch_meta = [], [], []

            # Convert to pandas for easier dict access (zero-copy where possible)
            df = batch.to_pandas()

            for _, row in df.iterrows():
                if self.max_size is not None and count >= self.max_size:
                    if batch_clues:
                        yield self._pack_batch(batch_clues, batch_grids, batch_meta)
                    return

                # Preserve ALL fields as metadata
                meta = row.to_dict()
                rows = meta.pop("rows")
                cols = meta.pop("cols")
                grid_raw = meta.pop("grid")

                batch_clues.append([rows, cols])
                batch_grids.append(self._grid_to_tensor(grid_raw))
                batch_meta.append(meta)
                count += 1

            if batch_clues:
                yield self._pack_batch(batch_clues, batch_grids, batch_meta)

    def _parse_npy(self, path: str) -> Iterator[t.ReformattedData]:
        arr = np.load(path, allow_pickle=True)

        # If it's a memmap or we want to chunk it
        total = min(len(arr), self.max_size) if self.max_size else len(arr)
        batch_sz = self.batch_size

        for start in range(0, total, batch_sz):
            end = min(start + batch_sz, total)
            batch_clues, batch_grids, batch_meta = [], [], []

            for item in arr[start:end]:
                # item assumed to be dict-like or structured array
                if isinstance(item, np.void):
                    item = {name: item[name] for name in item.dtype.names}

                meta = dict(item)
                rows = meta.pop("rows")
                cols = meta.pop("cols")
                grid_raw = meta.pop("grid")

                batch_clues.append([rows, cols])
                batch_grids.append(self._grid_to_tensor(grid_raw))
                batch_meta.append(meta)

            yield self._pack_batch(batch_clues, batch_grids, batch_meta)

    # ------------------------------------------------------------------ #
    #  In-memory parsers
    # ------------------------------------------------------------------ #

    def _parse_grid_dataset(self, data: t.GridDataset) -> t.ReformattedData:
        grids = self._normalize_grids(data)
        if self.max_size:
            grids = grids[:self.max_size]
        clues = [derive_clues_from_grid(g) for g in grids]
        meta = [self._make_metadata(c, g) for c, g in zip(clues, grids)]
        return self._pack_batch(clues, grids, meta)

    def _parse_entry_list(self, data: list[t.Entry]) -> t.ReformattedData:
        if self.max_size:
            data = data[:self.max_size]

        clues, grids, meta = [], [], []
        for entry in data:
            c, g, m = self._parse_single_entry(entry)
            clues.append(c)
            grids.append(g)
            meta.append(m)
        return self._pack_batch(clues, grids, meta)

    def _parse_split_data(self, data: t.SplitData) -> t.ReformattedData:
        clues_list, grids_raw = data
        if self.max_size:
            clues_list = clues_list[:self.max_size]
            grids_raw = grids_raw[:self.max_size]

        grids = [self._grid_to_tensor(g) for g in grids_raw]
        meta = [self._make_metadata(c, g) for c, g in zip(clues_list, grids)]
        return self._pack_batch(clues_list, grids, meta)

    def _parse_single_entry(self, entry: t.Entry) -> tuple[t.Clues, torch.Tensor, dict]:
        if isinstance(entry, dict):
            meta = dict(entry)
            rows = meta.pop("rows")
            cols = meta.pop("cols")
            grid = self._grid_to_tensor(meta.pop("grid"))
            return [rows, cols], grid, meta

        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            clues, grid_raw = entry
            grid = self._grid_to_tensor(grid_raw)
            meta = self._make_metadata(clues, grid)
            return clues, grid, meta

        else:
            raise ValueError(f"Entry must be EntryDict or (clues, grid) pair. Got: {type(entry)}")

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    def _pack_batch(
        self,
        clues: list[t.Clues],
        grids: list[torch.Tensor],
        meta: list[dict],
    ) -> t.ReformattedData:
        return clues, torch.stack(grids), meta

    def _is_split_data(self, data: list) -> bool:
        if len(data) != 2:
            return False
        first, second = data
        return (
            isinstance(first, list)
            and len(first) > 0
            and isinstance(first[0], list)
            and isinstance(first[0][0], list)
            and isinstance(second, (list, np.ndarray, torch.Tensor))
        )

    def _normalize_grids(self, data: t.GridDataset) -> list[torch.Tensor]:
        if isinstance(data, torch.Tensor):
            if data.dim() == 2:
                return [data]
            elif data.dim() == 3:
                return list(data.unbind(0))
            else:
                raise ValueError(f"Expected 2-D or 3-D tensor, got {data.dim()}-D")
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                return [torch.from_numpy(data)]
            elif data.ndim == 3:
                return [torch.from_numpy(g) for g in data]
            else:
                raise ValueError(f"Expected 2-D or 3-D ndarray, got {data.ndim}-D")
        elif isinstance(data, list):
            return [self._grid_to_tensor(g) for g in data]
        else:
            raise ValueError(f"Unsupported grid type: {type(data)}")

    def _grid_to_tensor(self, grid: t.Grid) -> torch.Tensor:
        if isinstance(grid, torch.Tensor):
            return grid if grid.dim() == 2 else grid.squeeze()
        elif isinstance(grid, np.ndarray):
            return torch.from_numpy(grid).squeeze()
        elif isinstance(grid, list):
            return torch.tensor(grid, dtype=torch.float32)
        else:
            raise ValueError(f"Cannot convert {type(grid)} to tensor")

    def _make_metadata(self, clues: t.Clues, grid: torch.Tensor) -> dict:
        grid_density = torch.sum(grid) / (grid.shape[0] * grid.shape[1])
        mean_clue_runs = np.mean([len(row) for row in clues[0]] +
                                 [len(col) for col in clues[1]])
        return {
            "shape": grid.shape,
            "density": grid_density,
            "mean_clue_runs": mean_clue_runs
        }
