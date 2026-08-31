import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

import compactreasoningmodels.utils.puzzle_types as t
from compactreasoningmodels.utils.grid import derive_clues_from_grid


class NonogramDataset:

    def __init__(self, data: str | Path | t.Dataset,
                 max_size: int | None = None,
                 batch_size: int = 256, padding: str = "positional") -> None:
        self.max_size = max_size
        self.batch_size = batch_size
        self.padding = padding

        self.X_raw, self.y_raw, self.meta = self._materialize(self.parse(data))
        self.meta = [m | self._make_metadata(c, g) for c, g, m in zip(self.X_raw, self.y_raw, self.meta)]
        max_dimension = self.y_raw.shape[-1]
        self.y = self.y_raw.flatten(start_dim=1)

        if padding == "positional":
            self.X, self.padding_mask = self._pad_positional(self.X_raw, max_dimension)
        elif padding == "sequence":
            self.X, self.padding_mask = self._pad_sequence(self.X_raw, max_dimension)
        else:
            raise ValueError(f"Unsupported padding type: {padding}")

    # ------------------------------------------------------------------ #
    #  Materialization: parse() may return either a single ReformattedData
    #  tuple (in-memory paths) or a generator of ReformattedData batches
    #  (streaming file paths). Normalize to a single materialized tuple.
    # ------------------------------------------------------------------ #

    def _materialize(self, parsed: t.ReformattedData | Iterator[t.ReformattedData]) -> t.ReformattedData:
        if isinstance(parsed, tuple):
            return parsed

        # It's a generator/iterator of batches: concatenate them.
        all_clues: list[t.Clues] = []
        all_grids: list[torch.Tensor] = []
        all_meta: list[dict] = []

        for clues_batch, grid_batch, meta_batch in parsed:
            all_clues.extend(clues_batch)
            all_grids.append(grid_batch)
            all_meta.extend(meta_batch)

        if not all_grids:
            raise ValueError("No data parsed: input produced zero entries.")

        return all_clues, torch.cat(all_grids, dim=0), all_meta

    def parse(self, data: str | Path | t.Dataset) -> t.ReformattedData | Iterator[t.ReformattedData]:
        if isinstance(data, (str, Path)):
            return self._parse_file(data)
        elif isinstance(data, (np.ndarray, torch.Tensor)):
            return self._parse_grid_dataset(data)
        elif isinstance(data, (list, tuple)):
            if len(data) == 2 and self._is_split_data(data):
                return self._parse_split_data(data)
            elif isinstance(data, tuple):
                raise ValueError(
                    f"Unsupported tuple data (expected 2-element SplitData): {data!r}"
                )
            elif len(data) == 0:
                raise ValueError("Cannot parse an empty dataset.")
            elif self._is_grid_list(data):
                return self._parse_grid_dataset(data)
            else:
                return self._parse_entry_list(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    # ------------------------------------------------------------------ #
    #  File loaders — lazy, streaming, metadata-preserving
    # ------------------------------------------------------------------ #

    def _parse_file(self, file_path: str | Path) -> Iterator[t.ReformattedData]:
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

                meta = dict(raw)
                rows = meta.pop("rows", meta.pop("row_clues", []))
                cols = meta.pop("cols", meta.pop("col_clues", []))
                grid_raw = meta.pop("grid", meta.pop("solution", []))

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
                rows = meta.pop("rows", meta.pop("row_clues", []))
                cols = meta.pop("cols", meta.pop("col_clues", []))
                grid_raw = meta.pop("grid", meta.pop("solution", []))

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
                rows = meta.pop("rows", meta.pop("row_clues", []))
                cols = meta.pop("cols", meta.pop("col_clues", []))
                grid_raw = meta.pop("grid", meta.pop("solution", []))

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
        list_grids = [g.tolist() for g in grids]
        clues = [list(derive_clues_from_grid(g)) for g in list_grids]
        meta = [{} for _ in range(len(clues))]
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
        meta = [{} for _ in range(len(clues_list))]
        return self._pack_batch(clues_list, grids, meta)

    def _parse_single_entry(self, entry: t.Entry) -> tuple[t.Clues, torch.Tensor, dict]:
        if isinstance(entry, dict):
            meta = dict(entry)
            rows = meta.pop("rows", meta.pop("row_clues", []))
            cols = meta.pop("cols", meta.pop("col_clues", []))
            grid = self._grid_to_tensor(meta.pop("grid", meta.pop("solution", [])))
            return [rows, cols], grid, meta

        elif isinstance(entry, (tuple, list)) and len(entry) == 2:
            clues, grid_raw = entry
            grid = self._grid_to_tensor(grid_raw)
            meta = {}
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
        if not (isinstance(first, list) and len(first) > 0):
            return False
        # first should look like list[Clues] == list[[rows_group, cols_group]]
        puzzle0 = first[0]
        is_clues_shaped = (
            isinstance(puzzle0, list)
            and len(puzzle0) == 2
            and all(isinstance(group, list) for group in puzzle0)
        )
        return is_clues_shaped and isinstance(second, (list, np.ndarray, torch.Tensor))

    def _is_grid_list(self, data: list) -> bool:
        first = data[0]
        if isinstance(first, dict):
            return False
        if isinstance(first, (np.ndarray, torch.Tensor)):
            return True
        if isinstance(first, list) and len(first) > 0:
            if len(first) == 2 and self._looks_like_clues(first[0]):
                return False
            return True
        return False

    def _looks_like_clues(self, obj) -> bool:
        return (
            isinstance(obj, list) and len(obj) > 0
            and isinstance(obj[0], list)
            and (len(obj[0]) == 0 or isinstance(obj[0][0], list))
        )

    def _normalize_grids(self, data: t.GridDataset) -> list[torch.Tensor]:
        if isinstance(data, torch.Tensor):
            if data.dim() == 2:
                return [self._grid_to_tensor(data)]
            elif data.dim() == 3:
                return [self._grid_to_tensor(g) for g in data.unbind(0)]
            else:
                raise ValueError(f"Expected 2-D or 3-D tensor, got {data.dim()}-D")
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                return [self._grid_to_tensor(data)]
            elif data.ndim == 3:
                return [self._grid_to_tensor(g) for g in data]
            else:
                raise ValueError(f"Expected 2-D or 3-D ndarray, got {data.ndim}-D")
        elif isinstance(data, list):
            return [self._grid_to_tensor(g) for g in data]
        else:
            raise ValueError(f"Unsupported grid type: {type(data)}")

    def _grid_to_tensor(self, grid: t.Grid) -> torch.Tensor:
        if isinstance(grid, torch.Tensor):
            tensor = grid if grid.dim() == 2 else grid.squeeze()
        elif isinstance(grid, np.ndarray):
            tensor = torch.from_numpy(grid).squeeze()
        elif isinstance(grid, list):
            tensor = torch.tensor(grid)
        else:
            raise ValueError(f"Cannot convert {type(grid)} to tensor")

        if tensor.dim() != 2:
            raise ValueError(f"Grid must be 2-D after squeeze, got shape {tuple(tensor.shape)}")

        return tensor.to(dtype=torch.float32)

    def _make_metadata(self, clues: t.Clues, grid: torch.Tensor) -> dict:
        grid_density = torch.sum(grid) / (grid.shape[0] * grid.shape[1])
        mean_clue_runs = np.mean([len(row) for row in clues[0]] +
                                 [len(col) for col in clues[1]])
        return {
            "shape": grid.shape,
            "density": grid_density,
            "mean_clue_runs": mean_clue_runs
        }

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict]:
        return self.X[idx], self.y[idx], self.padding_mask[idx], self.meta[idx]

    # ------------------------------------------------------------------ #
    #  Padding and reshaping utilities
    # ------------------------------------------------------------------ #

    def _pad_positional(self, clues: list[t.Clues], max_dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_runs_in_clue = (max_dimension + 1) // 2  # worst case: alternating 1s and 0s

        def pad(line: list[int]) -> list[int]:
            return line + [0] * (max_runs_in_clue - len(line))

        def mask(line: list[int]) -> list[float]:
            return [0.0] * len(line) + [-torch.inf] * (max_runs_in_clue - len(line))

        padded_clues = [[[pad(line) for line in group] for group in puzzle] for puzzle in clues]
        padding_mask = [[[mask(line) for line in group] for group in puzzle] for puzzle in clues]

        return (
            torch.tensor(padded_clues, dtype=torch.float32).flatten(start_dim=1),
            torch.tensor(padding_mask, dtype=torch.float32).flatten(start_dim=1),
        )

    def _pad_sequence(self, clues: list[t.Clues], max_dimension: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_padding = 2 * max_dimension * ((max_dimension + 1) // 2)
        padded_seqs, masks = [], []
        for puzzle in clues:
            flat_seq = [
                [val, row_or_col, row_or_col_idx, run_idx]
                for row_or_col, groups in enumerate(puzzle)
                for row_or_col_idx, line in enumerate(groups)
                for run_idx, val in enumerate(line)
            ]
            pad_len = max_padding - len(flat_seq)

            padded_seqs.append(flat_seq + [[0, 0, 0, 0]] * pad_len)
            masks.append([0.0] * len(flat_seq) + [-torch.inf] * pad_len)
        return (
            torch.tensor(padded_seqs, dtype=torch.float32).flatten(start_dim=1),
            torch.tensor(masks, dtype=torch.float32).flatten(start_dim=1),
        )