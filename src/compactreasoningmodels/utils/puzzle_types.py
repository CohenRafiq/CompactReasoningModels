from typing import TypedDict, TypeAlias

import torch
import numpy as np



Clues: TypeAlias = list[list[list[int]]]
Grid: TypeAlias = list[list[int]] | np.ndarray | torch.Tensor
class EntryDict(TypedDict, total=False):
    rows: list[list[int]]
    cols: list[list[int]]
    grid: Grid
    # + metadata fields
Entry: TypeAlias = tuple[Clues, Grid] | list[Clues, Grid] | EntryDict
GridDataset: TypeAlias = list[Grid] | np.ndarray | torch.Tensor
SplitData: TypeAlias = tuple[list[Clues], GridDataset] | list[list[Clues], GridDataset]
Dataset: TypeAlias = GridDataset | list[Entry] | SplitData
ReformattedData: TypeAlias = tuple[list[Clues], torch.Tensor, list[dict]]