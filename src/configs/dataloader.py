from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    _target_: str = "torch.utils.data.DataLoader"
    batch_size: int = 2048
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class SmallBatchConfig:
    _target_: str = "torch.utils.data.DataLoader"
    batch_size: int = 8
    num_workers: int = 1
    pin_memory: bool = True
    persistent_workers: bool = True
