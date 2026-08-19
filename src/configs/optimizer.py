from dataclasses import dataclass


@dataclass
class AdamConfig:
    _target_: str = "torch.optim.Adam"
    lr: float = 1e-6


@dataclass
class AdamWConfig:
    _target_: str = "torch.optim.AdamW"
    lr: float = 1e-6
    weight_decay: float = 0.01
