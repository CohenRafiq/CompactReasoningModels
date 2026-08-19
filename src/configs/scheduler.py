from dataclasses import dataclass


@dataclass
class CosineAnnealingConfig:
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    T_max: int = 100
    eta_min: float = 1e-6


@dataclass
class NullSchedulerConfig:
    _target_: str = "src.configs.null_target.NullTarget"


@dataclass
class WarmupCosineConfig:
    _target_: str = "src.configs.schedulers.WarmupCosineScheduler"
    warmup_epochs: int = 10
