from dataclasses import dataclass
from typing import Optional


@dataclass
class SupervisedTrainerConfig:
    _target_: str = "src.training.nng_supervised.NNGSupervisedTrainer"
    epochs: int = 100
    early_stopping_patience: Optional[int] = 10
    early_stopping_min_delta: float = 1e-6
    print_every: int = 10


@dataclass
class RewardTrainerConfig:
    _target_: str = "src.training.nng_reward.NNGRewardTrainer"
    epochs: int = 100
    early_stopping_patience: Optional[int] = 10
    early_stopping_min_delta: float = 1e-6
    print_every: int = 10
