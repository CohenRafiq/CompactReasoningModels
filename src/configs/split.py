from dataclasses import dataclass
from typing import Optional


@dataclass
class RandomSplitConfig:
    train_ratio: float = 0.8
    seed: int = 42
    max_samples: Optional[int] = None
