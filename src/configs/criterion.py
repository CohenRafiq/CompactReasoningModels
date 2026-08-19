from dataclasses import dataclass


@dataclass
class BCELossConfig:
    _target_: str = "torch.nn.BCEWithLogitsLoss"


@dataclass
class NonogramLossConfig:
    _target_: str = "src.data.criterion.nonogram.NonogramLoss"
    reduction: str = "mean"


@dataclass
class AbstainLossConfig:
    _target_: str = "src.data.criterion.categorical_abstain.AbstainLoss"
    abstain_penalty: float = 0.5
    entropy_weight: float = 0.1
