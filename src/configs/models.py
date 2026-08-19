from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MLPConfig:
    _target_: str = "src.models.mlp.MultiLayerPerceptron"
    hidden_size: int = 256
    num_layers: int = 9
    dropout: float = 0.3


@dataclass
class TransformerConfig:
    _target_: str = "src.models.tfm.Transformer"
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    dropout: float = 0.1
    num_patches: Optional[int] = None
    ff_dim: Optional[int] = None


@dataclass
class CNNConfig:
    _target_: str = "src.models.cnn.ConvNeuralNetwork"
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_size: int = 3
    dims: int = 2
    fc_hidden_size: int = 256
    fc_num_layers: int = 2
    pool_every: int = 2
    dropout: float = 0.1


@dataclass
class GridMLPConfig:
    _target_: str = "src.models.gridmlp.GridMLP"
    hidden_size: int = 256
    num_layers: int = 9
    dropout: float = 0.3


@dataclass
class RecursiveMLPConfig:
    _target_: str = "src.models.recursive_mlp.RecursiveMLP"
    hidden_size: int = 256
    num_layers: int = 9
    dropout: float = 0.3


@dataclass
class RecursiveGridMLPConfig:
    _target_: str = "src.models.recursive_gridmlp.RecursiveGridMLP"
    hidden_size: int = 256
    num_layers: int = 9
    dropout: float = 0.3
