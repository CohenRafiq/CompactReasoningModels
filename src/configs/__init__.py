from hydra.core.config_store import ConfigStore

from src.configs.models import (
    MLPConfig,
    TransformerConfig,
    CNNConfig,
    GridMLPConfig,
    RecursiveMLPConfig,
    RecursiveGridMLPConfig,
)
from src.configs.data import PuzzleDatasetConfig, ParquetPuzzleDatasetConfig
from src.configs.criterion import BCELossConfig, NonogramLossConfig, AbstainLossConfig
from src.configs.trainer import SupervisedTrainerConfig, RewardTrainerConfig
from src.configs.optimizer import AdamConfig, AdamWConfig
from src.configs.scheduler import CosineAnnealingConfig, NullSchedulerConfig, WarmupCosineConfig
from src.configs.dataloader import DataLoaderConfig, SmallBatchConfig
from src.configs.logger import WandbLoggerConfig, NullLoggerConfig
from src.configs.split import RandomSplitConfig


def register_configs():
    cs = ConfigStore.instance()

    # Models
    cs.store(name="mlp", node=MLPConfig, group="model")
    cs.store(name="transformer", node=TransformerConfig, group="model")
    cs.store(name="cnn", node=CNNConfig, group="model")
    cs.store(name="gridmlp", node=GridMLPConfig, group="model")
    cs.store(name="recursive_mlp", node=RecursiveMLPConfig, group="model")
    cs.store(name="recursive_gridmlp", node=RecursiveGridMLPConfig, group="model")

    # Data
    cs.store(name="puzzle_dataset", node=PuzzleDatasetConfig, group="data")
    cs.store(name="parquet_dataset", node=ParquetPuzzleDatasetConfig, group="data")

    # Criterion
    cs.store(name="bce", node=BCELossConfig, group="criterion")
    cs.store(name="nonogram", node=NonogramLossConfig, group="criterion")
    cs.store(name="abstain", node=AbstainLossConfig, group="criterion")

    # Trainer
    cs.store(name="supervised", node=SupervisedTrainerConfig, group="trainer")
    cs.store(name="reward", node=RewardTrainerConfig, group="trainer")

    # Optimizer
    cs.store(name="adam", node=AdamConfig, group="optimizer")
    cs.store(name="adamw", node=AdamWConfig, group="optimizer")

    # Scheduler
    cs.store(name="cosine_annealing", node=CosineAnnealingConfig, group="scheduler")
    cs.store(name="null_scheduler", node=NullSchedulerConfig, group="scheduler")
    cs.store(name="warmup_cosine", node=WarmupCosineConfig, group="scheduler")

    # Dataloader
    cs.store(name="default", node=DataLoaderConfig, group="dataloader")
    cs.store(name="small_batch", node=SmallBatchConfig, group="dataloader")

    # Logger
    cs.store(name="wandb", node=WandbLoggerConfig, group="logger")
    cs.store(name="null_logger", node=NullLoggerConfig, group="logger")

    # Split
    cs.store(name="random_split", node=RandomSplitConfig, group="split")


register_configs()
