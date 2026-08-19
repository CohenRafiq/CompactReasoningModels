from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseLogger(ABC):

    @abstractmethod
    def setup(self, cfg: Any = None):
        ...

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        ...

    @abstractmethod
    def log_model(self, model_path: str, name: str = "model"):
        ...

    @abstractmethod
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        ...

    @abstractmethod
    def finish(self):
        ...


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: list = None,
        group: Optional[str] = None,
        save_code: bool = True,
        should_log_model: bool = True,
        config: Dict[str, Any] = None
    ):
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.group = group
        self.save_code = save_code
        self.should_log_model = should_log_model
        self.config = config or {}
        self.run = None
    
    def setup(self, cfg=None):
        import wandb
        run_config = {**self.config}
        if cfg is not None:
            from omegaconf import OmegaConf
            run_config.update(OmegaConf.to_container(cfg, resolve=True))

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            tags=self.tags,
            group=self.group,
            save_code=self.save_code,
            config=run_config
        )
        return self.run
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.run:
            import wandb
            wandb.log(metrics, step=step)
    
    def log_model(self, model_path: str, name: str = "model"):
        if self.run and self.should_log_model:
            import wandb
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
    
    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        if self.run:
            import wandb
            wandb.watch(model, log="all", log_freq=log_freq)
    
    def finish(self):
        if self.run:
            import wandb
            wandb.finish()


class NullLogger(BaseLogger):
    def setup(self, cfg=None): pass
    def log_metrics(self, *args, **kwargs): pass
    def log_model(self, *args, **kwargs): pass
    def watch_model(self, *args, **kwargs): pass
    def finish(self): pass
