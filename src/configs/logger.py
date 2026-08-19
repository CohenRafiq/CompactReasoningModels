from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WandbLoggerConfig:
    _target_: str = "src.utils.logging.WandbLogger"
    project: str = "nonograms"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    group: Optional[str] = None
    save_code: bool = True
    should_log_model: bool = True


@dataclass
class NullLoggerConfig:
    _target_: str = "src.utils.logging.NullLogger"
