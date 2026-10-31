"""Experiment configuration loaded from YAML."""

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    """All settings for one experiment run."""

    seed: int = 974
    data_path: str = "data/diabetes.csv"
    test_size: float = 0.2
    val_size: float = 0.1
    hidden_sizes: list[int] = field(default_factory=lambda: [8, 16, 32, 16, 8])
    dropout: float = 0.5
    dropout_layers: list[int] = field(default_factory=lambda: [3])
    epochs: int = 7000
    learning_rate: float = 0.01
    log_every: int = 500
    output_dir: str = "results"

    def __post_init__(self) -> None:
        if not 0 < self.test_size + self.val_size < 1:
            raise ValueError("test_size + val_size must be between 0 and 1")
        if any(i >= len(self.hidden_sizes) for i in self.dropout_layers):
            raise ValueError("dropout_layers refers to a layer that does not exist")


def load_config(path: str | Path) -> Config:
    """Read a YAML config file; relative paths in it are resolved against its folder."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")

    for key in ("data_path", "output_dir"):
        if key in raw and not Path(raw[key]).is_absolute():
            raw[key] = str(path.parent / raw[key])
    return Config(**raw)
