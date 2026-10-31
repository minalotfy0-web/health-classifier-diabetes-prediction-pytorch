"""Experiment configuration loaded from YAML."""

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

MODEL_NAMES = (
    "nn",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "nn+gradient_boosting",
)
PATH_KEYS = ("data_path", "output_dir")


@dataclass
class Config:
    """All settings for one experiment. See configs/default.yaml for what each one does."""

    name: str = "default"
    seed: int = 974

    data_path: str = "data/diabetes.csv"
    test_size: float = 0.2
    val_size: float = 0.1

    zero_as_missing: list[str] = field(
        default_factory=lambda: ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    )
    missing_indicators: bool = True
    log_transform: list[str] = field(
        default_factory=lambda: ["Insulin", "DiabetesPedigreeFunction", "Age"]
    )

    hidden_sizes: list[int] = field(default_factory=lambda: [32, 16])
    dropout: float = 0.2
    dropout_layers: list[int] | None = None
    batch_norm: bool = False

    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int | None = 32
    max_epochs: int = 1000
    patience: int | None = 100
    lr_scheduler: bool = True
    pos_weight: float | str | None = "auto"
    log_every: int = 100

    threshold: float | str = "f1"
    ensemble_size: int = 5

    models: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    cv_folds: int = 5
    cv_repeats: int = 5

    output_dir: str = "results"

    def __post_init__(self) -> None:
        if not 0 < self.test_size + self.val_size < 1:
            raise ValueError("test_size + val_size must be between 0 and 1")
        if self.dropout_layers is not None and any(
            i >= len(self.hidden_sizes) for i in self.dropout_layers
        ):
            raise ValueError("dropout_layers refers to a layer that does not exist")
        if self.optimizer not in ("adamw", "sgd"):
            raise ValueError(f"optimizer must be 'adamw' or 'sgd', got {self.optimizer!r}")
        if isinstance(self.pos_weight, str) and self.pos_weight != "auto":
            raise ValueError("pos_weight must be a number, 'auto' or null")
        if isinstance(self.threshold, str) and self.threshold not in ("f1", "youden"):
            raise ValueError("threshold must be a probability, 'f1' or 'youden'")
        unknown = set(self.models) - set(MODEL_NAMES)
        if unknown:
            raise ValueError(f"Unknown models {sorted(unknown)}; choose from {MODEL_NAMES}")
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be at least 1")

    @property
    def val_fraction_of_development(self) -> float:
        """Validation share of the development (non-test) set."""
        return self.val_size / (1 - self.test_size)


def load_config(path: str | Path) -> Config:
    """Read a YAML config; relative paths in it are resolved against its folder."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")

    raw.setdefault("name", path.stem)
    for key in PATH_KEYS:
        if key in raw and not Path(raw[key]).is_absolute():
            raw[key] = str(path.parent / raw[key])
    return Config(**raw)


def save_config(config: Config, path: str | Path) -> None:
    """Write a config as YAML, with its paths relative to the file's folder."""
    path = Path(path)
    raw = asdict(config)
    for key in PATH_KEYS:
        raw[key] = Path(os.path.relpath(raw[key], path.parent)).as_posix()
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
