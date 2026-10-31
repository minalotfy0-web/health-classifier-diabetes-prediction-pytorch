from pathlib import Path

import pandas as pd
import pytest

from diabetes_classifier.config import Config
from diabetes_classifier.data import (
    FoldData,
    load_dataset,
    prepare_fold,
    split_development_test,
    stratified_split,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


@pytest.fixture
def small_config(tmp_path: Path) -> Config:
    """A fast configuration for tests: few epochs, small ensemble, 2-fold CV."""
    return Config(
        data_path=str(DATA_PATH),
        output_dir=str(tmp_path),
        hidden_sizes=[16, 8],
        max_epochs=30,
        patience=10,
        ensemble_size=2,
        cv_folds=2,
        cv_repeats=1,
        log_every=10,
    )


@pytest.fixture
def fold(df: pd.DataFrame, small_config: Config) -> FoldData:
    development, test = split_development_test(df, small_config)
    train_df, val_df = stratified_split(
        development, small_config.val_fraction_of_development, small_config.seed
    )
    return prepare_fold(train_df, val_df, test, small_config)
