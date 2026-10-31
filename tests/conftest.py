from pathlib import Path

import pandas as pd
import pytest

from diabetes_classifier.config import Config
from diabetes_classifier.data import (
    FEATURE_COLUMNS,
    FoldData,
    load_dataset,
    prepare_fold,
    split_development_test,
    stratified_split,
)
from diabetes_classifier.predict import Predictor

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "diabetes.csv"


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


@pytest.fixture(scope="session")
def tuned_model_path() -> Path:
    """The committed final model."""
    return REPO_ROOT / "results" / "tuned" / "model.joblib"


@pytest.fixture(scope="session")
def tuned_predictor(tuned_model_path: Path) -> Predictor:
    return Predictor.load(tuned_model_path)


@pytest.fixture
def patients(df: pd.DataFrame) -> pd.DataFrame:
    """A few real patients with only the eight feature columns."""
    return df.head(20)[list(FEATURE_COLUMNS)].reset_index(drop=True)
