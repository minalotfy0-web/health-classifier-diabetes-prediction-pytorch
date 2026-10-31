from pathlib import Path

import pytest

from diabetes_classifier.data import DataSplits, load_dataset, split_and_scale


@pytest.fixture(scope="session")
def data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"


@pytest.fixture(scope="session")
def splits(data_path: Path) -> DataSplits:
    return split_and_scale(load_dataset(data_path), test_size=0.2, val_size=0.1, seed=0)
