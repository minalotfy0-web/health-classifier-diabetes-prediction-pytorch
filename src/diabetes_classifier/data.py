"""Loading, splitting and scaling the Pima Indians Diabetes dataset."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Outcome"


@dataclass
class DataSplits:
    """Standardised train/validation/test tensors."""

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the dataset CSV and check that it has the target column."""
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected a '{TARGET_COLUMN}' column in {path}")
    return df


def split_and_scale(
    df: pd.DataFrame, test_size: float, val_size: float, seed: int
) -> DataSplits:
    """Split into stratified train/validation/test sets and standardise the features.

    `test_size` and `val_size` are fractions of the whole dataset. The scaler is fitted
    on the training set only, so no information leaks from the held-out sets.
    """
    features = df.drop(columns=TARGET_COLUMN)
    X = features.to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest,
        y_rest,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=y_rest,
    )

    scaler = StandardScaler().fit(X_train)

    def to_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array, dtype=np.float32))

    return DataSplits(
        X_train=to_tensor(scaler.transform(X_train)),
        y_train=to_tensor(y_train),
        X_val=to_tensor(scaler.transform(X_val)),
        y_val=to_tensor(y_val),
        X_test=to_tensor(scaler.transform(X_test)),
        y_test=to_tensor(y_test),
        feature_names=list(features.columns),
    )
