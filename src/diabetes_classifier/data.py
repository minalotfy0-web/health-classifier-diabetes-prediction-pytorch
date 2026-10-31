"""Loading and splitting the Pima Indians Diabetes dataset."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from diabetes_classifier.config import Config
from diabetes_classifier.preprocessing import Preprocessor

TARGET_COLUMN = "Outcome"


@dataclass
class FoldData:
    """Preprocessed arrays for one train / validation / evaluation split.

    The validation set is used for early stopping and threshold tuning; the evaluation
    set (a CV fold or the test set) only for scoring.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_eval: np.ndarray
    y_eval: np.ndarray
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


def stratified_split(
    df: pd.DataFrame, held_out_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (rest, held_out), keeping the outcome ratio in both parts."""
    rest, held_out = train_test_split(
        df, test_size=held_out_fraction, random_state=seed, stratify=df[TARGET_COLUMN]
    )
    return rest, held_out


def split_development_test(df: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The test set is split off first and used only by the final evaluation."""
    return stratified_split(df, config.test_size, config.seed)


def prepare_fold(
    train_df: pd.DataFrame, val_df: pd.DataFrame, eval_df: pd.DataFrame, config: Config
) -> FoldData:
    """Fit the preprocessor on the training rows and transform all three parts."""
    preprocessor = Preprocessor(
        config.zero_as_missing, config.missing_indicators, config.log_transform
    )
    preprocessor.fit(train_df.drop(columns=TARGET_COLUMN))

    def arrays(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = preprocessor.transform(part.drop(columns=TARGET_COLUMN))
        return X, part[TARGET_COLUMN].to_numpy(dtype=np.float32)

    X_train, y_train = arrays(train_df)
    X_val, y_val = arrays(val_df)
    X_eval, y_eval = arrays(eval_df)
    return FoldData(X_train, y_train, X_val, y_val, X_eval, y_eval, preprocessor.feature_names_)
