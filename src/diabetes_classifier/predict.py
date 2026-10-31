"""A saved, ready-to-use model: preprocessing, fitted models and decision threshold."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from diabetes_classifier.data import FEATURE_COLUMNS
from diabetes_classifier.models import FittedModels
from diabetes_classifier.preprocessing import Preprocessor


@dataclass
class Predictor:
    """Scores new patients exactly as the model was evaluated.

    Input columns are the eight dataset features; 0 in Glucose, BloodPressure,
    SkinThickness, Insulin or BMI means "not measured" and is imputed.
    """

    model_name: str
    threshold: float
    preprocessor: Preprocessor
    models: FittedModels

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Probability of diabetes for each row; raises ValueError on invalid input."""
        X = self.preprocessor.transform(_validated(features))
        return self.models.predict_proba(self.model_name, X)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """1 = diabetes, 0 = no diabetes, using the threshold tuned on validation data."""
        return (self.predict_proba(features) >= self.threshold).astype(int)

    def save(self, path: str | Path) -> None:
        """Write the model to `path` (compressed joblib)."""
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> "Predictor":
        """Load a saved model. Only load files you trust: this unpickles Python objects."""
        if not Path(path).is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        try:
            predictor = joblib.load(path)
        except Exception as error:  # unpickling can fail in many ways
            raise ValueError(f"Could not read a model from {path}: {error}") from error
        if not isinstance(predictor, Predictor):
            raise TypeError(f"{path} does not contain a Predictor")
        return predictor


def _validated(features: pd.DataFrame) -> pd.DataFrame:
    """The feature columns as floats, after checking they are present, numeric and sane."""
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if features.empty:
        raise ValueError("No patients to score: the input has no rows")
    numbers = features[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    invalid = [c for c in FEATURE_COLUMNS if not np.isfinite(numbers[c]).all()]
    if invalid:
        raise ValueError(f"Columns with non-numeric, blank or infinite values: {invalid}")
    negative = [c for c in FEATURE_COLUMNS if (numbers[c] < 0).any()]
    if negative:
        raise ValueError(f"Columns with negative values: {negative}")
    return numbers
