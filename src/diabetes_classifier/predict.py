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
        missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        X = self.preprocessor.transform(features[list(FEATURE_COLUMNS)])
        return self.models.predict_proba(self.model_name, X)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """1 = diabetes, 0 = no diabetes, using the threshold tuned on validation data."""
        return (self.predict_proba(features) >= self.threshold).astype(int)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> "Predictor":
        """Load a saved model. Only load files you trust: this unpickles Python objects."""
        predictor = joblib.load(path)
        if not isinstance(predictor, Predictor):
            raise TypeError(f"{path} does not contain a Predictor")
        return predictor
