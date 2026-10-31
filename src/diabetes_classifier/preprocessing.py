"""Feature preprocessing, fitted on the training set only to avoid leakage."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """Impute zeros that mean "not measured", flag them, log-transform and standardise.

    Args:
        zero_as_missing: Columns where 0 is a missing value; replaced with the median of
            the non-zero training values.
        missing_indicators: Add a `<column>_missing` 0/1 feature for each of those columns.
        log_transform: Skewed columns to transform with log(1 + x) before scaling.
    """

    def __init__(
        self,
        zero_as_missing: Sequence[str] = (),
        missing_indicators: bool = False,
        log_transform: Sequence[str] = (),
    ) -> None:
        self.zero_as_missing = list(zero_as_missing)
        self.missing_indicators = missing_indicators
        self.log_transform = list(log_transform)

    def fit(self, features: pd.DataFrame) -> "Preprocessor":
        """Learn the imputation medians and the scaling from the training features."""
        wanted = [*self.zero_as_missing, *self.log_transform]
        absent = [column for column in wanted if column not in features.columns]
        if absent:
            raise ValueError(f"Missing feature columns: {absent}")
        self.medians_ = {
            column: features.loc[features[column] != 0, column].median()
            for column in self.zero_as_missing
        }
        all_zero = [column for column, median in self.medians_.items() if np.isnan(median)]
        if all_zero:
            raise ValueError(f"Columns with only zeros cannot be imputed: {all_zero}")
        transformed = self._transform_frame(features)
        self.feature_names_ = list(transformed.columns)
        self.scaler_ = StandardScaler().fit(transformed)
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Impute, flag, log-transform and standardise `features` as float32."""
        return self.scaler_.transform(self._transform_frame(features)).astype(np.float32)

    def _transform_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.astype(float)
        for column in self.zero_as_missing:
            missing = out[column] == 0
            if self.missing_indicators:
                out[f"{column}_missing"] = missing.astype(float)
            out.loc[missing, column] = self.medians_[column]
        for column in self.log_transform:
            out[column] = np.log1p(out[column])
        return out
