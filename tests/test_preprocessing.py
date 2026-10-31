import numpy as np
import pandas as pd

from diabetes_classifier.preprocessing import Preprocessor


def make_features() -> pd.DataFrame:
    return pd.DataFrame({"Insulin": [0.0, 10.0, 20.0, 30.0], "Age": [21.0, 30.0, 40.0, 50.0]})


def test_zeros_are_replaced_with_training_median():
    pre = Preprocessor(zero_as_missing=["Insulin"]).fit(make_features())
    assert pre.medians_["Insulin"] == 20.0  # median of the non-zero values only
    frame = pre._transform_frame(make_features())
    assert frame.loc[0, "Insulin"] == 20.0


def test_medians_come_from_training_data_only():
    pre = Preprocessor(zero_as_missing=["Insulin"]).fit(make_features())
    other = pd.DataFrame({"Insulin": [0.0, 1000.0], "Age": [25.0, 35.0]})
    assert pre._transform_frame(other).loc[0, "Insulin"] == 20.0


def test_missing_indicator_and_log_transform():
    pre = Preprocessor(["Insulin"], missing_indicators=True, log_transform=["Age"])
    pre.fit(make_features())
    frame = pre._transform_frame(make_features())

    assert pre.feature_names_ == ["Insulin", "Age", "Insulin_missing"]
    assert frame["Insulin_missing"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert np.isclose(frame.loc[0, "Age"], np.log1p(21.0))


def test_transform_returns_scaled_float32():
    X = Preprocessor().fit(make_features()).transform(make_features())
    assert X.dtype == np.float32
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-6)
