"""The FastAPI demo in web/: /predict, /health and the static frontend."""

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from web.app import create_app  # noqa: E402

PATIENT = {
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 120,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50,
}
REQUIRED = {"diabetes_pedigree_function": 0.627, "age": 50}
LABS = ("glucose", "blood_pressure", "skin_thickness", "insulin", "bmi")
COLUMNS = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "blood_pressure": "BloodPressure",
    "skin_thickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "diabetes_pedigree_function": "DiabetesPedigreeFunction",
    "age": "Age",
}


def to_frame(payload: dict) -> pd.DataFrame:
    """The frame the API should score: omitted pregnancies and missing labs become 0."""
    row = {column: payload.get(key) or 0.0 for key, column in COLUMNS.items()}
    return pd.DataFrame([row])


@pytest.fixture(autouse=True)
def no_model_path_env(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)


@pytest.fixture
def client(tuned_model_path):
    return TestClient(create_app(tuned_model_path))


@pytest.fixture
def broken_client(tmp_path):
    return TestClient(create_app(tmp_path / "nope.joblib"))


# Health


def test_health_reports_the_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "nn+gradient_boosting",
        "model_loaded": True,
    }


def test_health_is_503_when_the_model_cannot_load(broken_client):
    response = broken_client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["model_loaded"] is False
    assert body["model"] is None


def test_predict_is_503_when_the_model_cannot_load(broken_client):
    response = broken_client.post("/predict", json=PATIENT)
    assert response.status_code == 503
    assert response.json() == {"detail": "Model not loaded"}


def test_a_corrupt_model_file_is_a_failed_load(tmp_path):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(b"not a model")
    response = TestClient(create_app(path)).get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_model_path_comes_from_the_environment(monkeypatch, tuned_model_path, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tuned_model_path))
    assert TestClient(create_app()).get("/health").status_code == 200

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    assert TestClient(create_app()).get("/health").status_code == 503


def test_the_argument_wins_over_the_environment(monkeypatch, tuned_model_path, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    assert TestClient(create_app(tuned_model_path)).get("/health").status_code == 200


def test_default_model_is_the_committed_one():
    response = TestClient(create_app()).get("/health")
    assert response.status_code == 200
    assert response.json()["model"] == "nn+gradient_boosting"


# Predict


def test_predict_returns_a_complete_response(client):
    response = client.post("/predict", json=PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"probability", "threshold", "prediction", "label", "model"}
    assert 0 <= body["probability"] <= 1
    assert 0 < body["threshold"] < 1
    assert body["model"] == "nn+gradient_boosting"
    assert body["prediction"] in (0, 1)
    expected = "Diabetes predicted" if body["prediction"] else "No diabetes predicted"
    assert body["label"] == expected


def test_prediction_matches_the_predictor(client, tuned_predictor):
    body = client.post("/predict", json=PATIENT).json()
    probability = float(tuned_predictor.predict_proba(to_frame(PATIENT))[0])

    assert body["probability"] == pytest.approx(probability, abs=1e-3)  # rounded to 3 places
    assert body["threshold"] == pytest.approx(tuned_predictor.threshold, abs=1e-3)
    assert body["prediction"] == int(probability >= tuned_predictor.threshold)


@pytest.mark.parametrize(
    "payload",
    [
        REQUIRED,
        {**REQUIRED, **dict.fromkeys(LABS)},
        {**REQUIRED, "glucose": 148, "bmi": None},
        {**REQUIRED, "pregnancies": 0},
    ],
    ids=["required-only", "labs-null", "some-labs", "zero-pregnancies"],
)
def test_missing_lab_values_are_scored_as_not_measured(client, tuned_predictor, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    probability = float(tuned_predictor.predict_proba(to_frame(payload))[0])
    assert response.json()["probability"] == pytest.approx(probability, abs=1e-3)


def test_omitting_a_lab_equals_sending_null(client):
    omitted = client.post("/predict", json=REQUIRED).json()
    null = client.post("/predict", json={**REQUIRED, "insulin": None}).json()
    assert omitted == null


def test_the_api_agrees_with_the_predictor_on_real_patients(client, tuned_predictor, df):
    for _, row in df.head(30).iterrows():
        payload = {
            key: (float(row[column]) or None) if key in LABS else float(row[column])
            for key, column in COLUMNS.items()
        }
        payload["pregnancies"] = int(row["Pregnancies"])
        payload["age"] = int(row["Age"])
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, payload

        probability = float(tuned_predictor.predict_proba(to_frame(payload))[0])
        assert response.json()["probability"] == pytest.approx(probability, abs=1e-3)
        assert response.json()["prediction"] == int(probability >= tuned_predictor.threshold)


@pytest.mark.parametrize(
    "overrides",
    [
        {"pregnancies": 0, "glucose": 20, "blood_pressure": 20, "skin_thickness": 1},
        {"insulin": 1, "bmi": 10, "diabetes_pedigree_function": 0, "age": 18},
        {"pregnancies": 20, "glucose": 400, "blood_pressure": 250, "skin_thickness": 100},
        {"insulin": 1000, "bmi": 100, "diabetes_pedigree_function": 5, "age": 120},
    ],
    ids=["lower-1", "lower-2", "upper-1", "upper-2"],
)
def test_values_on_the_limits_are_accepted(client, overrides):
    response = client.post("/predict", json={**PATIENT, **overrides})
    assert response.status_code == 200
    assert 0 <= response.json()["probability"] <= 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"pregnancies": -1},
        {"pregnancies": 21},
        {"glucose": 19},
        {"glucose": 0},
        {"glucose": 401},
        {"blood_pressure": 19},
        {"blood_pressure": 251},
        {"skin_thickness": 0},
        {"skin_thickness": 101},
        {"insulin": 0},
        {"insulin": 1001},
        {"bmi": 9.9},
        {"bmi": 100.1},
        {"diabetes_pedigree_function": -0.1},
        {"diabetes_pedigree_function": 5.1},
        {"age": 17},
        {"age": 121},
        {"age": 30.5},
    ],
)
def test_out_of_range_values_are_rejected(client, overrides):
    assert client.post("/predict", json={**PATIENT, **overrides}).status_code == 422


@pytest.mark.parametrize("missing", ["age", "diabetes_pedigree_function"])
def test_required_fields_are_enforced(client, missing):
    payload = {key: value for key, value in PATIENT.items() if key != missing}
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("required", ["age", "diabetes_pedigree_function"])
def test_required_fields_cannot_be_null(client, required):
    assert client.post("/predict", json={**PATIENT, required: None}).status_code == 422


@pytest.mark.parametrize(
    "overrides",
    [
        {"glucose": "high"},
        {"age": "old"},
        {"bmi": [33.6]},
        {"pregnancies": "many"},
        {"diabetes_pedigree_function": {}},
    ],
)
def test_wrong_types_are_rejected(client, overrides):
    assert client.post("/predict", json={**PATIENT, **overrides}).status_code == 422


def test_unknown_keys_are_rejected(client):
    assert client.post("/predict", json={**PATIENT, "outcome": 1}).status_code == 422
    assert client.post("/predict", json={**PATIENT, "Glucose": 150}).status_code == 422


def test_empty_and_malformed_bodies_are_rejected(client):
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", content=b"not json").status_code == 422
    assert client.post("/predict", json=[PATIENT]).status_code == 422
    assert client.post("/predict").status_code == 422


def test_validation_errors_do_not_depend_on_the_model(broken_client):
    # Input is validated before the model is needed
    assert broken_client.post("/predict", json={"age": 5}).status_code == 422


def test_wrong_methods_and_unknown_paths(client):
    assert client.get("/predict").status_code == 405
    assert client.post("/health").status_code == 405
    assert client.get("/nope").status_code == 404


# Frontend


def test_index_serves_html_with_the_disclaimer(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "clinical" in response.text.lower()


def test_static_files_are_served(client):
    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert len(script.text) > 0
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/missing.js").status_code == 404


def test_frontend_is_served_even_without_a_model(broken_client):
    assert broken_client.get("/").status_code == 200
    assert broken_client.get("/static/app.js").status_code == 200
