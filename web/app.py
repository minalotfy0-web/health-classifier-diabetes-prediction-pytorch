"""FastAPI app: `POST /predict`, `GET /health` and a single-page form at `/`.

Run from the repository root: `uvicorn web.app:app`. Set `MODEL_PATH` to use another model file.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from diabetes_classifier.predict import Predictor

logger = logging.getLogger("web")

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "results/tuned/model.joblib"


class PatientFeatures(BaseModel):
    """The eight Pima features. Omit or send null for lab values that were not measured."""

    model_config = ConfigDict(extra="forbid")

    pregnancies: int = Field(0, ge=0, le=20, description="Number of pregnancies")
    glucose: float | None = Field(None, ge=20, le=400, description="Plasma glucose (mg/dL)")
    blood_pressure: float | None = Field(None, ge=20, le=250, description="Diastolic BP (mm Hg)")
    skin_thickness: float | None = Field(None, ge=1, le=100, description="Triceps skin fold (mm)")
    insulin: float | None = Field(None, ge=1, le=1000, description="2-hour insulin (mu U/ml)")
    bmi: float | None = Field(None, ge=10, le=100, description="Body mass index (kg/m^2)")
    diabetes_pedigree_function: float = Field(..., ge=0, le=5, description="Pedigree function")
    age: int = Field(..., ge=18, le=120, description="Age in years")

    def to_frame(self) -> pd.DataFrame:
        """One-row frame with the dataset's column names; missing labs become 0."""
        return pd.DataFrame(
            [
                {
                    "Pregnancies": self.pregnancies,
                    "Glucose": self.glucose or 0.0,
                    "BloodPressure": self.blood_pressure or 0.0,
                    "SkinThickness": self.skin_thickness or 0.0,
                    "Insulin": self.insulin or 0.0,
                    "BMI": self.bmi or 0.0,
                    "DiabetesPedigreeFunction": self.diabetes_pedigree_function,
                    "Age": self.age,
                }
            ]
        )


class PredictionResponse(BaseModel):
    probability: float = Field(description="Estimated probability of diabetes, 0 to 1")
    threshold: float = Field(description="Decision threshold of the model")
    prediction: int = Field(description="1 = diabetes, 0 = no diabetes")
    label: str = Field(description="Human-readable prediction")
    model: str = Field(description="Name of the model that scored the patient")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model: str | None
    model_loaded: bool


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Build the app, loading the model once from `model_path`, `$MODEL_PATH` or the default."""
    path = Path(model_path or os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH)
    predictor: Predictor | None = None
    try:
        predictor = Predictor.load(path)
    except Exception:
        logger.exception("Could not load model from %s", path)

    app = FastAPI(title="Diabetes risk demo", version="1.0.0")
    app.state.predictor = predictor

    @app.get("/health", response_model=HealthResponse)
    def health() -> JSONResponse:
        body = HealthResponse(
            status="ok" if predictor else "model not loaded",
            model=predictor.model_name if predictor else None,
            model_loaded=predictor is not None,
        )
        return JSONResponse(body.model_dump(), status_code=200 if predictor else 503)

    @app.post("/predict", response_model=PredictionResponse)
    def predict(patient: PatientFeatures) -> PredictionResponse:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        probability = float(predictor.predict_proba(patient.to_frame())[0])
        positive = int(probability >= predictor.threshold)
        return PredictionResponse(
            probability=round(probability, 3),
            threshold=round(predictor.threshold, 3),
            prediction=positive,
            label="Diabetes predicted" if positive else "No diabetes predicted",
            model=predictor.model_name,
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
