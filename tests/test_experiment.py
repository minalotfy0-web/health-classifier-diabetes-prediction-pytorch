import dataclasses
import json
import shutil

import numpy as np
import pandas as pd
import yaml

from diabetes_classifier.cli import main
from diabetes_classifier.config import MODEL_NAMES, load_config, save_config
from diabetes_classifier.data import FEATURE_COLUMNS
from diabetes_classifier.experiment import (
    load_cv_folds,
    plot_cv_results,
    run_cv,
    run_final,
    write_cv_summary,
)
from diabetes_classifier.models import fit_predict
from diabetes_classifier.predict import Predictor
from diabetes_classifier.tune import tune


def test_every_model_predicts_probabilities(small_config, fold):
    outputs, fitted = fit_predict(small_config, fold, seed=0)
    assert list(outputs) == list(MODEL_NAMES)
    assert len(fitted.networks) == len(fitted.histories) == small_config.ensemble_size
    for output in outputs.values():
        assert output.eval_probs.shape == fold.y_eval.shape
        assert ((output.eval_probs >= 0) & (output.eval_probs <= 1)).all()


def test_run_cv_writes_summary(small_config, tmp_path):
    summary = run_cv(small_config)
    assert set(summary["model"]) == set(MODEL_NAMES)
    folds = pd.read_csv(tmp_path / "cv" / "default_folds.csv")
    assert len(folds) == len(MODEL_NAMES) * 2
    assert write_cv_summary(tmp_path).exists()
    assert plot_cv_results(tmp_path).exists()


def test_identical_baseline_rows_are_kept_once(small_config, tmp_path):
    run_cv(small_config)
    shutil.copy(tmp_path / "cv" / "default_folds.csv", tmp_path / "cv" / "copy_folds.csv")
    copy = pd.read_csv(tmp_path / "cv" / "copy_folds.csv").assign(config="copy")
    copy.to_csv(tmp_path / "cv" / "copy_folds.csv", index=False)

    folds = load_cv_folds(tmp_path)
    assert len(folds) == len(MODEL_NAMES) * 2
    assert set(folds["config"]) == {"copy"}  # first file alphabetically wins


def test_run_final_writes_outputs(small_config, df, tmp_path):
    results = run_final(small_config)
    assert 0 <= results["nn"]["accuracy"] <= 1

    run_dir = tmp_path / "default"
    assert json.loads((run_dir / "metrics.json").read_text())["test"].keys() == results.keys()
    for name in ("loss_curves", "roc_curves", "confusion_matrix"):
        assert (run_dir / "figures" / f"{name}.png").exists()
    assert (tmp_path / "trials.csv").exists()

    predictor = Predictor.load(run_dir / "model.joblib")
    assert predictor.model_name == small_config.models[0]
    assert predictor.threshold == results["nn"]["threshold"]
    probabilities = predictor.predict_proba(df.head(10))
    assert probabilities.shape == (10,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_tune_returns_a_valid_config(small_config, tmp_path):
    config = dataclasses.replace(small_config, max_epochs=5)
    tuned, study = tune(config, n_trials=2)
    assert tuned.name == "tuned"
    assert len(study.trials) == 2

    path = tmp_path / "tuned.yaml"
    save_config(tuned, path)
    assert load_config(path).hidden_sizes == tuned.hidden_sizes


def test_cli_train_and_predict(small_config, df, tmp_path, capsys):
    config_path = tmp_path / "small.yaml"
    save_config(dataclasses.replace(small_config, models=["nn+gradient_boosting"]), config_path)
    assert yaml.safe_load(config_path.read_text())["output_dir"] == "."

    main(["train", "--config", str(config_path), "--max-epochs", "3"])
    model_path = tmp_path / small_config.name / "model.joblib"
    assert model_path.exists()

    main(["predict", "--model", str(model_path), "--glucose", "148", "--bmi", "33.6",
          "--age", "50", "--pedigree", "0.627"])
    assert "probability" in capsys.readouterr().out

    patients = tmp_path / "patients.csv"
    df.head(5)[list(FEATURE_COLUMNS)].to_csv(patients, index=False)
    predictions = tmp_path / "predictions.csv"
    main(["predict", "--model", str(model_path), "--input", str(patients),
          "--output", str(predictions)])
    assert set(pd.read_csv(predictions)["prediction"]) <= {"diabetes", "no diabetes"}
