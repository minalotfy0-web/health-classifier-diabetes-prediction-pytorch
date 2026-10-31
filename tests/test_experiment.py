import dataclasses
import json

import pandas as pd
import yaml

from diabetes_classifier.cli import main
from diabetes_classifier.config import MODEL_NAMES, load_config, save_config
from diabetes_classifier.experiment import plot_cv_results, run_cv, run_final
from diabetes_classifier.models import fit_predict
from diabetes_classifier.tune import tune


def test_every_model_predicts_probabilities(small_config, fold):
    outputs, histories = fit_predict(small_config, fold, seed=0)
    assert list(outputs) == list(MODEL_NAMES)
    assert len(histories) == small_config.ensemble_size
    for output in outputs.values():
        assert output.eval_probs.shape == fold.y_eval.shape
        assert ((output.eval_probs >= 0) & (output.eval_probs <= 1)).all()


def test_run_cv_writes_summary(small_config, tmp_path):
    summary = run_cv(small_config)
    assert set(summary["model"]) == set(MODEL_NAMES)
    folds = pd.read_csv(tmp_path / "cv" / "default_folds.csv")
    assert len(folds) == len(MODEL_NAMES) * 2
    assert plot_cv_results(tmp_path).exists()


def test_run_final_writes_outputs(small_config, tmp_path):
    results = run_final(small_config)
    assert 0 <= results["nn"]["accuracy"] <= 1

    run_dir = tmp_path / "default"
    assert json.loads((run_dir / "metrics.json").read_text())["test"].keys() == results.keys()
    for name in ("loss_curves", "roc_curves", "confusion_matrix"):
        assert (run_dir / "figures" / f"{name}.png").exists()
    assert (tmp_path / "trials.csv").exists()


def test_tune_returns_a_valid_config(small_config, tmp_path):
    config = dataclasses.replace(small_config, max_epochs=5)
    tuned, study = tune(config, n_trials=2)
    assert tuned.name == "tuned"
    assert len(study.trials) == 2

    path = tmp_path / "tuned.yaml"
    save_config(tuned, path)
    assert load_config(path).hidden_sizes == tuned.hidden_sizes


def test_cli_train(small_config, tmp_path):
    config_path = tmp_path / "small.yaml"
    save_config(small_config, config_path)
    assert yaml.safe_load(config_path.read_text())["output_dir"] == "."

    main(["train", "--config", str(config_path), "--max-epochs", "3"])
    assert (tmp_path / small_config.name / "metrics.json").exists()
