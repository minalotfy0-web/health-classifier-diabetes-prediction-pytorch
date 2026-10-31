"""Command-line failures and the `predict` command against the committed model.

Error-handling tests only assert that the command fails (non-zero exit or an exception),
not how, so they stay valid when messages or exception types are tidied up.
"""

import dataclasses

import joblib
import pandas as pd
import pytest
import yaml

from diabetes_classifier.cli import main
from diabetes_classifier.config import save_config
from diabetes_classifier.data import FEATURE_COLUMNS

PATIENT_FLAGS = ["--glucose", "148", "--bmi", "33.6", "--age", "50", "--pedigree", "0.627"]


def assert_fails(argv: list[str]) -> None:
    """The command must not finish normally."""
    with pytest.raises((SystemExit, Exception)) as excinfo:
        main(argv)
    if isinstance(excinfo.value, SystemExit):
        assert excinfo.value.code not in (0, None)


def write_config(tmp_path, small_config, **overrides):
    path = tmp_path / "run.yaml"
    config = dataclasses.replace(small_config, output_dir=str(tmp_path / "out"), **overrides)
    save_config(config, path)
    return path


# Argument parsing


def test_no_arguments_exits_with_usage_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
    assert "usage" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("argv", [["frobnicate"], ["--seed", "1"], ["predict", "--nope"]])
def test_unknown_command_or_option_exits_with_usage_error(argv):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--seed", "abc"],
        ["train", "--max-epochs", "1.5"],
        ["train", "--lr", "fast"],
        ["train", "--hidden-sizes", "a", "b"],
        ["cv", "--cv-folds", "two"],
        ["tune", "--trials", "many"],
        ["predict", "--glucose", "high"],
        ["predict", "--age"],
    ],
)
def test_bad_argument_values_exit_with_usage_error(argv):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


@pytest.mark.parametrize("command", [[], ["predict"], ["train"], ["cv"], ["tune"]])
def test_help_exits_cleanly(command, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([*command, "--help"])
    assert excinfo.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


# Config problems


def test_missing_config_file_fails(tmp_path):
    for command in ("train", "cv", "tune"):
        assert_fails([command, "--config", str(tmp_path / "nope.yaml")])


def test_config_with_unknown_key_fails(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text(yaml.safe_dump({"learning_rte": 0.1, "output_dir": "out"}), encoding="utf-8")
    assert_fails(["train", "--config", str(path)])


def test_config_with_invalid_value_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"test_size": 2.0, "output_dir": "out"}), encoding="utf-8")
    assert_fails(["train", "--config", str(path)])


def test_override_that_makes_the_config_invalid_fails(tmp_path, small_config):
    path = write_config(tmp_path, small_config)
    assert_fails(["train", "--config", str(path), "--ensemble-size", "0"])


def test_one_bad_config_among_several_fails(tmp_path, small_config):
    good = write_config(tmp_path, small_config)
    assert_fails(["cv", "--config", str(good), str(tmp_path / "nope.yaml")])


def test_cv_with_a_single_fold_fails(tmp_path, small_config):
    path = write_config(tmp_path, small_config)
    assert_fails(["cv", "--config", str(path), "--cv-folds", "1"])


# Data problems


def test_missing_dataset_fails(tmp_path, small_config):
    path = write_config(tmp_path, small_config, data_path=str(tmp_path / "missing.csv"))
    assert_fails(["train", "--config", str(path)])
    assert_fails(["cv", "--config", str(path)])


def test_dataset_without_the_target_column_fails(tmp_path, small_config, df):
    csv = tmp_path / "no_target.csv"
    df.drop(columns="Outcome").to_csv(csv, index=False)
    path = write_config(tmp_path, small_config, data_path=str(csv))
    assert_fails(["train", "--config", str(path)])


def test_empty_dataset_fails(tmp_path, small_config):
    csv = tmp_path / "empty.csv"
    csv.write_text("", encoding="utf-8")
    path = write_config(tmp_path, small_config, data_path=str(csv))
    assert_fails(["train", "--config", str(path)])


def test_dataset_with_text_in_a_feature_column_fails(tmp_path, small_config, df):
    csv = tmp_path / "text.csv"
    df.astype({"Glucose": object}).assign(Glucose="high").to_csv(csv, index=False)
    path = write_config(tmp_path, small_config, data_path=str(csv))
    assert_fails(["train", "--config", str(path)])


# predict


def test_predict_without_patient_data_fails(tuned_model_path):
    assert_fails(["predict", "--model", str(tuned_model_path)])


@pytest.mark.parametrize("flags", [["--age", "50"], ["--pedigree", "0.6"], ["--glucose", "148"]])
def test_predict_without_the_required_features_fails(tuned_model_path, flags):
    assert_fails(["predict", "--model", str(tuned_model_path), *flags])


def test_predict_with_a_missing_model_fails(tmp_path):
    assert_fails(["predict", "--model", str(tmp_path / "nope.joblib"), *PATIENT_FLAGS])


def test_predict_with_a_file_that_is_not_a_model_fails(tmp_path):
    other = tmp_path / "other.joblib"
    joblib.dump({"not": "a predictor"}, other)
    garbage = tmp_path / "garbage.joblib"
    garbage.write_bytes(b"garbage")
    for path in (other, garbage):
        assert_fails(["predict", "--model", str(path), *PATIENT_FLAGS])


def test_predict_with_a_missing_input_file_fails(tuned_model_path, tmp_path):
    missing = tmp_path / "x.csv"
    assert_fails(["predict", "--model", str(tuned_model_path), "--input", str(missing)])


def test_predict_with_an_empty_csv_fails(tuned_model_path, tmp_path):
    csv = tmp_path / "empty.csv"
    csv.write_text("", encoding="utf-8")
    assert_fails(["predict", "--model", str(tuned_model_path), "--input", str(csv)])


def test_predict_with_missing_csv_columns_fails(tuned_model_path, tmp_path, patients):
    csv = tmp_path / "partial.csv"
    patients.drop(columns=["BMI"]).to_csv(csv, index=False)
    assert_fails(["predict", "--model", str(tuned_model_path), "--input", str(csv)])


def test_predict_with_text_in_the_csv_fails(tuned_model_path, tmp_path, patients):
    csv = tmp_path / "text.csv"
    patients.astype({"Age": object}).assign(Age="old").to_csv(csv, index=False)
    assert_fails(["predict", "--model", str(tuned_model_path), "--input", str(csv)])


def test_predict_to_an_unwritable_output_path_fails(tuned_model_path, tmp_path, patients):
    csv = tmp_path / "patients.csv"
    patients.to_csv(csv, index=False)
    output = tmp_path  # an existing folder cannot be written as a file
    argv = ["predict", "--model", str(tuned_model_path), "--input", str(csv)]
    assert_fails([*argv, "--output", str(output)])


def test_predict_with_blank_csv_cells_exits_with_an_error(tuned_model_path, tmp_path, patients):
    blank = patients.head(3).assign(Glucose=[148.0, None, 90.0])
    csv = tmp_path / "blank.csv"
    blank.to_csv(csv, index=False)
    output = tmp_path / "out.csv"
    argv = ["predict", "--model", str(tuned_model_path), "--input", str(csv)]
    with pytest.raises(SystemExit) as excinfo:
        main([*argv, "--output", str(output)])
    assert excinfo.value.code not in (0, None)
    assert not output.exists()


def test_predict_with_input_and_flags_fails(tuned_model_path, tmp_path, patients):
    csv = tmp_path / "patients.csv"
    patients.to_csv(csv, index=False)
    assert_fails(["predict", "--model", str(tuned_model_path), "--input", str(csv), "--age", "5"])


def test_predict_creates_the_output_folder(tuned_model_path, tmp_path, patients):
    csv = tmp_path / "patients.csv"
    patients.to_csv(csv, index=False)
    output = tmp_path / "new" / "folder" / "out.csv"
    argv = ["predict", "--model", str(tuned_model_path), "--input", str(csv)]
    main([*argv, "--output", str(output)])
    assert len(pd.read_csv(output)) == len(patients)


# predict, success paths


def test_predict_from_flags_prints_a_prediction(tuned_model_path, capsys):
    main(["predict", "--model", str(tuned_model_path), *PATIENT_FLAGS])
    out = capsys.readouterr().out
    assert "probability" in out
    assert "diabetes" in out
    assert "nn+gradient_boosting" in out
    assert "not for clinical use" in out


def test_predict_treats_omitted_values_as_zero(tuned_model_path, capsys):
    required = ["--age", "50", "--pedigree", "0.627"]
    main(["predict", "--model", str(tuned_model_path), *required])
    omitted = capsys.readouterr().out
    zeros = ["--glucose", "0", "--blood-pressure", "0", "--bmi", "0", "--insulin", "0"]
    main(["predict", "--model", str(tuned_model_path), *required, *zeros])
    assert capsys.readouterr().out == omitted


def test_predict_csv_to_output_file(tuned_model_path, tmp_path, patients, capsys):
    csv = tmp_path / "patients.csv"
    patients.to_csv(csv, index=False)
    output = tmp_path / "predictions.csv"
    argv = ["predict", "--model", str(tuned_model_path), "--input", str(csv)]
    main([*argv, "--output", str(output)])
    assert f"Saved {len(patients)} predictions" in capsys.readouterr().out
    result = pd.read_csv(output)
    assert list(result.columns) == [*FEATURE_COLUMNS, "probability", "prediction"]
    assert len(result) == len(patients)
    assert result["probability"].between(0, 1).all()
    assert set(result["prediction"]) <= {"diabetes", "no diabetes"}


def test_predict_csv_to_the_terminal(tuned_model_path, tmp_path, patients, capsys):
    csv = tmp_path / "patients.csv"
    patients.head(3).to_csv(csv, index=False)
    main(["predict", "--model", str(tuned_model_path), "--input", str(csv)])
    out = capsys.readouterr().out
    assert "prediction" in out
    assert len(out.strip().splitlines()) >= 4  # header + 3 rows + footer
