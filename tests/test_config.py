import dataclasses
from pathlib import Path

import pytest
import yaml

from diabetes_classifier.config import MODEL_NAMES, Config, load_config, save_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
BLANK_PATHS = {"data_path": "", "output_dir": ""}


def write_yaml(path: Path, content: dict) -> Path:
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_defaults_are_valid():
    config = Config()
    assert config.models == list(MODEL_NAMES)
    assert 0 < config.val_fraction_of_development < 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_size": 0.6, "val_size": 0.4},
        {"test_size": 0.9, "val_size": 0.3},
        {"test_size": 0.0, "val_size": 0.0},
        {"dropout_layers": [2]},
        {"optimizer": "rmsprop"},
        {"pos_weight": "heavy"},
        {"threshold": "median"},
        {"models": ["nn", "svm"]},
        {"ensemble_size": 0},
    ],
)
def test_invalid_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_unknown_field_is_rejected():
    with pytest.raises(TypeError):
        Config(learning_rte=0.1)


def test_wrong_type_is_rejected():
    with pytest.raises((TypeError, ValueError)):
        Config(test_size="0.2")


def test_valid_edge_values_are_accepted():
    Config(test_size=0.5, val_size=0.49)
    Config(dropout_layers=[0, 1], hidden_sizes=[8, 4])
    Config(pos_weight=None, threshold=0.5, optimizer="sgd", ensemble_size=1)
    Config(pos_weight=2.5, threshold="youden", models=["logistic_regression"])


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"test_size": -0.1, "val_size": 0.5}, id="negative-test-size"),
        pytest.param({"val_size": 0.0}, id="zero-val-size"),
        pytest.param({"threshold": 1.5}, id="threshold-above-1"),
        pytest.param({"threshold": -0.2}, id="negative-threshold"),
        pytest.param({"dropout": 1.5}, id="dropout-above-1"),
        pytest.param({"dropout": -0.1}, id="negative-dropout"),
        pytest.param({"learning_rate": -1.0}, id="negative-learning-rate"),
        pytest.param({"max_epochs": 0}, id="zero-epochs"),
        pytest.param({"cv_folds": 1}, id="one-cv-fold"),
        pytest.param({"cv_repeats": 0}, id="zero-cv-repeats"),
        pytest.param({"log_every": 0}, id="zero-log-every"),
        pytest.param({"patience": 0}, id="zero-patience"),
        pytest.param({"models": []}, id="no-models"),
        pytest.param({"batch_size": 0}, id="zero-batch-size"),
        pytest.param({"hidden_sizes": [8, 0]}, id="zero-width-layer"),
        pytest.param({"pos_weight": -1.0}, id="negative-pos-weight"),
    ],
)
def test_nonsensical_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_load_config_fills_in_defaults(tmp_path):
    path = write_yaml(tmp_path / "mine.yaml", {"seed": 7})
    config = load_config(path)
    assert config.seed == 7
    assert config.name == "mine"  # defaults to the file name
    assert config.hidden_sizes == Config().hidden_sizes


def test_load_config_keeps_an_explicit_name(tmp_path):
    path = write_yaml(tmp_path / "mine.yaml", {"name": "custom"})
    assert load_config(path).name == "custom"


def test_empty_file_gives_the_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    loaded = dataclasses.replace(load_config(path), name="x", **BLANK_PATHS)
    assert loaded == dataclasses.replace(Config(), name="x", **BLANK_PATHS)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_unknown_key_raises(tmp_path):
    path = write_yaml(tmp_path / "typo.yaml", {"learning_rte": 0.1})
    with pytest.raises(ValueError):
        load_config(path)


def test_invalid_value_in_file_raises(tmp_path):
    path = write_yaml(tmp_path / "bad.yaml", {"optimizer": "rmsprop"})
    with pytest.raises(ValueError):
        load_config(path)


def test_malformed_yaml_raises(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("seed: [1, 2\nname: {", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_config(path)


def test_non_mapping_yaml_raises(tmp_path):
    path = tmp_path / "text.yaml"
    path.write_text("just some text", encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        load_config(path)


def test_relative_paths_resolve_against_the_config_folder(tmp_path):
    folder = tmp_path / "configs"
    folder.mkdir()
    path = write_yaml(folder / "c.yaml", {"data_path": "../data/x.csv", "output_dir": "out"})
    config = load_config(path)
    assert Path(config.data_path).resolve() == (tmp_path / "data" / "x.csv").resolve()
    assert Path(config.output_dir).resolve() == (folder / "out").resolve()


def test_absolute_paths_are_kept(tmp_path):
    absolute = tmp_path / "data.csv"
    path = write_yaml(tmp_path / "c.yaml", {"data_path": str(absolute)})
    assert Path(load_config(path).data_path) == absolute


def test_save_and_load_round_trip(tmp_path):
    folder = tmp_path / "configs"
    folder.mkdir()
    config = Config(
        name="round",
        data_path=str(tmp_path / "data" / "d.csv"),
        output_dir=str(tmp_path / "out"),
        hidden_sizes=[4, 2],
        dropout_layers=[1],
        pos_weight=None,
        threshold=0.4,
    )
    path = folder / "round.yaml"
    save_config(config, path)
    loaded = load_config(path)

    assert Path(loaded.data_path).resolve() == Path(config.data_path).resolve()
    assert Path(loaded.output_dir).resolve() == Path(config.output_dir).resolve()
    assert dataclasses.replace(loaded, **BLANK_PATHS) == dataclasses.replace(config, **BLANK_PATHS)


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_committed_configs_are_valid(path):
    config = load_config(path)
    assert Path(config.data_path).exists()
