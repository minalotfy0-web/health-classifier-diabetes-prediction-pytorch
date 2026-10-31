import math

import torch

from diabetes_classifier.config import Config
from diabetes_classifier.experiment import run_experiment
from diabetes_classifier.model import DiabetesClassifier
from diabetes_classifier.train import train


def test_training_lowers_the_loss(splits):
    torch.manual_seed(0)
    model = DiabetesClassifier(splits.n_features, [16, 8])
    history = train(model, splits, epochs=300, learning_rate=0.1)

    assert len(history.train_loss) == len(history.val_loss) == 300
    assert history.train_loss[-1] < history.train_loss[0]
    assert 1 <= history.best_epoch <= 300
    assert math.isfinite(history.best_val_loss)


def test_best_epoch_weights_are_restored(splits):
    torch.manual_seed(0)
    model = DiabetesClassifier(splits.n_features, [16, 8])
    history = train(model, splits, epochs=100, learning_rate=0.1)

    model.eval()
    with torch.no_grad():
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(splits.X_val), splits.y_val
        ).item()
    assert math.isclose(val_loss, history.best_val_loss, rel_tol=1e-5)


def test_run_experiment_writes_outputs(data_path, tmp_path):
    config = Config(data_path=str(data_path), epochs=20, log_every=10, output_dir=str(tmp_path))
    trial = run_experiment(config)

    assert 0 <= trial["test_accuracy"] <= 1
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "trials.csv").exists()
    for name in ("loss_curves", "roc_curve", "confusion_matrix"):
        assert (tmp_path / "figures" / f"{name}.png").exists()
