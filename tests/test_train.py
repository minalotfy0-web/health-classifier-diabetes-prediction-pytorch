import dataclasses
import math

import numpy as np
import torch

from diabetes_classifier.train import predict_proba, train_model


def test_training_lowers_the_loss(small_config, fold):
    config = dataclasses.replace(small_config, max_epochs=50, patience=None)
    _, history = train_model(config, fold, seed=0)

    assert len(history.train_loss) == len(history.val_loss) == 50
    assert history.train_loss[-1] < history.train_loss[0]
    assert 1 <= history.best_epoch <= 50
    assert math.isfinite(history.best_val_loss)


def test_early_stopping(small_config, fold):
    # With a zero learning rate the validation loss never improves after epoch 1
    config = dataclasses.replace(
        small_config, learning_rate=0.0, patience=5, max_epochs=100, lr_scheduler=False
    )
    _, history = train_model(config, fold, seed=0)
    assert history.best_epoch == 1
    assert len(history.val_loss) == 1 + 5


def test_best_epoch_weights_are_restored(small_config, fold):
    config = dataclasses.replace(small_config, pos_weight=None, patience=None)
    model, history = train_model(config, fold, seed=0)

    model.eval()
    with torch.no_grad():
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(torch.from_numpy(fold.X_val)), torch.from_numpy(fold.y_val)
        ).item()
    assert math.isclose(val_loss, history.best_val_loss, rel_tol=1e-6)


def test_full_batch_sgd_and_batch_norm_run(small_config, fold):
    for overrides in ({"optimizer": "sgd", "batch_size": None}, {"batch_norm": True}):
        config = dataclasses.replace(small_config, max_epochs=5, **overrides)
        model, _ = train_model(config, fold, seed=0)
        probs = predict_proba(model, fold.X_eval)
        assert probs.shape == fold.y_eval.shape
        assert np.all((probs >= 0) & (probs <= 1))
