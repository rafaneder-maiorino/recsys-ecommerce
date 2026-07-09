"""Unit tests for the NCF model (tiny synthetic data, CPU)."""

import numpy as np
import pandas as pd
import torch

from recsys.models.ncf import NCFModel, _build_samples


def _interactions(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, 10, n),
            "item_id": rng.integers(0, 15, n),
            "weight": rng.choice([1.0, 3.0, 5.0], n),
            "timestamp": np.arange(n),
        }
    )


def test_build_samples_shapes_and_labels() -> None:
    users, items, labels, weights = _build_samples(
        _interactions(50), n_items=15, n_negatives=4, rng=np.random.default_rng(1)
    )
    assert len(users) == len(items) == len(labels) == len(weights) == 250
    assert labels[:50].sum() == 50 and labels[50:].sum() == 0


def test_fit_runs_and_records_history() -> None:
    torch.manual_seed(0)
    model = NCFModel(embedding_dim=4, hidden_dims=[8], epochs=2, batch_size=64, patience=5)
    model.fit(_interactions(), validation=_interactions(40))
    assert len(model.history_["train_loss"]) == 2
    assert model.history_["train_loss"][1] < model.history_["train_loss"][0] * 1.5


def test_score_and_recommend_shapes() -> None:
    torch.manual_seed(0)
    model = NCFModel(embedding_dim=4, hidden_dims=[8], epochs=1, batch_size=64)
    model.fit(_interactions())
    scores = model.score(np.array([0, 1, 2]), np.array([3, 4, 5]))
    assert scores.shape == (3,)
    recs = model.recommend(np.array([0]), top_k=5)
    assert recs.shape == (1, 5)
    assert len(set(recs[0])) == 5
