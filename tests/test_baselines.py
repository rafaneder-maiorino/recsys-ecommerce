"""Unit tests for the ItemKNN and SVD baselines."""

import numpy as np
import pandas as pd

from recsys.models.item_knn import ItemKNNModel
from recsys.models.svd import SVDModel


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2, 2, 3],
            "item_id": [0, 1, 0, 1, 0, 2, 2],
            "weight": [1.0, 1.0, 1.0, 3.0, 1.0, 5.0, 1.0],
            "timestamp": range(7),
        }
    )


def test_item_knn_scores_co_consumed_items_higher() -> None:
    model = ItemKNNModel().fit(_interactions())
    users = np.array([0, 0])
    # user 0 consumed items 0 and 1; item 1's neighbors should outscore item 2
    scores = model.score(users, np.array([1, 2]))
    assert scores[0] > scores[1]


def test_item_knn_recommend_excludes_history() -> None:
    model = ItemKNNModel().fit(_interactions())
    recs = model.recommend(np.array([0]), top_k=1)
    assert recs[0, 0] == 2


def test_svd_reconstructs_interaction_preference() -> None:
    model = SVDModel(n_components=2).fit(_interactions())
    scores = model.score(np.array([2, 2]), np.array([2, 1]))
    assert scores[0] > scores[1]  # user 2 strongly consumed item 2, never item 1


def test_svd_recommend_shape_and_exclusion() -> None:
    model = SVDModel(n_components=2).fit(_interactions())
    recs = model.recommend(np.array([3]), top_k=2)
    assert recs.shape == (1, 2)
    assert 2 not in recs[0]  # item 2 already seen by user 3
