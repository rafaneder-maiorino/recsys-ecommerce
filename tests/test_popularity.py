"""Unit tests for the popularity baseline."""

import numpy as np
import pandas as pd

from recsys.models.popularity import PopularityModel


def _interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2],
            "item_id": [7, 8, 7, 9, 7],
            "weight": [5.0, 1.0, 5.0, 3.0, 5.0],
        }
    )


def test_fit_ranks_items_by_weighted_popularity() -> None:
    model = PopularityModel(top_n=3).fit(_interactions())
    recs = model.recommend(np.array([99]), top_k=3)
    assert recs[0].tolist() == [7, 9, 8]


def test_recommend_excludes_seen_items() -> None:
    model = PopularityModel(top_n=3).fit(_interactions())
    recs = model.recommend(np.array([0]), top_k=3)
    assert 7 not in recs[0]
    assert 8 not in recs[0]
    assert recs[0][0] == 9


def test_score_returns_global_popularity() -> None:
    import numpy as np

    model = PopularityModel(top_n=3).fit(_interactions())
    scores = model.score(np.array([0, 0]), np.array([7, 8]))
    assert scores[0] > scores[1]


def test_recommend_pads_short_lists_with_minus_one() -> None:
    model = PopularityModel(top_n=3).fit(_interactions())
    recs = model.recommend(np.array([0]), top_k=3)
    assert recs[0].tolist()[1:] == [-1, -1]
