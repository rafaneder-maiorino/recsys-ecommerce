"""Unit tests for preprocessing strategies (Strategy pattern)."""

import pandas as pd

from recsys.data.preprocessors import (
    EventWeighting,
    MinInteractionsFilter,
    run_pipeline,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 3, 3],
            "item_id": [10, 11, 12, 10, 13, 14],
            "event": ["view", "addtocart", "transaction", "view", "view", "bogus"],
            "timestamp": range(6),
        }
    )


def test_min_interactions_filter_drops_sparse_users() -> None:
    filtered = MinInteractionsFilter(min_interactions=2).apply(_sample_frame())
    assert set(filtered["user_id"]) == {1, 3}


def test_min_interactions_filter_does_not_mutate_input() -> None:
    frame = _sample_frame()
    MinInteractionsFilter(min_interactions=3).apply(frame)
    assert len(frame) == 6


def test_event_weighting_maps_known_and_drops_unknown() -> None:
    weighted = EventWeighting().apply(_sample_frame())
    assert "weight" not in _sample_frame().columns
    assert weighted.loc[weighted["event"] == "transaction", "weight"].iloc[0] == 5.0
    assert "bogus" not in set(weighted["event"])


def test_run_pipeline_applies_strategies_in_order() -> None:
    result = run_pipeline(
        _sample_frame(),
        [MinInteractionsFilter(min_interactions=2), EventWeighting()],
    )
    assert set(result["user_id"]) == {1, 3}
    assert "weight" in result.columns
