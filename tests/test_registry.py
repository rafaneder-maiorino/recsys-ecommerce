"""Unit tests for quality gates and the pyfunc wrapper."""

import pandas as pd

from recsys.models.popularity import PopularityModel
from recsys.training.registry import RecommenderWrapper, failed_gates


def test_failed_gates_empty_when_all_pass() -> None:
    metrics = {"hit_rate_at_10": 0.66, "ndcg_at_10": 0.47}
    gates = {"hit_rate_at_10": 0.60, "ndcg_at_10": 0.40}
    assert failed_gates(metrics, gates) == []


def test_failed_gates_lists_violations() -> None:
    metrics = {"hit_rate_at_10": 0.50, "ndcg_at_10": 0.47}
    gates = {"hit_rate_at_10": 0.60, "ndcg_at_10": 0.40}
    assert failed_gates(metrics, gates) == ["hit_rate_at_10"]


def test_failed_gates_treats_missing_metric_as_failure() -> None:
    assert failed_gates({}, {"ndcg_at_10": 0.1}) == ["ndcg_at_10"]


def test_wrapper_predict_returns_topk_per_row() -> None:
    interactions = pd.DataFrame(
        {"user_id": [0, 0, 1], "item_id": [1, 2, 1], "weight": [1.0, 5.0, 1.0]}
    )
    wrapper = RecommenderWrapper(PopularityModel(top_n=3).fit(interactions))
    output = wrapper.predict(None, pd.DataFrame({"user_id": [0, 1]}), params={"top_k": 2})
    assert len(output) == 2
    assert len(output[0]) == 2
