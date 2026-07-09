"""Unit tests for ranking metrics."""

import numpy as np

from recsys.evaluation.metrics import compute_ranking_metrics, hit_ranks


def test_hit_ranks_locates_items_and_flags_misses() -> None:
    recs = np.array([[5, 3, 9], [1, 2, 4], [7, 8, 6]])
    truth = np.array([3, 99, 7])
    assert hit_ranks(recs, truth).tolist() == [1, -1, 0]


def test_metrics_perfect_recommendations() -> None:
    recs = np.array([[1, 2], [3, 4]])
    truth = np.array([1, 3])
    metrics = compute_ranking_metrics(recs, truth, k=2)
    assert metrics["hit_rate_at_2"] == 1.0
    assert metrics["ndcg_at_2"] == 1.0
    assert metrics["mrr_at_2"] == 1.0
    assert metrics["precision_at_2"] == 0.5


def test_metrics_all_misses_are_zero() -> None:
    recs = np.array([[1, 2], [3, 4]])
    truth = np.array([9, 9])
    metrics = compute_ranking_metrics(recs, truth, k=2)
    assert all(value == 0.0 for value in metrics.values())


def test_ndcg_discounts_lower_ranks() -> None:
    first = compute_ranking_metrics(np.array([[1, 2]]), np.array([1]), k=2)
    second = compute_ranking_metrics(np.array([[2, 1]]), np.array([1]), k=2)
    assert first["ndcg_at_2"] > second["ndcg_at_2"]
