"""Ranking metrics for top-k recommendation under leave-last-out.

Under this protocol each user has exactly one held-out item, so
``recall@k`` equals ``hit_rate@k``; we therefore report hit rate,
precision, NDCG and MRR as four distinct metrics.
"""

import numpy as np


def hit_ranks(recommendations: np.ndarray, true_items: np.ndarray) -> np.ndarray:
    """Locate each user's held-out item inside their recommendation list.

    Args:
        recommendations: Array ``(n_users, k)`` of recommended item ids.
        true_items: Array ``(n_users,)`` with the held-out item per user.

    Returns:
        Array ``(n_users,)`` with the 0-based rank of the hit, or ``-1``
        when the item is not in the top-k list.
    """
    matches = recommendations == true_items[:, None]
    return np.where(matches.any(axis=1), matches.argmax(axis=1), -1)


def hit_rate_at_k(ranks: np.ndarray) -> float:
    """Fraction of users whose held-out item appears in the top-k.

    Args:
        ranks: Output of :func:`hit_ranks`.

    Returns:
        Hit rate in ``[0, 1]``.
    """
    return float((ranks >= 0).mean())


def precision_at_k(ranks: np.ndarray, k: int) -> float:
    """Average precision@k with a single relevant item per user.

    Args:
        ranks: Output of :func:`hit_ranks`.
        k: Size of the recommendation list.

    Returns:
        Mean precision in ``[0, 1/k]``.
    """
    return float((ranks >= 0).sum() / (len(ranks) * k))


def ndcg_at_k(ranks: np.ndarray) -> float:
    """Mean NDCG@k with binary relevance and one relevant item.

    Args:
        ranks: Output of :func:`hit_ranks`.

    Returns:
        Mean NDCG in ``[0, 1]``.
    """
    gains = np.zeros(len(ranks), dtype=float)
    hits = ranks >= 0
    gains[hits] = 1.0 / np.log2(ranks[hits] + 2)
    return float(gains.mean())


def mrr_at_k(ranks: np.ndarray) -> float:
    """Mean reciprocal rank of the held-out item.

    Args:
        ranks: Output of :func:`hit_ranks`.

    Returns:
        MRR in ``[0, 1]``.
    """
    reciprocal = np.zeros(len(ranks), dtype=float)
    hits = ranks >= 0
    reciprocal[hits] = 1.0 / (ranks[hits] + 1)
    return float(reciprocal.mean())


def compute_ranking_metrics(
    recommendations: np.ndarray, true_items: np.ndarray, k: int
) -> dict[str, float]:
    """Compute the full metric suite for a batch of recommendations.

    Args:
        recommendations: Array ``(n_users, >=k)`` of recommended items.
        true_items: Array ``(n_users,)`` with the held-out item per user.
        k: Cutoff applied to the recommendation lists.

    Returns:
        Mapping of metric name (suffixed with ``@k``) to value.
    """
    ranks = hit_ranks(recommendations[:, :k], true_items)
    return {
        f"hit_rate_at_{k}": hit_rate_at_k(ranks),
        f"precision_at_{k}": precision_at_k(ranks, k),
        f"ndcg_at_{k}": ndcg_at_k(ranks),
        f"mrr_at_{k}": mrr_at_k(ranks),
    }
