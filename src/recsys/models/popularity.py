"""Popularity baseline: recommends globally popular unseen items."""

import logging

import numpy as np
import pandas as pd

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory

logger = logging.getLogger(__name__)


@ModelFactory.register("popularity")
class PopularityModel(RecommenderModel):
    """Non-personalized baseline ranking items by weighted popularity.

    Every neural recommender must beat this model to justify its
    complexity; it is also the hardest baseline to beat on hit rate
    in highly popularity-skewed catalogs.
    """

    def __init__(self, top_n: int = 500) -> None:
        """Configure the size of the popularity head kept after fit.

        Args:
            top_n: Number of most popular items retained for serving.
        """
        self.top_n = top_n
        self._item_scores: np.ndarray = np.empty(0, dtype=float)
        self._ranked_items: np.ndarray = np.empty(0, dtype=int)
        self._seen: dict[int, set[int]] = {}

    def fit(
        self, interactions: pd.DataFrame, validation: pd.DataFrame | None = None
    ) -> "PopularityModel":
        """Rank items by summed event weight and memorize seen items.

        Args:
            interactions: Frame with ``user_id``, ``item_id``, ``weight``.
            validation: Unused by this non-parametric baseline.

        Returns:
            The fitted model.
        """
        n_items = int(interactions["item_id"].max()) + 1
        sums = interactions.groupby("item_id")["weight"].sum()
        self._item_scores = np.zeros(n_items, dtype=float)
        self._item_scores[sums.index.to_numpy()] = sums.to_numpy()
        self._ranked_items = np.argsort(-self._item_scores)[: self.top_n]
        self._seen = interactions.groupby("user_id")["item_id"].agg(set).to_dict()
        logger.info("Fitted popularity over %d items", n_items)
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Return the global popularity of each item (user-independent).

        Args:
            user_ids: Ignored; present to satisfy the contract.
            item_ids: Items to score.

        Returns:
            Popularity scores for the requested items.
        """
        return self._item_scores[item_ids]

    def recommend(self, user_ids: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Return the most popular items each user has not seen yet.

        Args:
            user_ids: Users to score.
            top_k: List size per user.

        Returns:
            Array ``(len(user_ids), top_k)``; ``-1`` pads short lists.
        """
        output = np.full((len(user_ids), top_k), -1, dtype=int)
        for row, user in enumerate(user_ids):
            seen = self._seen.get(user, set())
            recs = [item for item in self._ranked_items if item not in seen][:top_k]
            output[row, : len(recs)] = recs
        return output
