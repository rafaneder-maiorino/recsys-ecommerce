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
        self._ranked_items: np.ndarray = np.empty(0, dtype=int)
        self._seen: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> "PopularityModel":
        """Rank items by summed event weight and memorize seen items.

        Args:
            interactions: Frame with ``user_id``, ``item_id``, ``weight``.

        Returns:
            The fitted model.
        """
        scores = interactions.groupby("item_id")["weight"].sum()
        self._ranked_items = scores.nlargest(self.top_n).index.to_numpy()
        self._seen = interactions.groupby("user_id")["item_id"].agg(set).to_dict()
        logger.info("Fitted popularity head with %d items", len(self._ranked_items))
        return self

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
