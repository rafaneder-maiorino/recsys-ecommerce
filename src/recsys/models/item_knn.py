"""Item-based KNN baseline over the weighted interaction matrix."""

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory

logger = logging.getLogger(__name__)


@ModelFactory.register("item_knn")
class ItemKNNModel(RecommenderModel):
    """Neighborhood model: items similar to what the user interacted with.

    Similarity is the cosine between item columns of the user-item
    matrix, kept sparse end to end. Long user histories are capped to
    bound the cost of building each user profile at inference time.
    """

    def __init__(self, history_cap: int = 50) -> None:
        """Configure inference-time history truncation.

        Args:
            history_cap: Most recent interactions kept per user when
                building the preference profile.
        """
        self.history_cap = history_cap
        self._similarity: sparse.csr_matrix | None = None
        self._history: dict[int, np.ndarray] = {}

    def fit(
        self, interactions: pd.DataFrame, validation: pd.DataFrame | None = None
    ) -> "ItemKNNModel":
        """Build the item-item cosine similarity matrix.

        Args:
            interactions: Frame with ``user_id``, ``item_id``,
                ``weight`` and ``timestamp``.
            validation: Unused by this memory-based baseline.

        Returns:
            The fitted model.
        """
        matrix = _to_user_item_matrix(interactions)
        items_normalized = normalize(matrix.T.tocsr(), norm="l2", axis=1)
        self._similarity = (items_normalized @ items_normalized.T).tocsr()
        recent = interactions.sort_values("timestamp").groupby("user_id")["item_id"]
        self._history = {user: items.to_numpy()[-self.history_cap :] for user, items in recent}
        logger.info("Item-item similarity: %d items, %d nonzeros", *self._shape_stats())
        return self

    def _shape_stats(self) -> tuple[int, int]:
        """Return (n_items, nnz) of the similarity matrix for logging."""
        assert self._similarity is not None
        return self._similarity.shape[0], int(self._similarity.nnz)

    def _profile(self, user: int) -> np.ndarray:
        """Dense similarity profile of one user against all items.

        Args:
            user: User id whose training history seeds the profile.

        Returns:
            Array ``(n_items,)`` with accumulated similarity scores.
        """
        assert self._similarity is not None
        history = self._history.get(user)
        if history is None or len(history) == 0:
            return np.zeros(self._similarity.shape[0], dtype=np.float32)
        return np.asarray(self._similarity[history].sum(axis=0)).ravel()

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Score pairs by summed similarity to the user's history.

        Consecutive rows for the same user reuse one cached profile.

        Args:
            user_ids: Array ``(n,)`` of user ids.
            item_ids: Array ``(n,)`` of item ids, paired with users.

        Returns:
            Array ``(n,)`` of similarity scores.
        """
        scores = np.empty(len(user_ids), dtype=np.float32)
        cached_user, profile = None, None
        for row, (user, item) in enumerate(zip(user_ids, item_ids, strict=True)):
            if user != cached_user:
                cached_user, profile = user, self._profile(int(user))
            scores[row] = profile[item]
        return scores

    def recommend(self, user_ids: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Recommend the most similar unseen items per user.

        Args:
            user_ids: Users to score.
            top_k: List size per user.

        Returns:
            Array ``(len(user_ids), top_k)`` of item ids.
        """
        output = np.full((len(user_ids), top_k), -1, dtype=int)
        for row, user in enumerate(user_ids):
            profile = self._profile(int(user))
            profile[self._history.get(int(user), np.empty(0, dtype=int))] = -np.inf
            output[row] = np.argsort(-profile)[:top_k]
        return output


def _to_user_item_matrix(interactions: pd.DataFrame) -> sparse.csr_matrix:
    """Build the sparse weighted user-item matrix.

    Args:
        interactions: Frame with ``user_id``, ``item_id``, ``weight``.

    Returns:
        CSR matrix of shape ``(n_users, n_items)``.
    """
    n_users = int(interactions["user_id"].max()) + 1
    n_items = int(interactions["item_id"].max()) + 1
    return sparse.csr_matrix(
        (
            interactions["weight"].to_numpy(dtype=np.float32),
            (interactions["user_id"].to_numpy(), interactions["item_id"].to_numpy()),
        ),
        shape=(n_users, n_items),
    )
