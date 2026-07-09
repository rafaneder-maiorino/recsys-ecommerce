"""Matrix-factorization baseline via scikit-learn TruncatedSVD."""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory
from recsys.models.item_knn import _to_user_item_matrix

logger = logging.getLogger(__name__)


@ModelFactory.register("svd")
class SVDModel(RecommenderModel):
    """Latent-factor baseline: truncated SVD of the interaction matrix.

    The linear counterpart of the neural model — comparing it against
    the NCF isolates how much the non-linear MLP actually adds.
    """

    def __init__(self, n_components: int = 64, random_state: int = 42) -> None:
        """Configure the factorization rank.

        Args:
            n_components: Number of latent factors.
            random_state: Seed for the randomized SVD solver.
        """
        self.n_components = n_components
        self.random_state = random_state
        self._user_factors: np.ndarray = np.empty((0, 0))
        self._item_factors: np.ndarray = np.empty((0, 0))
        self._seen: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame, validation: pd.DataFrame | None = None) -> "SVDModel":
        """Factorize the weighted user-item matrix.

        Args:
            interactions: Frame with ``user_id``, ``item_id``, ``weight``.
            validation: Unused by this baseline.

        Returns:
            The fitted model.
        """
        matrix = _to_user_item_matrix(interactions)
        svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        self._user_factors = svd.fit_transform(matrix)
        self._item_factors = svd.components_.T
        self._seen = interactions.groupby("user_id")["item_id"].agg(set).to_dict()
        logger.info(
            "SVD fitted: %d factors, %.1f%% variance explained",
            self.n_components,
            100 * float(svd.explained_variance_ratio_.sum()),
        )
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Score pairs by the dot product of latent factors.

        Args:
            user_ids: Array ``(n,)`` of user ids.
            item_ids: Array ``(n,)`` of item ids, paired with users.

        Returns:
            Array ``(n,)`` of reconstruction scores.
        """
        return np.einsum("ij,ij->i", self._user_factors[user_ids], self._item_factors[item_ids])

    def recommend(self, user_ids: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Recommend the highest-scoring unseen items per user.

        Args:
            user_ids: Users to score.
            top_k: List size per user.

        Returns:
            Array ``(len(user_ids), top_k)`` of item ids.
        """
        output = np.full((len(user_ids), top_k), -1, dtype=int)
        for row, user in enumerate(user_ids):
            scores = self._item_factors @ self._user_factors[int(user)]
            seen = list(self._seen.get(int(user), set()))
            scores[seen] = -np.inf
            output[row] = np.argsort(-scores)[:top_k]
        return output
