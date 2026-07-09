"""Base contract shared by every recommender in the project.

All models (PyTorch neural networks and scikit-learn baselines alike)
implement this interface so the training and evaluation pipelines can
treat them interchangeably (Liskov Substitution Principle).
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RecommenderModel(ABC):
    """Abstract recommender exposing fit/score/recommend contracts."""

    @abstractmethod
    def fit(
        self, interactions: pd.DataFrame, validation: pd.DataFrame | None = None
    ) -> "RecommenderModel":
        """Train the model on a user-item interactions frame.

        Args:
            interactions: DataFrame with at least the columns
                ``user_id``, ``item_id`` and ``weight``.
            validation: Optional holdout frame with the same columns,
                used by models that support early stopping.

        Returns:
            The fitted model instance (enables fluent chaining).
        """

    @abstractmethod
    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Score user-item pairs (higher means more relevant).

        Args:
            user_ids: Array ``(n,)`` of user ids.
            item_ids: Array ``(n,)`` of item ids, paired with users.

        Returns:
            Array ``(n,)`` of relevance scores.
        """

    @abstractmethod
    def recommend(self, user_ids: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Produce top-k item recommendations for each user.

        Args:
            user_ids: Array of user identifiers to score.
            top_k: Number of items to return per user.

        Returns:
            Array of shape ``(len(user_ids), top_k)`` with item ids
            ordered by descending relevance.
        """
