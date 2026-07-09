"""Base contract shared by every recommender in the project.

All models (PyTorch neural networks and scikit-learn baselines alike)
implement this interface so the training and evaluation pipelines can
treat them interchangeably (Liskov Substitution Principle).
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RecommenderModel(ABC):
    """Abstract recommender exposing a unified fit/recommend contract."""

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> "RecommenderModel":
        """Train the model on a user-item interactions frame.

        Args:
            interactions: DataFrame with at least the columns
                ``user_id``, ``item_id`` and ``weight``.

        Returns:
            The fitted model instance (enables fluent chaining).
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
