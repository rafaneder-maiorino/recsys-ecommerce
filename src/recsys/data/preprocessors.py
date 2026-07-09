"""Preprocessing strategies for interaction data (Strategy pattern).

Each transformation step is an interchangeable strategy, so the DVC
``preprocess`` stage can compose them from configuration without any
``if/else`` chains in pipeline code.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


class PreprocessingStrategy(ABC):
    """Single, composable transformation over an interactions frame."""

    @abstractmethod
    def apply(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Transform and return the interactions frame.

        Args:
            interactions: DataFrame with columns ``user_id``,
                ``item_id``, ``event`` and ``timestamp``.

        Returns:
            The transformed DataFrame (input is never mutated).
        """


class MinInteractionsFilter(PreprocessingStrategy):
    """Drops users with fewer than ``min_interactions`` events.

    Sparse users add noise and inflate the user embedding table
    without contributing learnable signal.
    """

    def __init__(self, min_interactions: int = 5) -> None:
        """Store the minimum number of events per user.

        Args:
            min_interactions: Users below this threshold are removed.
        """
        self.min_interactions = min_interactions

    def apply(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Filter out low-activity users. See base class."""
        counts = interactions.groupby("user_id")["item_id"].transform("count")
        kept = interactions.loc[counts >= self.min_interactions].copy()
        logger.info(
            "MinInteractionsFilter(min=%d): %d -> %d rows",
            self.min_interactions,
            len(interactions),
            len(kept),
        )
        return kept


class EventWeighting(PreprocessingStrategy):
    """Maps event types to implicit-feedback weights.

    RetailRocket events carry increasing intent: ``view`` <
    ``addtocart`` < ``transaction``. The weight becomes the training
    signal for the implicit-feedback loss.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Store the event-to-weight mapping.

        Args:
            weights: Mapping like ``{"view": 1.0, "addtocart": 3.0,
                "transaction": 5.0}``. Unknown events are dropped.
        """
        self.weights = weights or {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}

    def apply(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Attach a ``weight`` column derived from event type. See base class."""
        out = interactions.copy()
        out["weight"] = out["event"].map(self.weights)
        dropped = int(out["weight"].isna().sum())
        if dropped:
            logger.warning("EventWeighting: dropping %d rows with unknown events", dropped)
        return out.dropna(subset=["weight"])


def run_pipeline(
    interactions: pd.DataFrame, strategies: Sequence[PreprocessingStrategy]
) -> pd.DataFrame:
    """Apply strategies in order and return the final frame.

    Args:
        interactions: Raw interactions DataFrame.
        strategies: Ordered sequence of strategies to apply.

    Returns:
        The DataFrame after every strategy has been applied.
    """
    result = interactions
    for strategy in strategies:
        result = strategy.apply(result)
    return result
