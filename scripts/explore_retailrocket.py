"""Quick exploratory summary of the RetailRocket events dataset.

Prints the numbers that drive our preprocessing decisions (event
distribution, sparsity, user activity), so each choice in
``params.yaml`` is backed by data instead of guesswork.

Usage:
    uv run python scripts/explore_retailrocket.py
"""

import logging

import pandas as pd

from recsys.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("explore")


def load_events() -> pd.DataFrame:
    """Load the raw RetailRocket events file.

    Returns:
        DataFrame with columns timestamp, visitorid, event, itemid,
        transactionid.
    """
    path = get_settings().raw_data_dir / "events.csv"
    events = pd.read_csv(path)
    logger.info("Loaded %s: %s rows", path, f"{len(events):,}")
    return events


def summarize(events: pd.DataFrame) -> None:
    """Log the statistics that inform preprocessing choices.

    Args:
        events: Raw events frame from :func:`load_events`.
    """
    users = events["visitorid"].nunique()
    items = events["itemid"].nunique()
    density = len(events) / (users * items)
    logger.info("Unique users: %s | unique items: %s", f"{users:,}", f"{items:,}")
    logger.info("Interaction matrix density: %.6f%%", density * 100)
    logger.info("Event distribution:\n%s", events["event"].value_counts().to_string())
    _summarize_user_activity(events)
    _summarize_time_range(events)


def _summarize_user_activity(events: pd.DataFrame) -> None:
    """Log user activity quantiles to calibrate the min-interactions filter.

    Args:
        events: Raw events frame.
    """
    activity = events.groupby("visitorid").size()
    quantiles = activity.quantile([0.5, 0.75, 0.9, 0.99]).to_string()
    logger.info("Interactions per user (quantiles):\n%s", quantiles)
    for threshold in (2, 3, 5):
        kept = (activity >= threshold).sum()
        logger.info("Users with >= %d interactions: %s", threshold, f"{kept:,}")


def _summarize_time_range(events: pd.DataFrame) -> None:
    """Log the dataset time span (basis for the temporal split).

    Args:
        events: Raw events frame with a millisecond ``timestamp``.
    """
    ts = pd.to_datetime(events["timestamp"], unit="ms")
    logger.info("Time range: %s -> %s (%d days)", ts.min(), ts.max(), (ts.max() - ts.min()).days)


if __name__ == "__main__":
    summarize(load_events())
