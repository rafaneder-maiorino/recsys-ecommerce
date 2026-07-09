"""Id encoding and temporal splitting for implicit-feedback data.

A random split would leak future interactions into training, so the
split is strictly temporal per user: the most recent interaction goes
to test, the previous one to validation, everything else to train.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def encode_ids(interactions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map raw user/item ids to contiguous integer codes.

    Contiguous codes are required to index embedding tables directly.

    Args:
        interactions: Frame with raw ``user_id`` and ``item_id`` columns.

    Returns:
        A ``(frame, catalog)`` tuple: the frame with codes in place of
        raw ids, and a catalog with ``n_users`` and ``n_items``.
    """
    encoded = interactions.copy()
    encoded["user_id"] = pd.factorize(encoded["user_id"])[0]
    encoded["item_id"] = pd.factorize(encoded["item_id"])[0]
    catalog = {
        "n_users": int(encoded["user_id"].max()) + 1,
        "n_items": int(encoded["item_id"].max()) + 1,
    }
    logger.info("Encoded %(n_users)d users and %(n_items)d items", catalog)
    return encoded, catalog


def temporal_leave_last_out(
    interactions: pd.DataFrame, val_size: int = 1, test_size: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions per user by recency.

    Args:
        interactions: Frame with ``user_id`` and ``timestamp`` columns.
        val_size: Most-recent interactions per user (after test) for
            validation.
        test_size: Most-recent interactions per user for test.

    Returns:
        ``(train, val, test)`` frames covering all input rows.
    """
    ordered = interactions.sort_values(["user_id", "timestamp"], kind="stable")
    from_end = ordered.groupby("user_id").cumcount(ascending=False)
    test = ordered.loc[from_end < test_size]
    val = ordered.loc[(from_end >= test_size) & (from_end < test_size + val_size)]
    train = ordered.loc[from_end >= test_size + val_size]
    logger.info("Split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))
    return train, val, test


def drop_cold_items(train: pd.DataFrame, *holdouts: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Remove holdout rows whose item never appears in training.

    Items unseen at training time cannot be ranked by collaborative
    models, so keeping them would only distort the comparison.

    Args:
        train: Training interactions defining the known catalog.
        *holdouts: Validation/test frames to be filtered.

    Returns:
        The filtered holdout frames, in the order received.
    """
    known_items = set(train["item_id"].unique())
    filtered: list[pd.DataFrame] = []
    for frame in holdouts:
        kept = frame.loc[frame["item_id"].isin(known_items)]
        logger.info("Cold-item filter: %d -> %d rows", len(frame), len(kept))
        filtered.append(kept)
    return tuple(filtered)
