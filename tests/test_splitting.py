"""Unit tests for id encoding and temporal splitting."""

import pandas as pd

from recsys.features.splitting import drop_cold_items, encode_ids, temporal_leave_last_out


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [10, 10, 10, 10, 20, 20, 20],
            "item_id": [100, 101, 102, 103, 100, 104, 105],
            "timestamp": [1, 2, 3, 4, 1, 2, 3],
            "weight": [1.0] * 7,
        }
    )


def test_encode_ids_produces_contiguous_codes() -> None:
    encoded, catalog = encode_ids(_frame())
    assert catalog == {"n_users": 2, "n_items": 6}
    assert sorted(encoded["user_id"].unique()) == [0, 1]


def test_split_sends_most_recent_to_test() -> None:
    train, val, test = temporal_leave_last_out(_frame(), val_size=1, test_size=1)
    assert test.groupby("user_id")["timestamp"].max().tolist() == [4, 3]
    assert len(test) == 2
    assert len(val) == 2
    assert len(train) == 3
    assert set(train["timestamp"]).isdisjoint(set(test["timestamp"]).union({99}))


def test_drop_cold_items_removes_unseen_catalog() -> None:
    train = pd.DataFrame({"item_id": [1, 2, 3]})
    holdout = pd.DataFrame({"item_id": [2, 99]})
    (filtered,) = drop_cold_items(train, holdout)
    assert filtered["item_id"].tolist() == [2]
