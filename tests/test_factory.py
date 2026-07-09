"""Unit tests for the ModelFactory (Factory Method pattern)."""

import numpy as np
import pandas as pd
import pytest

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory


@ModelFactory.register("dummy")
class DummyModel(RecommenderModel):
    """Trivial model used only to exercise the factory contract."""

    def __init__(self, top_n: int = 10) -> None:
        self.top_n = top_n

    def fit(
        self, interactions: pd.DataFrame, validation: pd.DataFrame | None = None
    ) -> "DummyModel":
        return self

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        return np.zeros(len(user_ids))

    def recommend(self, user_ids: np.ndarray, top_k: int = 10) -> np.ndarray:
        return np.zeros((len(user_ids), top_k), dtype=int)


def test_create_returns_registered_instance() -> None:
    model = ModelFactory.create("dummy", top_n=99)
    assert isinstance(model, DummyModel)
    assert model.top_n == 99


def test_create_unknown_name_raises_keyerror_with_options() -> None:
    with pytest.raises(KeyError, match="dummy"):
        ModelFactory.create("does-not-exist")


def test_duplicate_registration_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @ModelFactory.register("dummy")
        class Another(DummyModel):
            pass


def test_available_lists_registered_models() -> None:
    assert "dummy" in ModelFactory.available()
