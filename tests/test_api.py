"""Integration tests for the inference API (TestClient)."""

import pickle

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from recsys.models.popularity import PopularityModel


@pytest.fixture()
def client(tmp_path, monkeypatch):
    interactions = pd.DataFrame(
        {"user_id": [0, 0, 1], "item_id": [1, 2, 3], "weight": [1.0, 5.0, 3.0]}
    )
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps(PopularityModel(top_n=5).fit(interactions)))
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from recsys.api.app import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_model_loaded(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_recommend_returns_ranked_items(client) -> None:
    response = client.get("/recommend/1", params={"top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1
    assert body["items"][0] == 2  # heaviest item not seen by user 1


def test_recommend_validates_top_k_bounds(client) -> None:
    assert client.get("/recommend/0", params={"top_k": 0}).status_code == 422
    assert client.get("/recommend/0", params={"top_k": 99}).status_code == 422
