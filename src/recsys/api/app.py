"""Inference API serving the Production recommender.

Exposes the model promoted in the MLflow Registry through a minimal
FastAPI service. The model artifact is loaded once at startup from
``MODEL_PATH`` (baked into the cloud image at build time from the
Production version) and served from memory.

Local run:
    uv run uvicorn recsys.api.app:app --port 8000
"""

import logging
import os
import pickle
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

_state: dict[str, Any] = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model artifact once when the service starts.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the framework with the model in memory.
    """
    path = Path(os.environ.get("MODEL_PATH", "models/model.pkl"))
    _state["model"] = pickle.loads(path.read_bytes())
    logger.info("Model loaded from %s (%s)", path, type(_state["model"]).__name__)
    yield
    _state["model"] = None


app = FastAPI(
    title="recsys-ecommerce API",
    description="Top-k product recommendations (FIAP Tech Challenge Phase 2)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness/readiness probe used by container healthchecks.

    Returns:
        Service status and whether the model is in memory.
    """
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_k: int = Query(default=10, ge=1, le=50)) -> dict[str, Any]:
    """Return the top-k recommended item ids for a known user.

    Args:
        user_id: Encoded user id (as produced by the feature pipeline).
        top_k: Number of items to return (1-50).

    Returns:
        The user id and the ranked list of recommended item ids.

    Raises:
        HTTPException: 404 when the user is unknown to the model
            (cold start) and 503 when the model is not loaded.
    """
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        items = model.recommend(np.array([user_id]), top_k=top_k)[0]
    except (IndexError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=f"Unknown user {user_id} (cold start)") from exc
    return {"user_id": user_id, "items": [int(i) for i in items if i >= 0]}
