"""DVC stage 4: sampled-ranking evaluation on the temporal test split.

Protocol from the NCF paper (He et al., 2017): each held-out item is
ranked against ``n_negatives`` items the user never interacted with.
Full-catalog ranking with an MLP scorer would require ~8 billion
forward passes; sampled ranking keeps the comparison across models
identical and computationally honest.
"""

import json
import logging
import pickle

import mlflow
import numpy as np
import pandas as pd

from recsys.config import get_settings
from recsys.evaluation.metrics import compute_ranking_metrics
from recsys.pipelines.params import load_params
from recsys.training.tracking import setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Evaluate the trained model and log metrics to DVC and MLflow."""
    settings = get_settings()
    params = load_params("evaluate")
    model = pickle.loads((settings.models_dir / "model.pkl").read_bytes())
    train = pd.read_parquet(settings.processed_data_dir / "train.parquet")
    test = pd.read_parquet(settings.processed_data_dir / "test.parquet")
    users, candidates = _build_candidates(train, test, params["n_negatives"], params["seed"])
    ranked = _rank_candidates(model, users, candidates)
    metrics = compute_ranking_metrics(ranked, test["item_id"].to_numpy(), params["top_k"])
    _report(metrics)


def _build_candidates(
    train: pd.DataFrame, test: pd.DataFrame, n_negatives: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample unseen negative items for every test interaction.

    Args:
        train: Training frame defining seen items and the catalog.
        test: Test frame with one held-out interaction per user.
        n_negatives: Negatives ranked against each positive.
        seed: Seed for reproducible sampling.

    Returns:
        ``(users, candidates)``: users ``(n,)`` and candidate item
        matrix ``(n, n_negatives + 1)`` with the true item in column 0.
    """
    rng = np.random.default_rng(seed)
    n_items = int(train["item_id"].max()) + 1
    seen = train.groupby("user_id")["item_id"].agg(set).to_dict()
    users = test["user_id"].to_numpy()
    positives = test["item_id"].to_numpy()
    candidates = np.empty((len(users), n_negatives + 1), dtype=np.int64)
    candidates[:, 0] = positives
    for row, (user, positive) in enumerate(zip(users, positives, strict=True)):
        candidates[row, 1:] = _sample_negatives(
            rng, n_items, seen.get(user, set()), int(positive), n_negatives
        )
    return users, candidates


def _sample_negatives(
    rng: np.random.Generator, n_items: int, seen: set[int], positive: int, count: int
) -> np.ndarray:
    """Draw ``count`` items the user has not interacted with.

    Args:
        rng: Seeded generator.
        n_items: Catalog size.
        seen: Items in the user's training history.
        positive: The held-out item (always excluded).
        count: Number of negatives to return.

    Returns:
        Array ``(count,)`` of negative item ids.
    """
    negatives: list[int] = []
    while len(negatives) < count:
        draws = rng.integers(0, n_items, size=2 * count)
        negatives += [int(d) for d in draws if d not in seen and d != positive]
    return np.asarray(negatives[:count], dtype=np.int64)


def _rank_candidates(model: object, users: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Order each user's candidate list by model score (descending).

    Args:
        model: Fitted model exposing ``score(user_ids, item_ids)``.
        users: Array ``(n,)`` of user ids.
        candidates: Matrix ``(n, m)`` of candidate item ids.

    Returns:
        Matrix ``(n, m)`` of item ids sorted by relevance per row.
    """
    n_rows, n_cols = candidates.shape
    flat_scores = model.score(np.repeat(users, n_cols), candidates.ravel())
    order = np.argsort(-flat_scores.reshape(n_rows, n_cols), axis=1, kind="stable")
    return np.take_along_axis(candidates, order, axis=1)


def _report(metrics: dict[str, float]) -> None:
    """Write metrics.json (DVC) and log the same values to MLflow.

    Args:
        metrics: Metric name to value mapping from the evaluation.
    """
    settings = get_settings()
    with open("metrics.json", "w", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, indent=2) + "\n")
    run_id = (settings.models_dir / "mlflow_run_id.txt").read_text(encoding="utf-8").strip()
    setup_mlflow(settings)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
