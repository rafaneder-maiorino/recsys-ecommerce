"""DVC stage 4: score the model on the temporal test split."""

import json
import logging
import pickle

import mlflow
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
    top_k = load_params("evaluate")["top_k"]
    model = pickle.loads((settings.models_dir / "model.pkl").read_bytes())
    test = pd.read_parquet(settings.processed_data_dir / "test.parquet")
    recommendations = model.recommend(test["user_id"].to_numpy(), top_k=top_k)
    metrics = compute_ranking_metrics(recommendations, test["item_id"].to_numpy(), top_k)
    _report(metrics)


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
