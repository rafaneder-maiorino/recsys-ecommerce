"""DVC stage 3: train the configured model and log the run to MLflow."""

import logging
import pickle

import mlflow
import pandas as pd

from recsys.config import get_settings
from recsys.models import ModelFactory
from recsys.pipelines.params import load_params
from recsys.seeding import set_global_seed
from recsys.training.tracking import read_dataset_version, setup_mlflow

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Fit the model selected in params.yaml inside a tracked MLflow run."""
    settings = get_settings()
    params = load_params("train")
    set_global_seed(params["seed"])
    train_df = pd.read_parquet(settings.processed_data_dir / "train.parquet")
    setup_mlflow(settings)
    with mlflow.start_run() as run:
        mlflow.set_tag("model_name", params["model"])
        mlflow.set_tag("train_data_version", read_dataset_version())
        mlflow.log_params({"model": params["model"], "seed": params["seed"]})
        mlflow.log_params(params.get("model_params", {}))
        model = ModelFactory.create(params["model"], **params.get("model_params", {}))
        model.fit(train_df)
        _persist(model, run.info.run_id)


def _persist(model: object, run_id: str) -> None:
    """Save the fitted model and the MLflow run id for later stages.

    Args:
        model: Fitted model instance to pickle.
        run_id: Active MLflow run identifier.
    """
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.models_dir / "model.pkl"
    model_path.write_bytes(pickle.dumps(model))
    mlflow.log_artifact(str(model_path))
    (settings.models_dir / "mlflow_run_id.txt").write_text(run_id, encoding="utf-8")
    logger.info("Saved %s under MLflow run %s", model_path, run_id)


if __name__ == "__main__":
    main()
