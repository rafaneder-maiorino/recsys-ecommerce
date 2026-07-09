"""MLflow helpers: tracking setup and data-to-model lineage tags."""

import logging
from pathlib import Path

import mlflow
import yaml

from recsys.config import Settings

logger = logging.getLogger(__name__)


def setup_mlflow(settings: Settings) -> None:
    """Point MLflow at the configured server and experiment.

    Args:
        settings: Project settings holding URI and experiment name.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    logger.info("MLflow tracking at %s", settings.mlflow_tracking_uri)


def read_dataset_version(dvc_file: Path = Path("data/raw.dvc")) -> str:
    """Read the DVC hash of the raw dataset for lineage tagging.

    Tagging each run with this hash links every model to the exact
    data snapshot that produced it (auditable data lineage).

    Args:
        dvc_file: Path to the ``.dvc`` metadata file of the dataset.

    Returns:
        The md5 hash recorded by DVC, or ``"unknown"`` when absent.
    """
    if not dvc_file.exists():
        logger.warning("%s not found; tagging data version as unknown", dvc_file)
        return "unknown"
    meta = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
    return str(meta["outs"][0]["md5"])
