"""Model Registry governance: quality gates and stage promotion.

Implements the governed promotion flow: a model version is only
promoted to Production after passing metric quality gates, and every
promotion carries audit tags (who approved, why, and which exact data
snapshot trained it).

MLflow stages (None -> Staging -> Production) are deprecated since
2.9 in favor of aliases; this module applies BOTH so the registry is
compliant with the classic flow while remaining future-proof.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from recsys.config import Settings
from recsys.training.tracking import setup_mlflow

logger = logging.getLogger(__name__)


class RecommenderWrapper(mlflow.pyfunc.PythonModel):
    """Pyfunc adapter exposing ``recommend`` through the MLflow API."""

    def __init__(self, model: Any) -> None:
        """Wrap a fitted :class:`RecommenderModel`.

        Args:
            model: Any fitted model implementing ``recommend``.
        """
        self._model = model

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> list[list[int]]:
        """Return top-k recommendations for each requested user.

        Args:
            context: MLflow context (unused).
            model_input: DataFrame with a ``user_id`` column.
            params: Optional dict supporting ``top_k`` (default 10).

        Returns:
            One list of recommended item ids per input row.
        """
        top_k = int((params or {}).get("top_k", 10))
        users = model_input["user_id"].to_numpy()
        return self._model.recommend(users, top_k=top_k).tolist()


def failed_gates(metrics: dict[str, float], gates: dict[str, float]) -> list[str]:
    """List every quality gate the metrics do not satisfy.

    Args:
        metrics: Evaluation metrics of the candidate model.
        gates: Minimum acceptable value per metric name.

    Returns:
        Names of failed gates (empty when the model may be promoted).
    """
    return [name for name, minimum in gates.items() if metrics.get(name, float("-inf")) < minimum]


def promote_model(settings: Settings, params: dict[str, Any]) -> int:
    """Register the current model and promote it under governance.

    Steps: quality gate -> log pyfunc into the training run ->
    register version -> Staging -> Production (archiving previous) ->
    alias + audit tags + description.

    Args:
        settings: Project settings (tracking URI, paths).
        params: The ``promote`` section of ``params.yaml``.

    Returns:
        Process exit code: ``0`` on promotion, ``1`` on gate failure.
    """
    metrics = json.loads(Path("metrics.json").read_text(encoding="utf-8"))
    failed = failed_gates(metrics, params["gates"])
    if failed:
        logger.error("Quality gate FAILED for %s; promotion blocked.", failed)
        return 1
    logger.info("Quality gates passed: %s", params["gates"])
    setup_mlflow(settings)
    run_id = (settings.models_dir / "mlflow_run_id.txt").read_text(encoding="utf-8").strip()
    version = _register_version(settings, params["registered_model_name"], run_id)
    _apply_governance(params, run_id, version, metrics)
    logger.info(
        "Model '%s' version %s promoted to Production.",
        params["registered_model_name"],
        version,
    )
    return 0


def _register_version(settings: Settings, name: str, run_id: str) -> str:
    """Log the fitted model as pyfunc inside its run and register it.

    Args:
        settings: Project settings (models directory).
        name: Registered model name.
        run_id: Training run that produced the model.

    Returns:
        The new registered model version.
    """
    model = pickle.loads((settings.models_dir / "model.pkl").read_bytes())
    with mlflow.start_run(run_id=run_id):
        info = mlflow.pyfunc.log_model(
            name="registry_model", python_model=RecommenderWrapper(model)
        )
    version = mlflow.register_model(info.model_uri, name).version
    logger.info("Registered '%s' version %s from run %s", name, version, run_id)
    return version


def _apply_governance(
    params: dict[str, Any], run_id: str, version: str, metrics: dict[str, float]
) -> None:
    """Promote through stages and attach the audit trail.

    Args:
        params: The ``promote`` section of ``params.yaml``.
        run_id: Source training run (for lineage tags).
        version: Registered model version being promoted.
        metrics: Metrics recorded on the version description.
    """
    name = params["registered_model_name"]
    client = MlflowClient()
    client.transition_model_version_stage(name, version, "Staging")
    logger.info("Version %s -> Staging", version)
    client.transition_model_version_stage(
        name, version, "Production", archive_existing_versions=True
    )
    logger.info("Version %s -> Production (previous versions archived)", version)
    client.set_registered_model_alias(name, "production", version)
    run_tags = client.get_run(run_id).data.tags
    client.set_model_version_tag(name, version, "approved_by", params["approved_by"])
    client.set_model_version_tag(name, version, "approval_notes", params["approval_notes"])
    for tag in ("train_data_version", "model_name"):
        if tag in run_tags:
            client.set_model_version_tag(name, version, tag, run_tags[tag])
    client.update_model_version(
        name, version, description=f"Promoted after quality gates. Metrics: {metrics}"
    )
