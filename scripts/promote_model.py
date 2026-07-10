"""Governed promotion of the trained model to Production.

Registers the model produced by the last ``dvc repro`` in the MLflow
Model Registry and promotes it None -> Staging -> Production, but only
if the evaluation metrics pass the quality gates in ``params.yaml``.

Usage:
    uv run python scripts/promote_model.py
"""

import logging
import sys

from recsys.config import get_settings
from recsys.pipelines.params import load_params
from recsys.training.registry import promote_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> int:
    """Run the governed promotion flow.

    Returns:
        ``0`` when promoted, ``1`` when blocked by quality gates.
    """
    return promote_model(get_settings(), load_params("promote"))


if __name__ == "__main__":
    sys.exit(main())
