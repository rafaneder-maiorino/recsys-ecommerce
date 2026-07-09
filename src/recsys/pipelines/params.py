"""Loader for the stage sections of params.yaml."""

from pathlib import Path
from typing import Any

import yaml


def load_params(stage: str, path: Path = Path("params.yaml")) -> dict[str, Any]:
    """Load one stage's parameter section from ``params.yaml``.

    Args:
        stage: Top-level section name (e.g. ``"train"``).
        path: Location of the params file.

    Returns:
        The parameter mapping for the requested stage.
    """
    params = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(params[stage])
