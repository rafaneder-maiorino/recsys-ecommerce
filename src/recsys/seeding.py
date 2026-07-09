"""Global seeding utilities for reproducible experiments."""

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Fix every relevant RNG so experiments are fully reproducible.

    Seeds Python's ``random``, ``PYTHONHASHSEED``, NumPy and — when
    installed — PyTorch (CPU and MPS/CUDA generators share the seed).

    Args:
        seed: The seed applied to all random number generators.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    _seed_torch(seed)
    logger.info("Global seed set to %d", seed)


def _seed_torch(seed: int) -> None:
    """Seed PyTorch generators if the library is available.

    Args:
        seed: The seed forwarded to ``torch.manual_seed``.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a prod dependency
        logger.warning("PyTorch not installed; skipping torch seeding")
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        torch.cuda.manual_seed_all(seed)
