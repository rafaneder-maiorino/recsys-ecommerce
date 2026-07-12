"""Unit tests for settings and reproducibility helpers."""

from pathlib import Path

import numpy as np

from recsys.config import Settings
from recsys.seeding import set_global_seed


def test_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.random_seed == 42
    assert settings.mlflow_experiment_name == "recsys-ecommerce"
    assert settings.raw_data_dir == Path("data/raw")


def test_set_global_seed_makes_numpy_deterministic() -> None:
    set_global_seed(123)
    first = np.random.rand(5)
    set_global_seed(123)
    second = np.random.rand(5)
    assert np.allclose(first, second)
