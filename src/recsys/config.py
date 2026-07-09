"""Centralized application settings loaded from the environment.

Every configurable value (tracking URI, credentials, seeds, paths)
lives here and is read from a ``.env`` file via Pydantic Settings, so
no configuration is ever hard-coded in pipeline code.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project settings resolved from environment variables / ``.env``.

    Attributes:
        mlflow_tracking_uri: MLflow server endpoint.
        mlflow_experiment_name: Experiment under which runs are logged.
        azure_storage_account: Storage account backing the DVC remote.
        azure_storage_sas_token: Read token for evaluators (optional).
        random_seed: Global seed for full reproducibility.
        raw_data_dir: Location of DVC-tracked raw data.
        processed_data_dir: Location of pipeline intermediate outputs.
        models_dir: Location of trained model artifacts.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "recsys-ecommerce"
    azure_storage_account: str = ""
    azure_storage_sas_token: str = ""
    random_seed: int = 42
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, process-wide :class:`Settings` instance."""
    return Settings()
