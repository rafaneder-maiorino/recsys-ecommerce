"""DVC stage 2: encode ids and build the temporal train/val/test split."""

import json
import logging

import pandas as pd

from recsys.config import get_settings
from recsys.features.splitting import drop_cold_items, encode_ids, temporal_leave_last_out
from recsys.pipelines.params import load_params

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Produce train/val/test parquet files plus the catalog metadata."""
    settings = get_settings()
    params = load_params("feature_eng")
    interactions = pd.read_parquet(settings.processed_data_dir / "interactions.parquet")
    encoded, catalog = encode_ids(interactions)
    train, val, test = temporal_leave_last_out(
        encoded, val_size=params["val_size"], test_size=params["test_size"]
    )
    val, test = drop_cold_items(train, val, test)
    for name, frame in {"train": train, "val": val, "test": test}.items():
        path = settings.processed_data_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(frame))
    catalog_path = settings.processed_data_dir / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    logger.info("Catalog: %s", catalog)


if __name__ == "__main__":
    main()
