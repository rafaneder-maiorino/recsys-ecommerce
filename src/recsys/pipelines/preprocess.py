"""DVC stage 1: clean raw events into weighted interactions."""

import logging

import pandas as pd

from recsys.config import get_settings
from recsys.data.preprocessors import EventWeighting, MinInteractionsFilter, run_pipeline
from recsys.pipelines.params import load_params

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Read raw events, apply preprocessing strategies, save parquet."""
    settings = get_settings()
    params = load_params("preprocess")
    events = pd.read_csv(settings.raw_data_dir / "events.csv")
    events = events.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
    strategies = [
        EventWeighting(params["event_weights"]),
        MinInteractionsFilter(params["min_interactions"]),
    ]
    interactions = run_pipeline(events, strategies)
    columns = ["user_id", "item_id", "event", "weight", "timestamp"]
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    output = settings.processed_data_dir / "interactions.parquet"
    interactions[columns].to_parquet(output, index=False)
    logger.info("Wrote %d interactions to %s", len(interactions), output)


if __name__ == "__main__":
    main()
