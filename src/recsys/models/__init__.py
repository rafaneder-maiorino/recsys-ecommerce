"""Model implementations; importing this package registers them."""

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory
from recsys.models.popularity import PopularityModel

__all__ = ["ModelFactory", "PopularityModel", "RecommenderModel"]
