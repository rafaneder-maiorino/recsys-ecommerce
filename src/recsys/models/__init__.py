"""Model implementations; importing this package registers them."""

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory
from recsys.models.item_knn import ItemKNNModel
from recsys.models.ncf import NCFModel
from recsys.models.popularity import PopularityModel
from recsys.models.svd import SVDModel

__all__ = [
    "ItemKNNModel",
    "ModelFactory",
    "NCFModel",
    "PopularityModel",
    "RecommenderModel",
    "SVDModel",
]
