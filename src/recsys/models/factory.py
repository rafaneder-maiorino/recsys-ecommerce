"""Factory Method for recommender models.

Centralizes model instantiation so pipeline code never hard-codes a
concrete class: the model to build is chosen by name (typically read
from ``params.yaml``), keeping training scripts open for extension and
closed for modification (Open/Closed Principle).

Example:
    >>> @ModelFactory.register("popularity")
    ... class PopularityModel(RecommenderModel):
    ...     ...
    >>> model = ModelFactory.create("popularity", top_n=100)
"""

import logging
from collections.abc import Callable
from typing import Any

from recsys.models.base import RecommenderModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Creates :class:`RecommenderModel` instances from a string name."""

    _registry: dict[str, type[RecommenderModel]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[RecommenderModel]], type[RecommenderModel]]:
        """Class decorator that registers a model under ``name``.

        Args:
            name: Unique key used later by :meth:`create`.

        Returns:
            The decorator that performs the registration.

        Raises:
            ValueError: If ``name`` is already registered.
        """

        def decorator(model_cls: type[RecommenderModel]) -> type[RecommenderModel]:
            if name in cls._registry:
                raise ValueError(f"Model '{name}' is already registered.")
            cls._registry[name] = model_cls
            logger.debug("Registered model '%s' -> %s", name, model_cls.__name__)
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> RecommenderModel:
        """Instantiate a registered model by name.

        Args:
            name: Key previously passed to :meth:`register`.
            **kwargs: Forwarded to the model constructor.

        Returns:
            A ready-to-fit model instance.

        Raises:
            KeyError: If ``name`` was never registered.
        """
        if name not in cls._registry:
            available = sorted(cls._registry)
            raise KeyError(f"Unknown model '{name}'. Available: {available}")
        logger.info("Creating model '%s' with params %s", name, kwargs)
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return the sorted names of all registered models."""
        return sorted(cls._registry)
