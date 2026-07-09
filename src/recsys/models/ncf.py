"""Neural Collaborative Filtering (He et al., 2017) in PyTorch.

User and item embeddings are concatenated and passed through an MLP,
trained as binary classification with sampled negatives (implicit
feedback). Event weights scale the loss of positive samples, and early
stopping monitors the validation loss.
"""

import copy
import logging

import numpy as np
import pandas as pd
import torch
from torch import nn

from recsys.models.base import RecommenderModel
from recsys.models.factory import ModelFactory

logger = logging.getLogger(__name__)


class _NCFNet(nn.Module):
    """Embedding-based MLP scoring network."""

    def __init__(
        self, n_users: int, n_items: int, embedding_dim: int, hidden_dims: list[int], dropout: float
    ) -> None:
        """Build embeddings and the MLP tower.

        Args:
            n_users: Size of the user embedding table.
            n_items: Size of the item embedding table.
            embedding_dim: Dimension of each embedding.
            hidden_dims: Sizes of the hidden MLP layers.
            dropout: Dropout applied between hidden layers.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        layers: list[nn.Module] = []
        input_dim = 2 * embedding_dim
        for hidden in hidden_dims:
            layers += [nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Return the raw relevance logit for each user-item pair.

        Args:
            users: Tensor ``(n,)`` of user indices.
            items: Tensor ``(n,)`` of item indices.

        Returns:
            Tensor ``(n,)`` of logits.
        """
        features = torch.cat([self.user_embedding(users), self.item_embedding(items)], dim=1)
        return self.mlp(features).squeeze(-1)


@ModelFactory.register("ncf")
class NCFModel(RecommenderModel):
    """Neural Collaborative Filtering with sampled implicit negatives."""

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 4096,
        epochs: int = 20,
        n_negatives: int = 4,
        patience: int = 3,
        seed: int = 42,
    ) -> None:
        """Store hyperparameters; the network is built lazily in fit.

        Args:
            embedding_dim: Dimension of user and item embeddings.
            hidden_dims: MLP hidden layer sizes (default ``[64, 32]``).
            dropout: Dropout rate between hidden layers.
            learning_rate: Adam learning rate.
            batch_size: Training mini-batch size.
            epochs: Maximum number of epochs.
            n_negatives: Sampled negatives per positive interaction.
            patience: Epochs without val-loss improvement before stop.
            seed: Seed for negative sampling.
        """
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_negatives = n_negatives
        self.patience = patience
        self.seed = seed
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._net: _NCFNet | None = None
        self._n_items = 0
        self._seen: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame, validation: pd.DataFrame | None = None) -> "NCFModel":
        """Train with BCE over positives plus sampled negatives.

        Args:
            interactions: Training frame (``user_id``, ``item_id``,
                ``weight``).
            validation: Holdout frame monitored for early stopping.

        Returns:
            The fitted model (weights from the best epoch).
        """
        device = _pick_device()
        n_users = int(interactions["user_id"].max()) + 1
        self._n_items = int(interactions["item_id"].max()) + 1
        self._seen = interactions.groupby("user_id")["item_id"].agg(set).to_dict()
        self._net = _NCFNet(
            n_users, self._n_items, self.embedding_dim, self.hidden_dims, self.dropout
        ).to(device)
        logger.info("Training NCF on %s (%d users, %d items)", device, n_users, self._n_items)
        self._training_loop(interactions, validation, device)
        self._net.to("cpu").eval()
        return self

    def _training_loop(
        self, train: pd.DataFrame, validation: pd.DataFrame | None, device: str
    ) -> None:
        """Run the epoch loop with early stopping on validation loss.

        Args:
            train: Training interactions.
            validation: Validation interactions (may be ``None``).
            device: Torch device string.
        """
        assert self._net is not None
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        rng = np.random.default_rng(self.seed)
        best_loss, best_state, stale = float("inf"), None, 0
        for epoch in range(self.epochs):
            train_loss = self._run_epoch(train, device, optimizer, rng)
            val_loss = self._validation_loss(validation, device, rng)
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            logger.info("Epoch %d: train=%.4f val=%.4f", epoch + 1, train_loss, val_loss)
            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_state = copy.deepcopy(self._net.state_dict())
            else:
                stale += 1
            if stale >= self.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, self.patience)
                break
        if best_state is not None:
            self._net.load_state_dict(best_state)

    def _run_epoch(
        self,
        train: pd.DataFrame,
        device: str,
        optimizer: torch.optim.Optimizer,
        rng: np.random.Generator,
    ) -> float:
        """Train one epoch and return the mean weighted BCE loss.

        Args:
            train: Training interactions.
            device: Torch device string.
            optimizer: Optimizer updating the network.
            rng: Generator used for negative sampling.

        Returns:
            Mean training loss over the epoch.
        """
        assert self._net is not None
        self._net.train()
        users, items, labels, weights = _build_samples(train, self._n_items, self.n_negatives, rng)
        order = rng.permutation(len(users))
        total, batches = 0.0, 0
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            loss = self._batch_loss(
                users[batch], items[batch], labels[batch], weights[batch], device
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, batches = total + float(loss.item()), batches + 1
        return total / max(batches, 1)

    def _batch_loss(
        self,
        users: np.ndarray,
        items: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        device: str,
    ) -> torch.Tensor:
        """Weighted BCE-with-logits loss for one mini-batch.

        Args:
            users: User indices of the batch.
            items: Item indices of the batch.
            labels: Binary labels (1 positive, 0 sampled negative).
            weights: Per-sample loss weights (event weights).
            device: Torch device string.

        Returns:
            Scalar loss tensor.
        """
        assert self._net is not None
        logits = self._net(
            torch.as_tensor(users, device=device), torch.as_tensor(items, device=device)
        )
        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            torch.as_tensor(labels, dtype=torch.float32, device=device),
            weight=torch.as_tensor(weights, dtype=torch.float32, device=device),
        )

    def _validation_loss(
        self, validation: pd.DataFrame | None, device: str, rng: np.random.Generator
    ) -> float:
        """Compute the loss on the validation split (no gradients).

        Args:
            validation: Validation interactions (``None`` returns nan).
            device: Torch device string.
            rng: Generator used for negative sampling.

        Returns:
            Mean validation loss, or ``nan`` without validation data.
        """
        if validation is None or self._net is None:
            return float("nan")
        self._net.eval()
        users, items, labels, weights = _build_samples(
            validation, self._n_items, self.n_negatives, rng
        )
        with torch.no_grad():
            loss = self._batch_loss(users, items, labels, weights, device)
        return float(loss.item())

    def score(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Score pairs with the trained network (CPU inference).

        Args:
            user_ids: Array ``(n,)`` of user ids.
            item_ids: Array ``(n,)`` of item ids, paired with users.

        Returns:
            Array ``(n,)`` of logits (monotonic in relevance).
        """
        assert self._net is not None
        scores = np.empty(len(user_ids), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(user_ids), 100_000):
                end = start + 100_000
                logits = self._net(
                    torch.as_tensor(user_ids[start:end]), torch.as_tensor(item_ids[start:end])
                )
                scores[start:end] = logits.numpy()
        return scores

    def recommend(self, user_ids: np.ndarray, top_k: int = 10) -> np.ndarray:
        """Recommend the highest-scoring unseen items per user.

        Args:
            user_ids: Users to score.
            top_k: List size per user.

        Returns:
            Array ``(len(user_ids), top_k)`` of item ids.
        """
        all_items = np.arange(self._n_items)
        output = np.full((len(user_ids), top_k), -1, dtype=int)
        for row, user in enumerate(user_ids):
            scores = self.score(np.full(self._n_items, user), all_items)
            scores[list(self._seen.get(int(user), set()))] = -np.inf
            output[row] = np.argsort(-scores)[:top_k]
        return output


def _pick_device() -> str:
    """Select the best available torch device (mps > cuda > cpu)."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
        return "cuda"
    return "cpu"


def _build_samples(
    interactions: pd.DataFrame, n_items: int, n_negatives: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble positives plus uniformly sampled negatives.

    Negatives may rarely collide with true positives; with 104k items
    the collision rate is negligible and standard practice accepts it.

    Args:
        interactions: Frame of positive interactions.
        n_items: Catalog size for uniform sampling.
        n_negatives: Negatives generated per positive.
        rng: Seeded generator (reproducible sampling).

    Returns:
        Tuple of ``(users, items, labels, weights)`` arrays.
    """
    pos_users = interactions["user_id"].to_numpy(dtype=np.int64)
    pos_items = interactions["item_id"].to_numpy(dtype=np.int64)
    pos_weights = interactions["weight"].to_numpy(dtype=np.float32)
    neg_users = np.repeat(pos_users, n_negatives)
    neg_items = rng.integers(0, n_items, size=len(neg_users), dtype=np.int64)
    users = np.concatenate([pos_users, neg_users])
    items = np.concatenate([pos_items, neg_items])
    labels = np.concatenate([np.ones(len(pos_users)), np.zeros(len(neg_users))])
    weights = np.concatenate([pos_weights, np.ones(len(neg_users), dtype=np.float32)])
    return users, items, labels.astype(np.float32), weights
