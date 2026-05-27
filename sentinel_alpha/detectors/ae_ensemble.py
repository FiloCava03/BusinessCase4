"""Ensemble of K denoising autoencoders with different seeds.

Direct implementation of hint #4 from `EarlyWarningSystemPoliMI.ipynb`:

    > Or use an ensemble of Autoencoders with different initializations (they
    > provide different results). You can:
    >   - Average their anomaly scores
    >   - Or use their disagreement (variance) as a proxy for predictive
    >     uncertainty: models that disagree on a sample may be revealing an
    >     ambiguous or borderline case -- which could be flagged for deeper
    >     analysis.

We expose both:
    - `score_samples(X)` returns the *mean* reconstruction MSE across members
      (used as the anomaly score by the stacker, same contract as `AEDetector`).
    - `score_with_uncertainty(X)` additionally returns the across-member
      standard deviation, the "disagreement" signal.
"""
from __future__ import annotations
import numpy as np

from sentinel_alpha.detectors.base import AnomalyDetector
from sentinel_alpha.detectors.autoencoder import AEDetector
from sentinel_alpha.config import SEED


class AEEnsembleDetector(AnomalyDetector):
    """K independent AEDetectors with different seeds; mean = score, std = uncertainty."""

    def __init__(
        self, n_members: int = 5, base_seed: int = SEED,
        bottleneck: int = 8, hidden: int = 32, mid: int = 16,
        dropout: float = 0.2, noise_std: float = 0.05,
        lr: float = 1e-3, batch_size: int = 64,
        max_epochs: int = 200, patience: int = 15, val_frac: float = 0.15,
    ) -> None:
        self.n_members = n_members
        self.base_seed = base_seed
        # AE hyperparameters duplicated so sklearn's `get_params` discovers them.
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.mid = mid
        self.dropout = dropout
        self.noise_std = noise_std
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac

    def _make_member(self, seed: int) -> AEDetector:
        return AEDetector(
            bottleneck=self.bottleneck, hidden=self.hidden, mid=self.mid,
            dropout=self.dropout, noise_std=self.noise_std,
            lr=self.lr, batch_size=self.batch_size,
            max_epochs=self.max_epochs, patience=self.patience,
            val_frac=self.val_frac, random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AEEnsembleDetector":
        self.members_: list[AEDetector] = []
        # Deterministic seeds: base, base+1, base+2, ...
        for i in range(self.n_members):
            member = self._make_member(seed=self.base_seed + i)
            member.fit(X, y)
            self.members_.append(member)
        return self

    def _stack_scores(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([m.score_samples(X) for m in self.members_])

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self._stack_scores(X).mean(axis=0)

    def score_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_score, std_score) over the K members. std_score is the
        per-week disagreement, useable as an uncertainty proxy."""
        stacked = self._stack_scores(X)
        return stacked.mean(axis=0), stacked.std(axis=0, ddof=0)
