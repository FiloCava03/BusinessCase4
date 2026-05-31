"""UMAP projection helpers for visualising the feature panel and the EWS output.

Two recommended figures for the report (cf. EarlyWarningSystemPoliMI.ipynb,
where Zenti uses UMAP to visualise TP / FP / TN / FN separation):

1. **All weeks coloured by calibrated p_t** -- shows that the stacker is not
   degenerate (it puts high probability mass on geometrically coherent
   regions of feature space, not on a single axis).
2. **Hold-out weeks coloured by confusion-matrix bucket** (TP / FP / TN / FN)
   -- direct visual replica of the professor's figure.

UMAP is non-deterministic across implementations; we fix ``random_state`` to
``config.SEED`` for reproducibility.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentinel_alpha.config import SEED


def umap_embed(
    F: pd.DataFrame | np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = SEED,
) -> np.ndarray:
    """Compute a 2-D UMAP embedding of the feature matrix.

    Parameters
    ----------
    F : DataFrame or ndarray of shape (n_weeks, n_features)
        The (already-standardised) feature panel to project.
    n_neighbors : int
        UMAP local connectivity; higher = more global structure.
    min_dist : float in [0, 1]
        UMAP local packing; lower = tighter clusters.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    ndarray of shape (n_weeks, 2)
    """
    try:
        import umap  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - import guard only
        raise ImportError(
            "umap_embed requires `umap-learn` (>=0.5). Install via "
            "`pip install -e .` or `pip install umap-learn`."
        ) from exc

    X = np.asarray(F.values if isinstance(F, pd.DataFrame) else F, dtype=float)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def confusion_buckets(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Label every row with one of {"TP", "FP", "TN", "FN"}."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    out = np.empty(y_true.shape, dtype=object)
    out[(y_true == 1) & (y_pred == 1)] = "TP"
    out[(y_true == 0) & (y_pred == 1)] = "FP"
    out[(y_true == 0) & (y_pred == 0)] = "TN"
    out[(y_true == 1) & (y_pred == 0)] = "FN"
    return out


def plot_embedding(
    emb: np.ndarray,
    color_by: np.ndarray,
    *,
    title: str = "",
    ax=None,
    cmap: str = "viridis",
    s: int = 18,
    legend: bool = True,
):
    """Scatter a 2-D embedding coloured by a continuous or categorical array.

    Continuous colours (numeric ``color_by``) get a colorbar; categorical
    colours (e.g. TP/FP/TN/FN) get a discrete legend. Returns the Matplotlib
    Axes used.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.0))
    emb = np.asarray(emb)
    color_by = np.asarray(color_by)

    if np.issubdtype(color_by.dtype, np.number):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=color_by, cmap=cmap, s=s,
                        alpha=0.85, edgecolors="none")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("")
    else:
        # Categorical: stable colour mapping for the standard confusion-matrix
        # buckets, fallback to tab10 for any other category set.
        palette = {
            "TP": "#1b7837", "FP": "#762a83",
            "TN": "#bbbbbb", "FN": "#e08214",
        }
        categories = list(pd.unique(color_by))
        for cat in categories:
            mask = color_by == cat
            col = palette.get(str(cat))
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                label=str(cat), s=s, alpha=0.85, edgecolors="none",
                color=col,
            )
        if legend:
            ax.legend(loc="best", frameon=False, fontsize=9)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
