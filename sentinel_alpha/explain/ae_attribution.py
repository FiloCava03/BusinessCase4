"""Gradient-times-input attribution on the autoencoder reconstruction loss."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from sentinel_alpha.detectors.autoencoder import AEDetector


def grad_times_input(ae: AEDetector, X: np.ndarray) -> np.ndarray:
    """Per-feature attribution to the reconstruction MSE.

    For each row x, returns x_i * d(MSE)/d(x_i). Same shape as X.
    Higher (positive) values mean that feature contributes more to the
    reconstruction error.
    """
    ae.net_.eval()
    xt = torch.tensor(np.asarray(X, dtype=np.float32), requires_grad=True)
    rec = ae.net_(xt)
    mse_per_row = ((xt - rec) ** 2).mean(dim=1)
    grads = torch.autograd.grad(
        outputs=mse_per_row.sum(), inputs=xt, create_graph=False
    )[0].detach().numpy()
    return xt.detach().numpy() * grads


def attribution_by_crisis(
    attribution: np.ndarray, feature_names: list[str], index: pd.DatetimeIndex,
    crises: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Mean attribution per feature inside each crisis window."""
    df = pd.DataFrame(attribution, columns=feature_names, index=index)
    rows = []
    for name, (s, e) in crises.items():
        m = (df.index >= pd.Timestamp(s)) & (df.index <= pd.Timestamp(e))
        if m.any():
            rows.append({"crisis": name, **df.loc[m].mean().to_dict()})
    return pd.DataFrame(rows).set_index("crisis") if rows else pd.DataFrame()
