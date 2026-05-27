"""Asymmetry gate: only allow risk-off when risk appetite is below tau."""
from __future__ import annotations
import numpy as np


def apply_gate(p: np.ndarray, risk_appetite: np.ndarray, tau: float) -> np.ndarray:
    """Element-wise: signal = p where RA < tau, else 0.0."""
    p = np.asarray(p, dtype=float)
    ra = np.asarray(risk_appetite, dtype=float)
    assert p.shape == ra.shape, "p and risk_appetite must have the same shape"
    return np.where(ra < tau, p, 0.0)
