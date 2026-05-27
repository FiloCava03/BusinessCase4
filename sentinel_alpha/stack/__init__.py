"""Stack: detectors -> rank features -> logreg -> isotonic calibration."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sentinel_alpha.config import SEED
from sentinel_alpha.detectors import build_default_detectors, AnomalyDetector
from sentinel_alpha.stack.rank import EmpiricalQuantile
from sentinel_alpha.stack.stacker import LogRegStacker
from sentinel_alpha.stack.calibrate import IsotonicCalibrator


def _stratified_holdout_indices(y: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (in_idx, out_idx) with `frac` of each class assigned to `out_idx`."""
    rng = np.random.default_rng(seed)
    out_parts: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_out = max(1, int(round(frac * cls_idx.size)))
        chosen = rng.choice(cls_idx, size=min(n_out, cls_idx.size - 1) if cls_idx.size > 1 else 1,
                            replace=False)
        out_parts.append(chosen)
    out_idx = np.sort(np.concatenate(out_parts)) if out_parts else np.array([], dtype=int)
    in_mask = np.ones(y.size, dtype=bool); in_mask[out_idx] = False
    return np.where(in_mask)[0], out_idx


@dataclass
class StackPipeline:
    """End-to-end fit-on-train / apply-on-test pipeline for one fold.

    Pipeline (all fit on training-fold data only):
        1. StandardScaler.
        2. Carve a stratified calibration sub-fold (seeded) from training so
           positives appear in calibration even in early folds.
        3. Detectors fit on the non-calibration sub-fold (semi-supervised via y).
        4. Detector scores -> per-detector training-fold empirical quantile.
        5. LogReg stacker fits on training-quantile features.
        6. Isotonic calibrator fits on the cal-sub-fold (p_raw, y_cal).

    The pipeline emits both `p_raw` (uncalibrated logreg probability, suitable
    for ranking metrics) and `p_cal` (calibrated probability, suitable for
    threshold-based decisions in the state machine).
    """
    cal_frac: float = 0.20
    detector_factory: Callable[[], dict[str, AnomalyDetector]] = field(default=build_default_detectors)
    detector_names: list[str] = field(default_factory=lambda: ["mvg", "gmm", "iforest", "kpca", "copod", "ae", "lof"])
    random_state: int = SEED

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "StackPipeline":
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train).ravel()

        self.scaler_ = StandardScaler().fit(X_train)
        Xs = self.scaler_.transform(X_train)

        det_idx, cal_idx = _stratified_holdout_indices(
            y_train, self.cal_frac, seed=self.random_state
        )
        Xs_det, y_det = Xs[det_idx], y_train[det_idx]
        Xs_cal, y_cal = Xs[cal_idx], y_train[cal_idx]

        self.detectors_: dict[str, AnomalyDetector] = self.detector_factory()
        self.detectors_ = {k: self.detectors_[k] for k in self.detector_names}
        for det in self.detectors_.values():
            det.fit(Xs_det, y_det)

        train_scores: dict[str, np.ndarray] = {}
        cal_scores: dict[str, np.ndarray] = {}
        for name, det in self.detectors_.items():
            train_scores[name] = det.score_samples(Xs_det)
            cal_scores[name] = det.score_samples(Xs_cal)

        self.quantilers_: dict[str, EmpiricalQuantile] = {}
        Q_train = np.zeros((Xs_det.shape[0], len(self.detector_names)), dtype=float)
        Q_cal = np.zeros((Xs_cal.shape[0], len(self.detector_names)), dtype=float)
        for j, name in enumerate(self.detector_names):
            q = EmpiricalQuantile().fit(train_scores[name])
            self.quantilers_[name] = q
            Q_train[:, j] = q.transform(train_scores[name])
            Q_cal[:, j] = q.transform(cal_scores[name])

        self.stacker_ = LogRegStacker().fit(Q_train, y_det)
        p_cal_raw = self.stacker_.predict_proba(Q_cal)
        # Guard: if calibration sub-fold has only one class, skip isotonic (identity).
        if len(np.unique(y_cal)) == 2:
            self.calibrator_ = IsotonicCalibrator().fit(p_cal_raw, y_cal)
        else:
            self.calibrator_ = None

        self.n_train_det_ = Xs_det.shape[0]
        self.n_train_cal_ = Xs_cal.shape[0]
        self.cal_pos_ = int(y_cal.sum())
        return self

    def _detector_quantiles(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        Q = np.zeros((Xs.shape[0], len(self.detector_names)), dtype=float)
        for j, name in enumerate(self.detector_names):
            s = self.detectors_[name].score_samples(Xs)
            Q[:, j] = self.quantilers_[name].transform(s)
        return Q

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        Q = self._detector_quantiles(X)
        return self.stacker_.predict_proba(Q)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_raw = self.predict_proba_raw(X)
        if self.calibrator_ is None:
            return p_raw
        return self.calibrator_.transform(p_raw)

    def predict_detector_quantiles_df(self, X: np.ndarray, index: pd.Index) -> pd.DataFrame:
        Q = self._detector_quantiles(X)
        return pd.DataFrame(Q, index=index, columns=list(self.detector_names))


__all__ = ["StackPipeline"]
