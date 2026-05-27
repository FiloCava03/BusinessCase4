"""Faithful replicas of the professor's baseline pipelines (EarlyWarningSystemPoliMI.ipynb).

Each function trains on the chronological pre-2019 history and predicts a binary
risk-off label on the 2019-2021 hold-out. Hyperparameters (thresholds, nu, ...) are
tuned on a held-out validation tail of the training period, **never** on the hold-out.

These are kept honest reproductions, not improved variants. The intent is a fair
head-to-head with Sentinel-alpha on the same data.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score

from sentinel_alpha.config import SEED
from sentinel_alpha.detectors.autoencoder import AEDetector


VAL_FRAC = 0.20  # last 20% of training = validation for threshold tuning (prof's recipe)


def _split_train_val(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    n_val = max(1, int(VAL_FRAC * n))
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]


def _best_threshold_f1(p: np.ndarray, y_true: np.ndarray, direction: str = "above") -> float:
    """Pick the threshold that maximises F1 on (p, y_true).

    direction = 'above': predict 1 when p >= threshold (e.g. raw anomaly score)
    direction = 'below': predict 1 when p <= threshold (e.g. likelihood/pdf)
    """
    best = (-1.0, 0.0)
    if direction == "above":
        grid = np.unique(p)
        if grid.size > 200:
            grid = np.quantile(p, np.linspace(0.01, 0.99, 200))
        for thr in grid:
            f1 = f1_score(y_true, (p >= thr).astype(int), zero_division=0)
            if f1 > best[0]:
                best = (f1, float(thr))
    else:
        grid = np.unique(p)
        if grid.size > 200:
            grid = np.quantile(p, np.linspace(0.01, 0.99, 200))
        for thr in grid:
            f1 = f1_score(y_true, (p <= thr).astype(int), zero_division=0)
            if f1 > best[0]:
                best = (f1, float(thr))
    return best[1]


@dataclass
class BaselineResult:
    name: str
    scores_holdout: np.ndarray   # continuous score on hold-out (higher = more anomalous)
    preds_holdout: np.ndarray    # binary predictions after F1-tuned threshold
    threshold: float
    direction: str               # 'above' or 'below'


def fit_predict_prof_mvg(X_train: np.ndarray, y_train: np.ndarray, X_ho: np.ndarray) -> BaselineResult:
    """Prof's MVG baseline (notebook cells 30-36).

    - StandardScaler fit on full TRAIN (no per-fold; faithful to prof).
    - MVG fit on rows of TRAIN where Y == 0 (novelty setup).
    - Threshold epsilon on the PDF p(x): predict 1 when p(x) < eps.
    - eps chosen on the validation tail by max F1.
    """
    sc = StandardScaler().fit(X_train)
    Xs = sc.transform(X_train); Xho = sc.transform(X_ho)
    Xtr, ytr, Xva, yva = _split_train_val(Xs, y_train)
    Xtr_normal = Xtr[ytr == 0]
    mu = Xtr_normal.mean(axis=0)
    sigma = np.cov(Xtr_normal, rowvar=False)
    dist = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
    p_val = dist.pdf(Xva)
    eps = _best_threshold_f1(p_val, yva, direction="below")
    p_ho = dist.pdf(Xho)
    return BaselineResult(
        name="prof_MVG",
        scores_holdout=-p_ho,  # negate so "higher = more anomalous" (uniform convention)
        preds_holdout=(p_ho < eps).astype(int),
        threshold=eps, direction="below",
    )


def fit_predict_prof_iforest(X_train: np.ndarray, y_train: np.ndarray, X_ho: np.ndarray) -> BaselineResult:
    """Prof's Isolation Forest baseline (cell 50)."""
    sc = StandardScaler().fit(X_train)
    Xs = sc.transform(X_train); Xho = sc.transform(X_ho)
    Xtr, ytr, Xva, yva = _split_train_val(Xs, y_train)
    contamination = float(np.clip(ytr.mean(), 1e-3, 0.5))
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=SEED).fit(Xtr)
    s_val = -model.decision_function(Xva)   # higher = more anomalous
    thr = _best_threshold_f1(s_val, yva, direction="above")
    s_ho = -model.decision_function(Xho)
    return BaselineResult(
        name="prof_IsolationForest",
        scores_holdout=s_ho, preds_holdout=(s_ho >= thr).astype(int),
        threshold=thr, direction="above",
    )


def fit_predict_prof_ocsvm(X_train: np.ndarray, y_train: np.ndarray, X_ho: np.ndarray) -> BaselineResult:
    """Prof's One-Class SVM baseline (cell 52)."""
    sc = StandardScaler().fit(X_train)
    Xs = sc.transform(X_train); Xho = sc.transform(X_ho)
    Xtr, ytr, Xva, yva = _split_train_val(Xs, y_train)
    nu = float(np.clip(ytr.mean(), 1e-3, 0.5))
    model = OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(Xtr[ytr == 0])
    s_val = -model.decision_function(Xva).ravel()
    thr = _best_threshold_f1(s_val, yva, direction="above")
    s_ho = -model.decision_function(Xho).ravel()
    return BaselineResult(
        name="prof_OCSVM",
        scores_holdout=s_ho, preds_holdout=(s_ho >= thr).astype(int),
        threshold=thr, direction="above",
    )


def fit_predict_prof_lof(X_train: np.ndarray, y_train: np.ndarray, X_ho: np.ndarray) -> BaselineResult:
    """Prof's Local Outlier Factor (novelty=True) baseline (cell 54)."""
    sc = StandardScaler().fit(X_train)
    Xs = sc.transform(X_train); Xho = sc.transform(X_ho)
    Xtr, ytr, Xva, yva = _split_train_val(Xs, y_train)
    contamination = float(np.clip(ytr.mean(), 1e-3, 0.5))
    model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=contamination).fit(Xtr[ytr == 0])
    s_val = -model.decision_function(Xva).ravel()
    thr = _best_threshold_f1(s_val, yva, direction="above")
    s_ho = -model.decision_function(Xho).ravel()
    return BaselineResult(
        name="prof_LOF",
        scores_holdout=s_ho, preds_holdout=(s_ho >= thr).astype(int),
        threshold=thr, direction="above",
    )


def fit_predict_prof_gmm(X_train: np.ndarray, y_train: np.ndarray, X_ho: np.ndarray) -> BaselineResult:
    """Prof's GMM-2-comp baseline (cell 56). Unsupervised: fit on the full batch."""
    sc = StandardScaler().fit(X_train)
    Xs = sc.transform(X_train); Xho = sc.transform(X_ho)
    Xtr, ytr, Xva, yva = _split_train_val(Xs, y_train)
    model = GaussianMixture(n_components=2, covariance_type="full", reg_covar=1e-4,
                            random_state=SEED, max_iter=300).fit(Xtr)
    s_val = -model.score_samples(Xva)
    thr = _best_threshold_f1(s_val, yva, direction="above")
    s_ho = -model.score_samples(Xho)
    return BaselineResult(
        name="prof_GMM2",
        scores_holdout=s_ho, preds_holdout=(s_ho >= thr).astype(int),
        threshold=thr, direction="above",
    )


def fit_predict_prof_random_forest(X_train: np.ndarray, y_train: np.ndarray, X_ho: np.ndarray) -> BaselineResult:
    """Prof's Random Forest supervised baseline (cell 45)."""
    sc = StandardScaler().fit(X_train)
    Xs = sc.transform(X_train); Xho = sc.transform(X_ho)
    Xtr, ytr, Xva, yva = _split_train_val(Xs, y_train)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=SEED, n_jobs=1).fit(Xtr, ytr)
    p_val = rf.predict_proba(Xva)[:, 1]
    thr = _best_threshold_f1(p_val, yva, direction="above")
    p_ho = rf.predict_proba(Xho)[:, 1]
    return BaselineResult(
        name="prof_RandomForest",
        scores_holdout=p_ho, preds_holdout=(p_ho >= thr).astype(int),
        threshold=thr, direction="above",
    )


def fit_predict_prof_logreg(X_train: np.ndarray, y_train: np.ndarray, X_ho: np.ndarray) -> BaselineResult:
    """Prof's Logistic Regression supervised baseline."""
    sc = StandardScaler().fit(X_train)
    Xs = sc.transform(X_train); Xho = sc.transform(X_ho)
    Xtr, ytr, Xva, yva = _split_train_val(Xs, y_train)
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED).fit(Xtr, ytr)
    p_val = lr.predict_proba(Xva)[:, 1]
    thr = _best_threshold_f1(p_val, yva, direction="above")
    p_ho = lr.predict_proba(Xho)[:, 1]
    return BaselineResult(
        name="prof_LogReg",
        scores_holdout=p_ho, preds_holdout=(p_ho >= thr).astype(int),
        threshold=thr, direction="above",
    )


def fit_predict_prof_autoencoder(X_train: np.ndarray, y_train: np.ndarray, X_ho: np.ndarray) -> BaselineResult:
    """Prof's denoising autoencoder baseline (cells 64-68)."""
    sc = StandardScaler().fit(X_train)
    Xs = sc.transform(X_train); Xho = sc.transform(X_ho)
    Xtr, ytr, Xva, yva = _split_train_val(Xs, y_train)
    ae = AEDetector(random_state=SEED).fit(Xtr, ytr)
    s_val = ae.score_samples(Xva)
    thr = _best_threshold_f1(s_val, yva, direction="above")
    s_ho = ae.score_samples(Xho)
    return BaselineResult(
        name="prof_AE",
        scores_holdout=s_ho, preds_holdout=(s_ho >= thr).astype(int),
        threshold=thr, direction="above",
    )


PROF_BASELINES: dict[str, Callable[..., BaselineResult]] = {
    "prof_MVG":             fit_predict_prof_mvg,
    "prof_IsolationForest": fit_predict_prof_iforest,
    "prof_OCSVM":           fit_predict_prof_ocsvm,
    "prof_LOF":             fit_predict_prof_lof,
    "prof_GMM2":            fit_predict_prof_gmm,
    "prof_RandomForest":    fit_predict_prof_random_forest,
    "prof_LogReg":          fit_predict_prof_logreg,
    "prof_AE":              fit_predict_prof_autoencoder,
}
