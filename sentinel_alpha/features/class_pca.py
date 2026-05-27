"""Per-asset-class PCA feature engineering.

Direct implementation of hint #3 from `EarlyWarningSystemPoliMI.ipynb`:

    > Feature engineering via statistical aggregates or dimensionality reduction
    > (e.g. PCA per cluster of correlated assets).

We fit one PCA per asset class (Equity Index, Bond Yield, Bond Index, ...) on
the training portion of a fold and append the top-K principal components of
each class as new features. The class membership comes from the dataset
metadata, so the grouping is finance-driven rather than data-driven.

This module is leakage-aware: `fit` is called only on training-fold data and
`transform` applies the same fitted PCAs to validation / hold-out data. The
StackPipeline plugs this in optionally via the `class_pca_components` knob.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def map_columns_to_classes(
    feature_columns: list[str], ticker_to_class: dict[str, str],
) -> dict[str, list[str]]:
    """Group feature columns by asset class via reverse mapping.

    A feature column like 'MXUS_logret' or 'VIX_lvl_lag1' is matched to the
    longest ticker prefix present in `ticker_to_class`. Unmatched columns are
    ignored.
    """
    classes: dict[str, list[str]] = {}
    # Sort tickers by length (descending) so 'USGG30YR' wins over 'USGG3M' on prefix match.
    tickers_sorted = sorted(ticker_to_class.keys(), key=len, reverse=True)
    for col in feature_columns:
        matched = None
        for tkr in tickers_sorted:
            if col.startswith(tkr):
                matched = tkr; break
        if matched is None:
            continue
        cls = ticker_to_class[matched]
        classes.setdefault(cls, []).append(col)
    return classes


@dataclass
class PerClassPCA:
    """Fit one PCA per asset class, append the components as new features.

    Usage:
        pcca = PerClassPCA(class_to_cols, n_components=2)
        pcca.fit(F_train)
        F_train_aug = pcca.transform(F_train)
        F_val_aug   = pcca.transform(F_val)
    """
    class_to_cols: dict[str, list[str]]
    n_components: int = 2
    min_cols: int = 3  # need at least this many cols in a class to fit a PCA
    scalers_: dict[str, StandardScaler] = field(default_factory=dict, init=False)
    pcas_:    dict[str, PCA] = field(default_factory=dict, init=False)
    new_names_: list[str] = field(default_factory=list, init=False)

    def fit(self, F: pd.DataFrame) -> "PerClassPCA":
        self.scalers_.clear(); self.pcas_.clear(); self.new_names_.clear()
        for cls, cols in self.class_to_cols.items():
            cols_present = [c for c in cols if c in F.columns]
            if len(cols_present) < self.min_cols:
                continue
            sc = StandardScaler().fit(F[cols_present].values)
            Xs = sc.transform(F[cols_present].values)
            k = min(self.n_components, len(cols_present))
            pca = PCA(n_components=k, random_state=0).fit(Xs)
            self.scalers_[cls] = sc; self.pcas_[cls] = pca
            self.new_names_ += [f"ENG_pc_{_safe_class_label(cls)}_{i+1}" for i in range(k)]
        return self

    def transform(self, F: pd.DataFrame) -> pd.DataFrame:
        out = F.copy()
        new_parts: list[pd.DataFrame] = []
        for cls, sc in self.scalers_.items():
            cols_present = [c for c in self.class_to_cols[cls] if c in F.columns]
            Xs = sc.transform(F[cols_present].values)
            comps = self.pcas_[cls].transform(Xs)
            label = _safe_class_label(cls)
            df = pd.DataFrame(
                comps, index=F.index,
                columns=[f"ENG_pc_{label}_{i+1}" for i in range(comps.shape[1])],
            )
            new_parts.append(df)
        if new_parts:
            out = pd.concat([out] + new_parts, axis=1)
        return out

    @property
    def component_names(self) -> list[str]:
        return list(self.new_names_)


def _safe_class_label(cls: str) -> str:
    return (
        cls.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )
