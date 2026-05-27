"""Cluster flagged anomalies by their SHAP attribution profile.

Direct implementation of hint #3 from `EarlyWarningSystemPoliMI.ipynb`:

    > Anomaly interpretation based on macro/market context (e.g. clustering
    > anomalies by cause).

For each week flagged as risk-off we have a SHAP attribution vector over the
detector axes (mvg, gmm, iforest, kpca, copod, ae_ensemble, lof). KMeans on
those vectors groups together weeks that *fire the alarm for the same
reasons*. A cluster dominated by COPOD + AE-ensemble reads as "tail-dependent
multivariate stress" whereas one dominated by MVG + GMM reads as "elliptic
distribution shift".
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sentinel_alpha.config import SEED


@dataclass(frozen=True)
class AnomalyClusterReport:
    labels: pd.Series                       # cluster id per flagged week
    centroids: pd.DataFrame                 # cluster centroid in SHAP space (one row per cluster)
    crisis_assignment: pd.DataFrame         # for each named crisis: dominant cluster + member count
    cluster_signature: pd.DataFrame         # per cluster: top-2 contributing detectors


def cluster_anomalies(
    shap_attribution: pd.DataFrame,
    flagged_index: pd.DatetimeIndex,
    crises: dict[str, tuple[str, str]],
    n_clusters: int = 3,
) -> AnomalyClusterReport:
    """KMeans on the SHAP rows of `flagged_index`, return interpretable labels.

    Parameters
    ----------
    shap_attribution : DataFrame [n_weeks_holdout, n_detectors]
        SHAP values of each detector at each hold-out week (output of
        `stacker_shap_values`).
    flagged_index : DatetimeIndex
        Subset of `shap_attribution.index` for the weeks the strategy flagged
        as risk-off. We cluster only those.
    crises : dict
        Named crisis windows -- used to label clusters by their crisis content.
    n_clusters : int
        Default 3 (broad / fast / shallow stress regimes).
    """
    if len(flagged_index) < n_clusters:
        empty = pd.DataFrame()
        return AnomalyClusterReport(
            labels=pd.Series([], dtype=int, name="cluster"),
            centroids=empty, crisis_assignment=empty, cluster_signature=empty,
        )
    sub = shap_attribution.loc[flagged_index].copy()
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED).fit(sub.values)
    labels = pd.Series(km.labels_, index=flagged_index, name="cluster")

    centroids = pd.DataFrame(km.cluster_centers_, columns=sub.columns,
                             index=[f"C{k}" for k in range(n_clusters)])

    # Top-2 contributing detectors per cluster (by absolute centroid weight).
    sig_rows = []
    for k in range(n_clusters):
        top = centroids.iloc[k].abs().sort_values(ascending=False).head(2)
        sig_rows.append({
            "cluster": f"C{k}",
            "size": int((labels == k).sum()),
            "top_1": top.index[0],
            "top_1_weight": float(centroids.iloc[k][top.index[0]]),
            "top_2": top.index[1] if len(top) > 1 else "",
            "top_2_weight": float(centroids.iloc[k][top.index[1]]) if len(top) > 1 else 0.0,
        })
    cluster_signature = pd.DataFrame(sig_rows)

    # Crisis -> dominant cluster mapping.
    rows = []
    for name, (start, end) in crises.items():
        mask = (labels.index >= pd.Timestamp(start)) & (labels.index <= pd.Timestamp(end))
        if not mask.any():
            continue
        counts = labels[mask].value_counts().to_dict()
        dom = max(counts, key=counts.get)
        rows.append({
            "crisis": name,
            "n_flagged_in_crisis": int(mask.sum()),
            "dominant_cluster": f"C{dom}",
            "dominant_share": counts[dom] / mask.sum(),
            "cluster_counts": counts,
        })
    crisis_assignment = pd.DataFrame(rows)

    return AnomalyClusterReport(
        labels=labels,
        centroids=centroids,
        crisis_assignment=crisis_assignment,
        cluster_signature=cluster_signature,
    )
