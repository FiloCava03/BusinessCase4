"""Anomaly detectors. Higher score = more anomalous."""
from sentinel_alpha.detectors.base import AnomalyDetector
from sentinel_alpha.detectors.mvg import LedoitWolfMVG
from sentinel_alpha.detectors.gmm import GMMDetector
from sentinel_alpha.detectors.iforest import IForestDetector
from sentinel_alpha.detectors.kpca import KPCADetector
from sentinel_alpha.detectors.copod import COPODDetector
from sentinel_alpha.detectors.autoencoder import AEDetector
from sentinel_alpha.detectors.ae_ensemble import AEEnsembleDetector
from sentinel_alpha.detectors.lof import LOFDetector
from sentinel_alpha.detectors.supervised import RFDetector, LogRegDetector


DETECTOR_REGISTRY: dict[str, type[AnomalyDetector]] = {
    "mvg":         LedoitWolfMVG,
    "gmm":         GMMDetector,
    "iforest":     IForestDetector,
    "kpca":        KPCADetector,
    "copod":       COPODDetector,
    "ae":          AEDetector,
    "ae_ensemble": AEEnsembleDetector,
    "lof":         LOFDetector,
    "rf":          RFDetector,        # supervised
    "logreg":      LogRegDetector,    # supervised
}


def build_default_detectors() -> dict[str, AnomalyDetector]:
    """Construct the nine default detectors with package-wide defaults.

    Seven unsupervised / semi-supervised (mvg, gmm, iforest, kpca, copod, ae, lof)
    plus two supervised (rf, logreg). All have been shown to add marginal
    information to the stack on the CV ablation.
    """
    return {name: cls() for name, cls in DETECTOR_REGISTRY.items()}


__all__ = [
    "AnomalyDetector", "LedoitWolfMVG", "GMMDetector",
    "IForestDetector", "KPCADetector", "COPODDetector", "AEDetector",
    "AEEnsembleDetector",
    "LOFDetector", "RFDetector", "LogRegDetector",
    "DETECTOR_REGISTRY", "build_default_detectors",
]
