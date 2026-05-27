"""Anomaly detectors. Higher score = more anomalous."""
from sentinel_alpha.detectors.base import AnomalyDetector
from sentinel_alpha.detectors.mvg import LedoitWolfMVG
from sentinel_alpha.detectors.gmm import GMMDetector
from sentinel_alpha.detectors.iforest import IForestDetector
from sentinel_alpha.detectors.kpca import KPCADetector
from sentinel_alpha.detectors.copod import COPODDetector
from sentinel_alpha.detectors.autoencoder import AEDetector
from sentinel_alpha.detectors.lof import LOFDetector


DETECTOR_REGISTRY: dict[str, type[AnomalyDetector]] = {
    "mvg":    LedoitWolfMVG,
    "gmm":    GMMDetector,
    "iforest": IForestDetector,
    "kpca":   KPCADetector,
    "copod":  COPODDetector,
    "ae":     AEDetector,
    "lof":    LOFDetector,
}


def build_default_detectors() -> dict[str, AnomalyDetector]:
    """Construct the seven default detectors with package-wide defaults."""
    return {name: cls() for name, cls in DETECTOR_REGISTRY.items()}


__all__ = [
    "AnomalyDetector", "LedoitWolfMVG", "GMMDetector",
    "IForestDetector", "KPCADetector", "COPODDetector", "AEDetector",
    "LOFDetector",
    "DETECTOR_REGISTRY", "build_default_detectors",
]
