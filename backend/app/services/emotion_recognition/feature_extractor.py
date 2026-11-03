"""
Acoustic feature extractor for runtime emotion inference.
Ensures exact feature ordering and extraction logic matches model training.
"""

from backend.ml.feature_extraction import (
    extract_acoustic_features,
    extract_features_vector,
    FEATURE_NAMES,
)

__all__ = ["extract_acoustic_features", "extract_features_vector", "FEATURE_NAMES"]
