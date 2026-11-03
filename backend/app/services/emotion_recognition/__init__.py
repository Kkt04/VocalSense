"""Speech emotion recognition services package."""

from .emotion_service import EmotionRecognitionService, get_emotion_service
from .preprocessor import load_and_preprocess_audio, AudioPreprocessingError
from .feature_extractor import extract_acoustic_features, extract_features_vector, FEATURE_NAMES

__all__ = [
    "EmotionRecognitionService",
    "get_emotion_service",
    "load_and_preprocess_audio",
    "AudioPreprocessingError",
    "extract_acoustic_features",
    "extract_features_vector",
    "FEATURE_NAMES",
]
