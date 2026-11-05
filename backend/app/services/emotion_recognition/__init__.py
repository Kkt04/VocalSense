"""Speech emotion recognition services package."""

from .emotion_service import EmotionRecognitionService, get_emotion_service
from .preprocessor import load_and_preprocess_audio, AudioPreprocessingError
from .feature_extractor import extract_acoustic_features, extract_features_vector, FEATURE_NAMES
from .wav2vec2_model import Wav2Vec2Pipeline
from .hubert_model import HuBERTPipeline
from .embedding_extractor import DatasetEmbeddingExtractor
from .deep_emotion_classifier import DeepEmotionClassifier
from .model_comparison import evaluate_speaker_independent_cv, run_full_benchmark_comparison

__all__ = [
    "EmotionRecognitionService",
    "get_emotion_service",
    "load_and_preprocess_audio",
    "AudioPreprocessingError",
    "extract_acoustic_features",
    "extract_features_vector",
    "FEATURE_NAMES",
    "Wav2Vec2Pipeline",
    "HuBERTPipeline",
    "DatasetEmbeddingExtractor",
    "DeepEmotionClassifier",
    "evaluate_speaker_independent_cv",
    "run_full_benchmark_comparison",
]
