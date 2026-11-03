import os
import json
import time
import logging
from typing import Optional, Union, Dict, Any, List
import numpy as np
import joblib

from backend.app.schemas.emotion import EmotionPredictionResponse
from backend.app.services.emotion_recognition.preprocessor import load_and_preprocess_audio
from backend.app.services.emotion_recognition.feature_extractor import (
    extract_acoustic_features,
    extract_features_vector,
    FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "backend/saved_models/emotion_classifier.joblib"
DEFAULT_METADATA_PATH = "backend/saved_models/model_metadata.json"


class EmotionRecognitionService:
    """
    Speech Emotion Recognition service.
    Loads trained scikit-learn pipeline, extracts standardized acoustic features,
    and returns calibrated prediction probabilities with reliability thresholds.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.classes: List[str] = []
        self.confidence_threshold: float = 0.32
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load serialized pipeline and metadata if available."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
                self.classes = self.metadata.get("classes", [])
                self.confidence_threshold = self.metadata.get("confidence_threshold", 0.32)
            except Exception as e:
                logger.warning(f"Could not load emotion model metadata: {e}")

        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info(f"Loaded emotion recognition model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load emotion model from {self.model_path}: {e}")
                self.pipeline = None

    def is_ready(self) -> bool:
        """Return True if model is loaded and ready for inference."""
        return self.pipeline is not None

    def predict_emotion(
        self,
        audio_input: Union[bytes, str, np.ndarray],
    ) -> EmotionPredictionResponse:
        """
        Run acoustic feature extraction and emotion inference.

        Args:
            audio_input: Audio bytes, file path, or numpy array.

        Returns:
            EmotionPredictionResponse with predicted pattern, confidence, and vocal traits.
        """
        start_time = time.perf_counter()

        # Step 1: Preprocess audio (16kHz mono, silence trim, peak normalization)
        y, sr = load_and_preprocess_audio(audio_input)

        # Step 2: Extract acoustic traits
        feat_dict = extract_acoustic_features(y, sr)
        feature_vector = np.array([feat_dict[name] for name in FEATURE_NAMES], dtype=np.float32).reshape(1, -1)

        # Build readable physical vocal traits for insights panel
        vocal_traits = {
            "pitch_mean_hz": round(feat_dict.get("pitch_f0_mean", 0.0), 1),
            "pitch_std_hz": round(feat_dict.get("pitch_f0_std", 0.0), 1),
            "rms_energy": round(feat_dict.get("rms_energy_mean", 0.0), 4),
            "spectral_centroid_hz": round(feat_dict.get("spectral_centroid_mean", 0.0), 1),
            "pause_fraction": round(feat_dict.get("pause_fraction", 0.0), 3),
            "duration_sec": round(float(len(y)) / sr, 2),
        }

        # Step 3: Run inference if model is loaded
        if self.is_ready():
            probabilities = self.pipeline.predict_proba(feature_vector)[0]
            classes = self.pipeline.classes_ if hasattr(self.pipeline, "classes_") else self.classes

            prob_dict = {
                str(c): round(float(prob), 4)
                for c, prob in zip(classes, probabilities)
            }

            top_idx = int(np.argmax(probabilities))
            top_label = str(classes[top_idx])
            top_prob = round(float(probabilities[top_idx]), 4)

            # Reliability / Abstention logic:
            # If highest confidence is below threshold, mark as unreliable
            is_reliable = top_prob >= self.confidence_threshold
            reported_emotion = top_label if is_reliable else "uncertain"
            model_name = self.metadata.get("model_name", "Emotion Classifier")

        else:
            # Fallback when model is not yet trained
            prob_dict = {}
            top_prob = 0.0
            reported_emotion = "uncertain"
            is_reliable = False
            model_name = "Baseline Heuristic (Model Untrained)"

        latency = round(time.perf_counter() - start_time, 3)

        return EmotionPredictionResponse(
            success=True,
            emotion_label=reported_emotion,
            confidence=top_prob,
            probabilities=prob_dict,
            vocal_traits=vocal_traits,
            is_reliable=is_reliable,
            model_used=model_name,
            processing_time_sec=latency,
        )


_emotion_service_instance: Optional[EmotionRecognitionService] = None


def get_emotion_service() -> EmotionRecognitionService:
    """Retrieve or initialize the global EmotionRecognitionService singleton."""
    global _emotion_service_instance
    if _emotion_service_instance is None:
        _emotion_service_instance = EmotionRecognitionService()
    return _emotion_service_instance
