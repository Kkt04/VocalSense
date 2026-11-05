"""
Unified Speech Emotion Recognition (SER) Runtime Service.
Supports:
1. Deep Learning: Wav2Vec2 Contextual Speech Representations (768-dim)
2. Deep Learning: HuBERT Masked Prediction Representations (768-dim)
3. Traditional ML: Phase 3 SVM Baseline (43 Acoustic Features)

Features:
- Singleton lifecycle with lazy loading
- Configurable via settings.EMOTION_MODEL_TYPE
- Real physical acoustic trait extraction (pitch, RMS, spectral, pauses) for companion telemetry
- Bounded confidence threshold with abstention guardrails ('uncertain' label)
- Graceful fallback if a deep learning checkpoint or artifact is missing
"""

import os
import json
import time
import logging
from typing import Optional, Union, Dict, Any, List
import numpy as np
import joblib

from backend.app.config import settings
from backend.app.schemas.emotion import EmotionPredictionResponse
from backend.app.services.emotion_recognition.preprocessor import load_and_preprocess_audio
from backend.app.services.emotion_recognition.feature_extractor import (
    extract_acoustic_features,
    extract_features_vector,
    FEATURE_NAMES,
)
from backend.app.services.emotion_recognition.wav2vec2_model import Wav2Vec2Pipeline
from backend.app.services.emotion_recognition.hubert_model import HuBERTPipeline
from backend.app.services.emotion_recognition.deep_emotion_classifier import DeepEmotionClassifier

logger = logging.getLogger(__name__)

DEFAULT_SVM_MODEL_PATH = "backend/saved_models/emotion_classifier.joblib"
DEFAULT_SVM_METADATA_PATH = "backend/saved_models/model_metadata.json"

PHASE4_MODELS_DIR = "backend/saved_models/phase4"


class EmotionRecognitionService:
    """
    Multimodal SER Runtime Service.
    Seamlessly executes deep-learning (Wav2Vec2/HuBERT) or traditional ML inference
    with physical acoustic feature extraction and calibrated abstention logic.
    """

    def __init__(
        self,
        model_type: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self.model_type = (model_type or getattr(settings, "EMOTION_MODEL_TYPE", "wav2vec2")).lower()
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(settings, "EMOTION_CONFIDENCE_THRESHOLD", 0.30)
        )

        # Traditional SVM artifact state
        self.svm_pipeline = None
        self.svm_metadata: Dict[str, Any] = {}

        # Deep learning pipelines & classifiers
        self.wav2vec2_pipeline: Optional[Wav2Vec2Pipeline] = None
        self.wav2vec2_classifier: Optional[DeepEmotionClassifier] = None

        self.hubert_pipeline: Optional[HuBERTPipeline] = None
        self.hubert_classifier: Optional[DeepEmotionClassifier] = None

        self.classes: List[str] = []
        self._load_active_model()

    def _load_active_model(self) -> None:
        """Loads model artifacts according to the active model_type."""
        # Always try to load traditional SVM as a reliable fallback
        self._load_traditional_svm()

        if self.model_type == "wav2vec2":
            self._load_wav2vec2()
        elif self.model_type == "hubert":
            self._load_hubert()
        elif self.model_type in ["traditional_svm", "svm"]:
            self.classes = self.svm_metadata.get("classes", self.classes)

    def _load_traditional_svm(self) -> None:
        if os.path.exists(DEFAULT_SVM_METADATA_PATH):
            try:
                with open(DEFAULT_SVM_METADATA_PATH, "r") as f:
                    self.svm_metadata = json.load(f)
                self.classes = self.svm_metadata.get("classes", [])
            except Exception as e:
                logger.warning("Could not read SVM metadata: %s", e)

        if os.path.exists(DEFAULT_SVM_MODEL_PATH):
            try:
                self.svm_pipeline = joblib.load(DEFAULT_SVM_MODEL_PATH)
                logger.info("Loaded traditional SVM baseline from %s", DEFAULT_SVM_MODEL_PATH)
            except Exception as e:
                logger.warning("Failed to load traditional SVM model: %s", e)

    def _load_wav2vec2(self) -> None:
        model_path = os.path.join(PHASE4_MODELS_DIR, "wav2vec2_emotion_classifier.joblib")
        meta_path = os.path.join(PHASE4_MODELS_DIR, "wav2vec2_metadata.json")

        try:
            checkpoint = getattr(settings, "WAV2VEC2_CHECKPOINT", "facebook/wav2vec2-base")
            device = getattr(settings, "EMOTION_DEVICE", "auto")
            self.wav2vec2_pipeline = Wav2Vec2Pipeline(checkpoint=checkpoint, device=device)

            if os.path.exists(model_path) and os.path.exists(meta_path):
                clf = DeepEmotionClassifier(model_type="wav2vec2")
                clf.load(model_path, meta_path)
                self.wav2vec2_classifier = clf
                self.classes = clf.classes
                logger.info("Loaded Wav2Vec2 downstream classifier from %s", model_path)
            else:
                logger.warning(
                    "Wav2Vec2 downstream classifier weights not found at %s. (Will use fallback or fit on demand)",
                    model_path,
                )
        except Exception as exc:
            logger.error("Error initializing Wav2Vec2: %s", exc)

    def _load_hubert(self) -> None:
        model_path = os.path.join(PHASE4_MODELS_DIR, "hubert_emotion_classifier.joblib")
        meta_path = os.path.join(PHASE4_MODELS_DIR, "hubert_metadata.json")

        try:
            checkpoint = getattr(settings, "HUBERT_CHECKPOINT", "facebook/hubert-base-ls960")
            device = getattr(settings, "EMOTION_DEVICE", "auto")
            self.hubert_pipeline = HuBERTPipeline(checkpoint=checkpoint, device=device)

            if os.path.exists(model_path) and os.path.exists(meta_path):
                clf = DeepEmotionClassifier(model_type="hubert")
                clf.load(model_path, meta_path)
                self.hubert_classifier = clf
                self.classes = clf.classes
                logger.info("Loaded HuBERT downstream classifier from %s", model_path)
            else:
                logger.warning(
                    "HuBERT downstream classifier weights not found at %s.",
                    model_path,
                )
        except Exception as exc:
            logger.error("Error initializing HuBERT: %s", exc)

    def is_ready(self) -> bool:
        """Determines if the active model or any valid fallback model is ready for inference."""
        if self.model_type == "wav2vec2":
            if self.wav2vec2_pipeline is not None and self.wav2vec2_classifier is not None:
                return True
        elif self.model_type == "hubert":
            if self.hubert_pipeline is not None and self.hubert_classifier is not None:
                return True
        return self.svm_pipeline is not None

    def get_active_model_name(self) -> str:
        if self.model_type == "wav2vec2" and self.wav2vec2_classifier is not None:
            return "Wav2Vec2 Deep Speech Classifier (facebook/wav2vec2-base)"
        if self.model_type == "hubert" and self.hubert_classifier is not None:
            return "HuBERT Deep Speech Classifier (facebook/hubert-base-ls960)"
        if self.svm_pipeline is not None:
            return "Support Vector Machine (Phase 3 Baseline)"
        return "Baseline Heuristic (Uncalibrated)"

    def predict_emotion(
        self,
        audio_input: Union[bytes, str, np.ndarray],
        override_model_type: Optional[str] = None,
    ) -> EmotionPredictionResponse:
        """
        Runs speech representation extraction and emotion inference.
        Returns prediction probabilities and physical vocal traits.
        """
        start_time = time.perf_counter()
        target_model = (override_model_type or self.model_type).lower()

        # Step 1: Preprocess audio (16kHz mono, silence trim, peak normalization)
        y, sr = load_and_preprocess_audio(audio_input)

        # Step 2: Extract real acoustic traits for companion insights panel
        feat_dict = extract_acoustic_features(y, sr)
        vocal_traits = {
            "pitch_mean_hz": round(float(feat_dict.get("pitch_f0_mean", 0.0)), 1),
            "pitch_std_hz": round(float(feat_dict.get("pitch_f0_std", 0.0)), 1),
            "rms_energy": round(float(feat_dict.get("rms_energy_mean", 0.0)), 4),
            "spectral_centroid_hz": round(float(feat_dict.get("spectral_centroid_mean", 0.0)), 1),
            "pause_fraction": round(float(feat_dict.get("pause_fraction", 0.0)), 3),
            "duration_sec": round(float(len(y)) / sr, 2),
        }

        prob_dict: Dict[str, float] = {}
        top_prob: float = 0.0
        reported_emotion: str = "uncertain"
        is_reliable: bool = False
        model_name: str = self.get_active_model_name()

        # Step 3: Run Inference according to model choice
        try:
            # Wav2Vec2 Inference
            if target_model == "wav2vec2" and self.wav2vec2_pipeline and self.wav2vec2_classifier:
                self.wav2vec2_classifier.confidence_threshold = self.confidence_threshold
                emb = self.wav2vec2_pipeline.extract_embedding(y, sample_rate=sr)
                pred_out = self.wav2vec2_classifier.predict_single(emb)
                prob_dict = pred_out["probabilities"]
                top_prob = pred_out["confidence"]
                is_reliable = pred_out["is_reliable"]
                reported_emotion = pred_out["emotion_label"]
                model_name = "Wav2Vec2 Deep Speech Classifier"

            # HuBERT Inference
            elif target_model == "hubert" and self.hubert_pipeline and self.hubert_classifier:
                self.hubert_classifier.confidence_threshold = self.confidence_threshold
                emb = self.hubert_pipeline.extract_embedding(y, sample_rate=sr)
                pred_out = self.hubert_classifier.predict_single(emb)
                prob_dict = pred_out["probabilities"]
                top_prob = pred_out["confidence"]
                is_reliable = pred_out["is_reliable"]
                reported_emotion = pred_out["emotion_label"]
                model_name = "HuBERT Deep Speech Classifier"

            # Traditional SVM Baseline Inference
            elif self.svm_pipeline is not None:
                feature_vector = np.array([feat_dict[name] for name in FEATURE_NAMES], dtype=np.float32).reshape(1, -1)
                probabilities = self.svm_pipeline.predict_proba(feature_vector)[0]
                classes = self.svm_pipeline.classes_ if hasattr(self.svm_pipeline, "classes_") else self.classes

                prob_dict = {
                    str(c): round(float(prob), 4)
                    for c, prob in zip(classes, probabilities)
                }
                top_idx = int(np.argmax(probabilities))
                top_label = str(classes[top_idx])
                top_prob = round(float(probabilities[top_idx]), 4)

                is_reliable = top_prob >= self.confidence_threshold
                reported_emotion = top_label if is_reliable else "uncertain"
                model_name = "Support Vector Machine (Acoustic Features)"

            else:
                reported_emotion = "uncertain"
                is_reliable = False
                model_name = "Baseline Heuristic (Model Untrained)"

        except Exception as exc:
            logger.warning("Error during %s emotion inference: %s. Reverting to baseline heuristic.", target_model, exc)
            reported_emotion = "uncertain"
            is_reliable = False
            model_name = f"Inference Error Fallback ({target_model})"

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
