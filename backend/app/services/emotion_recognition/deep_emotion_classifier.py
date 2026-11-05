"""
Downstream Emotion Classifier for Deep Speech Representations.
Trains and serves classification heads on top of contextual embeddings (Wav2Vec2 and HuBERT),
providing calibrated probability distributions and speaker-independent validation.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)


class DeepEmotionClassifier:
    """
    Downstream classification model for 768-dimensional speech embeddings.
    Provides calibrated probability inference and serialization.
    """

    def __init__(
        self,
        model_type: str = "wav2vec2",  # "wav2vec2" or "hubert"
        classifier_type: str = "logistic_regression",  # "logistic_regression", "mlp", "svm"
        confidence_threshold: float = 0.30,
    ):
        self.model_type = model_type
        self.classifier_type = classifier_type
        self.confidence_threshold = confidence_threshold
        self.pipeline: Optional[Pipeline] = None
        self.classes: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def build_pipeline(self) -> Pipeline:
        """Constructs the downstream classification pipeline."""
        if self.classifier_type == "logistic_regression":
            clf = LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )
        elif self.classifier_type == "mlp":
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 64),
                max_iter=500,
                activation="relu",
                early_stopping=True,
                random_state=42,
            )
        elif self.classifier_type == "svm":
            clf = SVC(
                kernel="rbf",
                C=2.0,
                probability=True,
                random_state=42,
            )
        else:
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepEmotionClassifier":
        """Fits the pipeline on extracted deep embeddings."""
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X, y)
        self.classes = list(self.pipeline.classes_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Classifier has not been fitted or loaded.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Classifier has not been fitted or loaded.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.pipeline.predict(X)

    def predict_single(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Infers emotion pattern from a single embedding vector with reliability abstention.
        """
        probs = self.predict_proba(embedding)[0]
        top_idx = int(np.argmax(probs))
        top_class = str(self.classes[top_idx])
        top_prob = round(float(probs[top_idx]), 4)

        is_reliable = top_prob >= self.confidence_threshold
        reported_emotion = top_class if is_reliable else "uncertain"

        prob_dict = {
            str(c): round(float(p), 4)
            for c, p in zip(self.classes, probs)
        }

        return {
            "emotion_label": reported_emotion,
            "confidence": top_prob,
            "probabilities": prob_dict,
            "is_reliable": is_reliable,
            "top_raw_class": top_class,
        }

    def save(self, model_path: str, metadata_path: str, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Serializes the pipeline and metadata to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.pipeline, model_path)

        meta = {
            "model_type": self.model_type,
            "classifier_type": self.classifier_type,
            "classes": self.classes,
            "confidence_threshold": self.confidence_threshold,
            "saved_at": time.time(),
        }
        if extra_metadata:
            meta.update(extra_metadata)

        self.metadata = meta
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved classifier to %s and metadata to %s", model_path, metadata_path)

    def load(self, model_path: str, metadata_path: str) -> "DeepEmotionClassifier":
        """Loads serialized pipeline and metadata from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.pipeline = joblib.load(model_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            self.classes = self.metadata.get("classes", list(getattr(self.pipeline, "classes_", [])))
            self.confidence_threshold = self.metadata.get("confidence_threshold", self.confidence_threshold)
            self.model_type = self.metadata.get("model_type", self.model_type)
        else:
            self.classes = list(getattr(self.pipeline, "classes_", []))
        return self
