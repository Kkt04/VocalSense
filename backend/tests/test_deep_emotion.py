"""
Unit tests for Phase 4 Deep Learning Speech Emotion Recognition modules.
Tests Wav2Vec2 and HuBERT feature extractors, embedding dimension consistency,
classifier inference, abstention logic, serialization, and runtime service dispatching.
"""

import os
import pytest
import numpy as np
import torch

from backend.app.services.emotion_recognition.wav2vec2_model import Wav2Vec2Pipeline, TARGET_SAMPLE_RATE
from backend.app.services.emotion_recognition.hubert_model import HuBERTPipeline
from backend.app.services.emotion_recognition.deep_emotion_classifier import DeepEmotionClassifier
from backend.app.services.emotion_recognition.emotion_service import EmotionRecognitionService


@pytest.fixture(scope="module")
def sample_audio_signal():
    """Generates 2 seconds of 16kHz synthetic sine wave speech-like audio."""
    duration = 2.0
    t = np.linspace(0, duration, int(TARGET_SAMPLE_RATE * duration), endpoint=False)
    # Fundamental frequency ~200Hz + harmonics
    y = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)
    return y.astype(np.float32)


def test_wav2vec2_pipeline_initialization_and_metadata():
    pipeline = Wav2Vec2Pipeline(device="cpu")
    assert pipeline.is_ready()
    assert pipeline.hidden_size == 768

    meta = pipeline.get_metadata()
    assert meta["model_name"] == "Wav2Vec2 Base"
    assert meta["hidden_size"] == 768
    assert meta["sampling_rate"] == 16000
    assert "LibriSpeech" in meta["pretraining_dataset"]


def test_wav2vec2_embedding_extraction_dimension(sample_audio_signal):
    pipeline = Wav2Vec2Pipeline(device="cpu")
    emb = pipeline.extract_embedding(sample_audio_signal, sample_rate=16000, pooling="mean")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (768,)
    assert not np.isnan(emb).any()


def test_hubert_pipeline_initialization_and_metadata():
    pipeline = HuBERTPipeline(device="cpu")
    assert pipeline.is_ready()
    assert pipeline.hidden_size == 768

    meta = pipeline.get_metadata()
    assert meta["model_name"] == "HuBERT Base"
    assert meta["hidden_size"] == 768
    assert meta["sampling_rate"] == 16000
    assert "Masked Unit Prediction" in meta["architecture"]


def test_hubert_embedding_extraction_dimension(sample_audio_signal):
    pipeline = HuBERTPipeline(device="cpu")
    emb = pipeline.extract_embedding(sample_audio_signal, sample_rate=16000, pooling="mean")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (768,)
    assert not np.isnan(emb).any()


def test_deep_emotion_classifier_fit_predict_and_abstention(tmp_path):
    classes = ["angry", "calm", "happy", "sad"]
    rng = np.random.RandomState(42)

    # 40 synthetic samples of 768 dimensions
    X = rng.randn(40, 768).astype(np.float32)
    y = np.array([classes[i % len(classes)] for i in range(40)])

    clf = DeepEmotionClassifier(model_type="wav2vec2", classifier_type="logistic_regression", confidence_threshold=0.30)
    clf.fit(X, y)
    assert clf.classes == sorted(classes)

    # Single inference
    test_vec = rng.randn(768).astype(np.float32)
    result = clf.predict_single(test_vec)
    assert "emotion_label" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "is_reliable" in result
    assert len(result["probabilities"]) == len(classes)
    assert 0.0 <= result["confidence"] <= 1.0

    # Serialization test
    model_path = str(tmp_path / "test_clf.joblib")
    meta_path = str(tmp_path / "test_meta.json")
    clf.save(model_path, meta_path)

    loaded_clf = DeepEmotionClassifier().load(model_path, meta_path)
    assert loaded_clf.classes == clf.classes
    loaded_res = loaded_clf.predict_single(test_vec)
    assert loaded_res["emotion_label"] == result["emotion_label"]


def test_emotion_recognition_service_dispatching(sample_audio_signal):
    # Test service with traditional fallback
    service = EmotionRecognitionService(model_type="traditional_svm")
    res = service.predict_emotion(sample_audio_signal)
    assert res.success is True
    assert "vocal_traits" in res.model_dump()
    assert "pitch_mean_hz" in res.vocal_traits
    assert res.vocal_traits["pitch_mean_hz"] > 0
