import io
import math
import struct
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.ml.feature_extraction import extract_features_vector, FEATURE_NAMES
from backend.app.services.emotion_recognition import (
    get_emotion_service,
    load_and_preprocess_audio,
)

client = TestClient(app)


@pytest.fixture
def synthesized_wav_bytes():
    """Create 1.2s valid 16kHz PCM WAV audio with alternating pitch frequencies."""
    buffer = io.BytesIO()
    sample_rate = 16000
    duration_sec = 1.2
    num_samples = int(sample_rate * duration_sec)

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            # Dynamic pitch tone (200Hz to 350Hz)
            freq = 220 + 80 * math.sin(2 * math.pi * 2 * (i / sample_rate))
            val = int(24000 * math.sin(2 * math.pi * freq * (i / sample_rate)))
            frames.extend(struct.pack("<h", val))
        wav_file.writeframes(frames)

    return buffer.getvalue()


def test_emotion_health_endpoint():
    """Verify emotion service health check."""
    response = client.get("/api/v1/emotion/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_name" in data
    assert len(data["classes"]) == 8


def test_feature_extraction_vector(synthesized_wav_bytes):
    """Verify acoustic feature vector shape and numerical validity."""
    vec = extract_features_vector(synthesized_wav_bytes)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert not np.isnan(vec).any()
    assert not np.isinf(vec).any()


def test_emotion_prediction_service(synthesized_wav_bytes):
    """Verify EmotionRecognitionService produces valid prediction and vocal traits."""
    service = get_emotion_service()
    res = service.predict_emotion(synthesized_wav_bytes)

    assert res.success is True
    assert isinstance(res.emotion_label, str)
    assert 0.0 <= res.confidence <= 1.0
    assert "pitch_mean_hz" in res.vocal_traits
    assert "rms_energy" in res.vocal_traits
    assert "pause_fraction" in res.vocal_traits
    assert "interpretation_notice" in res.model_dump()


def test_predict_endpoint_valid_audio(synthesized_wav_bytes):
    """Test POST /api/v1/emotion/predict with valid audio."""
    files = {"file": ("speech_test.wav", synthesized_wav_bytes, "audio/wav")}
    response = client.post("/api/v1/emotion/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "emotion_label" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "vocal_traits" in data
    assert "interpretation_notice" in data
    assert data["processing_time_sec"] > 0


def test_predict_endpoint_empty_audio():
    """Verify rejection of empty audio files (0 bytes)."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/api/v1/emotion/predict", files=files)
    assert response.status_code == 400


def test_predict_endpoint_unsupported_format():
    """Verify rejection of non-audio files."""
    files = {"file": ("document.txt", b"hello emotion", "text/plain")}
    response = client.post("/api/v1/emotion/predict", files=files)
    assert response.status_code == 400


def test_abstention_on_unreliable_prediction():
    """Verify that low-confidence predictions trigger abstention with 'uncertain'."""
    service = get_emotion_service()
    # Force an unreasonably high threshold to verify abstention behavior
    original_threshold = service.confidence_threshold
    try:
        service.confidence_threshold = 1.05  # impossible confidence (> 1.0)
        # 0.5s soft noise
        noise = np.random.randn(8000).astype(np.float32) * 0.01
        res = service.predict_emotion(noise)
        assert res.is_reliable is False
        assert res.emotion_label == "uncertain"
    finally:
        service.confidence_threshold = original_threshold
