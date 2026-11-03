import wave
import struct
import io
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.transcription.audio_preprocessing import (
    AudioPreprocessor,
    PreprocessingError,
)
from backend.app.services.transcription.language_detection import (
    detect_language_characteristics,
)
from backend.evaluation.wer_cer import compute_metrics, normalize_text_for_eval

client = TestClient(app)


@pytest.fixture
def synthesized_wav_bytes():
    """Create 1.0s valid 16kHz PCM WAV audio containing a soft 440Hz tone."""
    buffer = io.BytesIO()
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            # 440 Hz soft sine-like wave
            import math
            sample_val = int(16000 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
            frames.extend(struct.pack("<h", sample_val))
        wav_file.writeframes(frames)

    return buffer.getvalue()


def test_transcription_health():
    """Verify speech recognition health check endpoint."""
    response = client.get("/api/v1/transcription/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_size" in data
    assert "device" in data


def test_audio_preprocessor_resampling(synthesized_wav_bytes):
    """Test loading and resampling audio into 16kHz mono float32."""
    audio_array, duration = AudioPreprocessor.load_and_resample(synthesized_wav_bytes)
    assert len(audio_array) == 16000
    assert abs(duration - 1.0) < 0.05
    assert audio_array.dtype.name == "float32"


def test_audio_preprocessor_empty():
    """Test that preprocessor rejects empty or corrupted data."""
    with pytest.raises(PreprocessingError):
        AudioPreprocessor.load_and_resample(b"")


def test_language_detection_hinglish():
    """Verify Hinglish code-switching detection for romanized Hindi + English terms."""
    text = "Yaar mujhe kal apne project ka presentation complete karna hai"
    lang, is_hinglish, details = detect_language_characteristics(
        whisper_detected_lang="hi",
        language_prob=0.88,
        transcript_text=text,
    )
    assert is_hinglish is True
    assert lang == "hinglish"
    assert "yaar" in details["hinglish_markers_found"]


def test_language_detection_pure_english():
    """Verify English detection for standard English text."""
    text = "I am preparing for my technical interview tomorrow morning."
    lang, is_hinglish, details = detect_language_characteristics(
        whisper_detected_lang="en",
        language_prob=0.99,
        transcript_text=text,
    )
    assert is_hinglish is False
    assert lang == "en"


def test_language_detection_hindi_devanagari():
    """Verify Hindi detection for Devanagari script."""
    text = "मुझे आज बहुत अच्छा महसूस हो रहा है"
    lang, is_hinglish, details = detect_language_characteristics(
        whisper_detected_lang="hi",
        language_prob=0.97,
        transcript_text=text,
    )
    assert is_hinglish is False
    assert lang == "hi"


def test_transcribe_endpoint_valid_audio(synthesized_wav_bytes):
    """Test POST /api/v1/transcription/transcribe with synthetic audio."""
    files = {"file": ("tone.wav", synthesized_wav_bytes, "audio/wav")}
    response = client.post("/api/v1/transcription/transcribe", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "transcript" in data
    assert "detected_language" in data
    assert data["duration"] >= 0.9
    assert "processing_time_sec" in data
    assert "segments" in data


def test_wer_cer_metrics_exact_match():
    """Test that identical strings produce 0.0 WER and 0.0 CER."""
    ref = "Sometimes you just need someone to listen."
    hyp = "Sometimes you just need someone to listen."
    res = compute_metrics(ref, hyp)
    assert res["wer"] == 0.0
    assert res["cer"] == 0.0


def test_wer_cer_metrics_with_errors():
    """Test WER and CER calculations with substitution and deletion."""
    ref = "I want to complete my project today."
    hyp = "I want to finish my project"
    res = compute_metrics(ref, hyp)
    assert res["wer"] > 0.0
    assert res["cer"] > 0.0
    assert res["deletions"] >= 1
    assert res["substitutions"] >= 1
