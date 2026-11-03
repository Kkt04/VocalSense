import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import settings

client = TestClient(app)


@pytest.fixture
def sample_wav_bytes():
    """Create a minimal valid RIFF/WAVE header buffer."""
    # 4 bytes 'RIFF', 4 bytes file size - 8, 4 bytes 'WAVE', 'fmt ' subchunk, 'data' subchunk
    riff_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    return riff_header


@pytest.fixture
def sample_webm_bytes():
    """Create a minimal EBML (WebM) header buffer."""
    return b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08"


def test_upload_valid_wav(sample_wav_bytes):
    """Test uploading a valid WAV file."""
    files = {"file": ("test_speech.wav", sample_wav_bytes, "audio/wav")}
    data = {"session_id": "test-session-123"}
    response = client.post("/api/audio/upload", files=files, data=data)

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    assert res_data["session_id"] == "test-session-123"
    assert res_data["audio"]["filename"] == "test_speech.wav"
    assert res_data["audio"]["format_detected"] == "wav"
    assert res_data["audio"]["size_bytes"] == len(sample_wav_bytes)


def test_upload_valid_webm(sample_webm_bytes):
    """Test uploading a valid WebM audio file."""
    files = {"file": ("recording.webm", sample_webm_bytes, "audio/webm")}
    response = client.post("/api/audio/upload", files=files)

    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    assert res_data["audio"]["format_detected"] == "webm"


def test_upload_empty_file():
    """Test rejection of empty audio files (0 bytes)."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/api/audio/upload", files=files)

    assert response.status_code == 400
    res_data = response.json()
    assert "detail" in res_data
    assert res_data["detail"]["code"] == "EMPTY_FILE"


def test_upload_unsupported_extension():
    """Test rejection of files with invalid extensions."""
    files = {"file": ("script.py", b"print('hello')", "text/plain")}
    response = client.post("/api/audio/upload", files=files)

    assert response.status_code == 400
    res_data = response.json()
    assert res_data["detail"]["code"] == "UNSUPPORTED_EXTENSION"


def test_upload_oversized_file(monkeypatch):
    """Test rejection of audio files exceeding max size limit."""
    # Temporarily set max size to 100 bytes for testing
    monkeypatch.setattr(settings, "MAX_AUDIO_SIZE_BYTES", 100)

    oversized_data = b"RIFF" + (b"\x00" * 200)
    files = {"file": ("large.wav", oversized_data, "audio/wav")}
    response = client.post("/api/audio/upload", files=files)

    assert response.status_code == 413
    res_data = response.json()
    assert res_data["detail"]["code"] == "FILE_TOO_LARGE"
