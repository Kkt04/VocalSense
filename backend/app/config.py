from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    APP_NAME: str = "VocalSense API"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Host & Port
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Audio upload parameters
    MAX_AUDIO_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    ALLOWED_AUDIO_EXTENSIONS: List[str] = [
        ".wav",
        ".webm",
        ".ogg",
        ".mp3",
        ".m4a",
        ".mp4",
    ]
    ALLOWED_AUDIO_MIME_TYPES: List[str] = [
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/x-m4a",
        "audio/aac",
        "video/webm",  # Some Chromium versions record webm audio with video/webm mime
    ]

    # Privacy & Storage
    RETAIN_RAW_AUDIO: bool = False
    TEMP_STORAGE_DIR: str = "/tmp/vocalsense/audio"

    # Speech Recognition (Whisper / faster-whisper)
    WHISPER_MODEL_SIZE: str = "base"  # "tiny", "base", "small", "medium", "large-v3"
    WHISPER_DEVICE: str = "auto"       # "auto", "cpu", "cuda"
    WHISPER_COMPUTE_TYPE: str = "auto" # "auto", "int8", "float16", "float32"
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_INITIAL_PROMPT: str = (
        "Conversations in English, Hindi, and Hinglish. "
        "Yaar, mujhe kal presentation complete karna hai."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
