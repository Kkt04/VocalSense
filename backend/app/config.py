from pathlib import Path
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
    # Whisper initial prompt: rich multilingual vocabulary to bias Whisper toward
    # preserving code-switching (Hinglish) rather than translating into English.
    # Includes Devanagari, Romanized Hindi, and mixed Hindi-English sentences so the
    # model's vocabulary index covers the full Hinglish lexicon.
    WHISPER_INITIAL_PROMPT: str = (
        "Multilingual conversation in English, Hindi, and Hinglish (Hindi-English code-switching). "
        "Yaar, aaj mera mood bilkul theek nahi hai. "
        "Mujhe samajh nahi aa raha kya karun. "
        "College mein bhi kuch accha nahi lag raha. "
        "I am feeling really stressed because of my exams. "
        "Chalo, hum iske baare mein baat karte hain. "
        "Main sun raha hoon, koi tension nahi."
    )

    # LLM provider for conversational response generation.
    # Uses an OpenAI-compatible chat completions API so any provider works
    # (Gemini, Groq, OpenRouter, OpenAI, local Ollama, etc.).
    # Leave LLM_API_KEY empty to disable LLM and use the template fallback.
    LLM_API_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    LLM_API_KEY: str = ""  # Set via .env: LLM_API_KEY=<your-key>
    LLM_MODEL: str = "gemini-2.0-flash"  # Override via .env: LLM_MODEL=gpt-4o-mini
    LLM_MAX_TOKENS: int = 400
    LLM_TEMPERATURE: float = 0.85
    LLM_TIMEOUT_SECONDS: float = 20.0

    # MongoDB Settings
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "vocalsense"

    # Emotion Recognition Deep Learning & Traditional ML
    EMOTION_MODEL_TYPE: str = "wav2vec2"  # "wav2vec2", "hubert", "traditional_svm"
    WAV2VEC2_CHECKPOINT: str = "facebook/wav2vec2-base"
    HUBERT_CHECKPOINT: str = "facebook/hubert-base-ls960"
    EMOTION_DEVICE: str = "auto"          # "auto", "mps", "cuda", "cpu"
    EMOTION_CONFIDENCE_THRESHOLD: float = 0.30
    EMOTION_SAVED_MODELS_DIR: str = "backend/saved_models"

    # Text-to-Speech
    ENABLE_TTS: bool = True

    model_config = SettingsConfigDict(
        env_file=[str(Path(__file__).resolve().parent.parent.parent / ".env"), ".env"],
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
