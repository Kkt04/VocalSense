"""Speech recognition and multilingual transcription package."""

from .audio_preprocessing import AudioPreprocessor, PreprocessingError
from .whisper_service import WhisperService, get_whisper_service
from .language_detection import detect_language_characteristics

__all__ = [
    "AudioPreprocessor",
    "PreprocessingError",
    "WhisperService",
    "get_whisper_service",
    "detect_language_characteristics",
]
