from typing import List, Optional
from pydantic import BaseModel, Field


class TranscriptionSegment(BaseModel):
    """Segment-level transcription with start and end timestamps."""

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    avg_logprob: Optional[float] = Field(None, description="Average log probability")
    no_speech_prob: Optional[float] = Field(None, description="Probability of no speech")


class TranscriptionResponse(BaseModel):
    """Response schema for speech-to-text transcription."""

    success: bool = True
    transcript: str = Field(..., description="Full consolidated transcript")
    detected_language: str = Field(..., description="Detected language code (e.g. 'en', 'hi', 'hinglish')")
    language_probability: float = Field(..., description="Confidence score of language detection (0.0 to 1.0)")
    duration: float = Field(..., description="Audio duration in seconds")
    segments: List[TranscriptionSegment] = Field(default_factory=list, description="Timestamped segments")
    is_hinglish: bool = Field(False, description="Flag indicating potential code-switching (Hinglish)")
    model_used: str = Field(..., description="Whisper model size/name used for inference")
    processing_time_sec: float = Field(..., description="Time taken to process and transcribe in seconds")
    message: str = "Audio transcribed successfully."


class TranscriptionHealthResponse(BaseModel):
    """Health check response for the transcription service."""

    status: str = "healthy"
    service: str = "Speech Recognition Service"
    model_size: str
    device: str
    compute_type: str
    model_loaded: bool
