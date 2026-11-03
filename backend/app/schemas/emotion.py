from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class EmotionPredictionResponse(BaseModel):
    """Response schema for speech emotion pattern prediction."""

    success: bool = True
    emotion_label: str = Field(..., description="Estimated emotion category or 'uncertain'")
    confidence: float = Field(..., description="Estimated prediction probability score (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="Class-wise probability distribution")
    vocal_traits: Dict[str, Any] = Field(default_factory=dict, description="Extracted physical acoustic traits")
    is_reliable: bool = Field(True, description="Whether prediction meets minimum reliability confidence threshold")
    interpretation_notice: str = Field(
        "This prediction identifies measurable vocal acoustic patterns, not a person's actual inner state or diagnosis. Your stated words always take precedence.",
        description="Responsible AI disclaimer",
    )
    model_used: str = Field(..., description="Trained model identifier and version")
    processing_time_sec: float = Field(..., description="Inference latency in seconds")


class EmotionHealthResponse(BaseModel):
    """Health check response for the emotion recognition service."""

    status: str = "healthy"
    service: str = "Speech Emotion Recognition Service"
    model_loaded: bool = False
    model_name: str = "Not Loaded"
    classes: List[str] = Field(default_factory=list)
