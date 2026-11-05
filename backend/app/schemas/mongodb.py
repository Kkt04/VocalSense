from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class UserProfileSchema(BaseModel):
    user_id: str
    display_name: Optional[str] = None
    preferred_language: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationSchema(BaseModel):
    conversation_id: str
    user_id: Optional[str] = None
    style: str = "just_listen"
    language: str = "en"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MessageSchema(BaseModel):
    message_id: str
    conversation_id: str
    role: str  # "user" | "assistant"
    text: str
    detected_language: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmotionInsightSchema(BaseModel):
    insight_id: str
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    predicted_vocal_pattern: str
    probabilities: Dict[str, float] = Field(default_factory=dict)
    model_name: str
    model_version: str
    is_reliable: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelMetadataSchema(BaseModel):
    model_name: str
    model_version: str
    model_type: str  # "traditional_svm", "wav2vec2", "hubert"
    supported_languages: List[str] = Field(default_factory=list)
    training_dataset: str
    evaluation_metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
