from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: str  # "user" or "assistant"
    text: str
    language: Optional[str] = None


class ConversationRespondRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="Transcribed user speech")
    conversation_id: Optional[str] = Field(None, description="Active conversation identifier")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    detected_language: Optional[str] = Field(None, description="Language detected from speech (e.g. en, hi, hinglish, es, fr)")
    language_code: Optional[str] = Field(None, description="ISO-639 language code")
    language_preference: Optional[str] = Field(None, description="Explicit user language override")
    conversation_style: str = Field(
        default="just_listen",
        description="Companion style: just_listen, help_me_think, practical_suggestions, distract_me"
    )
    emotion_estimate: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional acoustic or deep-learning vocal pattern prediction"
    )
    history: Optional[List[ConversationTurn]] = Field(
        default_factory=list,
        description="Optional recent conversation turns"
    )
    synthesize_speech: bool = Field(
        default=False,
        description="Whether to generate TTS audio for the response"
    )


class ConversationRespondResponse(BaseModel):
    response: str
    response_language: str
    language_code: str
    conversation_style: str
    conversation_id: str
    user_id: Optional[str] = None
    tts_available: bool = False
    tts_audio_base64: Optional[str] = None
    audio_mime_type: Optional[str] = None
    validation_passed: bool = True
    code_switching_preserved: bool = False
    error: Optional[str] = None
