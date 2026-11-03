from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., json_schema_extra={"example": "healthy"})
    app_name: str = Field(..., json_schema_extra={"example": "VocalSense API"})
    version: str = Field(..., json_schema_extra={"example": "0.1.0"})
    environment: str = Field(..., json_schema_extra={"example": "development"})
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AudioMetadata(BaseModel):
    """Metadata extracted from the uploaded audio file."""

    filename: str
    content_type: str
    size_bytes: int
    size_kb: float
    format_detected: str
    validated: bool = True
    details: Optional[Dict[str, Any]] = None


class AudioUploadResponse(BaseModel):
    """Response schema for successful audio upload."""

    success: bool = True
    message: str = "Audio file received and validated successfully."
    audio: AudioMetadata
    session_id: Optional[str] = None
    processed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ErrorDetailResponse(BaseModel):
    """Standard error response schema."""

    success: bool = False
    error: str
    detail: Optional[str] = None
    code: str = "INVALID_REQUEST"
