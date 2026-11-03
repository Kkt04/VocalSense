import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from backend.app.schemas.audio import (
    AudioUploadResponse,
    AudioMetadata,
    ErrorDetailResponse,
)
from backend.app.services.audio_validator import (
    AudioValidator,
    AudioValidationError,
)
from backend.app.services.audio_storage import AudioStorageService

router = APIRouter(prefix="/audio", tags=["Audio"])
storage_service = AudioStorageService()


@router.post(
    "/upload",
    response_model=AudioUploadResponse,
    responses={
        400: {"model": ErrorDetailResponse, "description": "Validation failure"},
        413: {"model": ErrorDetailResponse, "description": "File too large"},
        500: {"model": ErrorDetailResponse, "description": "Internal server error"},
    },
)
async def upload_audio(
    file: UploadFile = File(..., description="Audio recording file (webm, wav, ogg, mp3, m4a)"),
    session_id: Optional[str] = Form(None, description="Optional conversation session ID"),
) -> AudioUploadResponse:
    """
    Validate and accept an audio recording from the frontend.

    In Phase 1, this endpoint validates:
    - Audio MIME type and container format
    - File size limits (max 25MB)
    - Audio header magic numbers
    - Safe ephemeral handling with zero permanent retention by default
    """
    try:
        content, detected_format, size_bytes = await AudioValidator.validate_upload(file)
    except AudioValidationError as e:
        status_code = (
            status.HTTP_413_CONTENT_TOO_LARGE
            if e.code == "FILE_TOO_LARGE"
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(
            status_code=status_code,
            detail={"error": e.message, "code": e.code},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to process audio: {str(e)}", "code": "UPLOAD_ERROR"},
        )

    # Save to ephemeral storage for processing
    temp_path = None
    try:
        temp_path = storage_service.save_ephemeral(content, ext=detected_format)
    except Exception as e:
        # Ephemeral storage error should not crash the upload if storage is not critical
        pass

    # Enforce privacy by design: delete immediately if raw storage is disabled
    storage_service.handle_upload_privacy(temp_path)

    active_session_id = session_id or str(uuid.uuid4())
    size_kb = round(size_bytes / 1024, 2)

    metadata = AudioMetadata(
        filename=file.filename or "recording.webm",
        content_type=file.content_type or f"audio/{detected_format}",
        size_bytes=size_bytes,
        size_kb=size_kb,
        format_detected=detected_format,
        validated=True,
        details={
            "privacy_mode": "zero_retention",
            "ephemeral_processed": True,
        },
    )

    return AudioUploadResponse(
        success=True,
        message="Audio received and validated successfully.",
        audio=metadata,
        session_id=active_session_id,
    )
