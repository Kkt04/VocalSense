from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from backend.app.schemas.transcription import (
    TranscriptionResponse,
    TranscriptionHealthResponse,
)
from backend.app.schemas.audio import ErrorDetailResponse
from backend.app.services.audio_validator import (
    AudioValidator,
    AudioValidationError,
)
from backend.app.services.transcription.audio_preprocessing import PreprocessingError
from backend.app.services.transcription.whisper_service import get_whisper_service

router = APIRouter(prefix="/transcription", tags=["Speech Recognition"])


@router.get("/health", response_model=TranscriptionHealthResponse)
async def get_transcription_health() -> TranscriptionHealthResponse:
    """
    Check availability and operational status of the speech recognition model.
    """
    try:
        service = get_whisper_service()
        return TranscriptionHealthResponse(
            status="healthy" if service.is_ready() else "degraded",
            service="Speech Recognition Service (faster-whisper)",
            model_size=service.model_size,
            device=service.device,
            compute_type=service.compute_type,
            model_loaded=service.is_ready(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Transcription model unavailable: {str(e)}",
        )


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={
        400: {"model": ErrorDetailResponse, "description": "Invalid audio file or empty speech"},
        413: {"model": ErrorDetailResponse, "description": "Audio file too large"},
        500: {"model": ErrorDetailResponse, "description": "Transcription error"},
        503: {"model": ErrorDetailResponse, "description": "Model service unavailable"},
    },
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe (WAV, WebM, MP3, M4A, OGG)"),
    language: Optional[str] = Form(None, description="Optional language override ('en', 'hi', or auto)"),
    prompt: Optional[str] = Form(None, description="Optional prompt to guide vocabulary or code-switching"),
) -> TranscriptionResponse:
    """
    Transcribe spoken speech into text using Whisper.
    Supports English, Hindi, and Hinglish with zero-shot code-switching preservation.
    """
    # 1. Validate file format and size
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

    # 2. Transcribe via Whisper Service
    try:
        service = get_whisper_service()
        result = service.transcribe(
            audio_input=content,
            language=language,
            initial_prompt=prompt,
        )
        return result
    except PreprocessingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": e.message, "code": e.code},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Transcription processing failed: {str(e)}", "code": "TRANSCRIPTION_FAILED"},
        )
