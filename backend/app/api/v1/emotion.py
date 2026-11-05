from typing import Optional
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, status
from backend.app.schemas.emotion import (
    EmotionPredictionResponse,
    EmotionHealthResponse,
)
from backend.app.schemas.audio import ErrorDetailResponse
from backend.app.services.audio_validator import (
    AudioValidator,
    AudioValidationError,
)
from backend.app.services.emotion_recognition import (
    get_emotion_service,
    AudioPreprocessingError,
)

router = APIRouter(prefix="/emotion", tags=["Emotion Recognition"])


@router.get("/health", response_model=EmotionHealthResponse)
async def get_emotion_health() -> EmotionHealthResponse:
    """
    Check the status and readiness of the speech emotion recognition model.
    """
    service = get_emotion_service()
    return EmotionHealthResponse(
        status="healthy" if service.is_ready() else "degraded",
        service=f"Speech Emotion Recognition ({service.model_type.upper()})",
        model_loaded=service.is_ready(),
        model_name=service.get_active_model_name(),
        classes=service.classes,
    )


@router.post(
    "/predict",
    response_model=EmotionPredictionResponse,
    responses={
        400: {"model": ErrorDetailResponse, "description": "Invalid audio file or empty speech"},
        413: {"model": ErrorDetailResponse, "description": "Audio file too large"},
        500: {"model": ErrorDetailResponse, "description": "Emotion prediction processing error"},
    },
)
async def predict_emotion_pattern(
    file: UploadFile = File(..., description="Audio recording to analyze (WAV, WebM, MP3, M4A, OGG)"),
    model_type: Optional[str] = Query(None, description="Optional model selector: 'wav2vec2', 'hubert', or 'traditional_svm'"),
) -> EmotionPredictionResponse:
    """
    Analyze vocal acoustic patterns (pitch, energy, spectral features, pauses)
    and predict an emotion pattern using deep speech representations (Wav2Vec2 / HuBERT)
    or traditional ML baseline.

    Returns the predicted pattern, probability distribution, and physical vocal traits.
    """
    # 1. Validate file format and size limits
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

    # 2. Extract features and infer emotion
    try:
        service = get_emotion_service()
        response = service.predict_emotion(content, override_model_type=model_type)
        return response
    except AudioPreprocessingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"Audio processing error: {str(e)}", "code": "AUDIO_PREPROCESSING_ERROR"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Emotion prediction failed: {str(e)}", "code": "PREDICTION_FAILED"},
        )
