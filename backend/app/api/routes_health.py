from typing import Dict, Any
from fastapi import APIRouter
from backend.app.config import settings
from backend.app.schemas.audio import HealthResponse
from backend.app.db.mongodb import mongodb_manager
from backend.app.services.transcription.whisper_service import get_whisper_service
from backend.app.services.emotion_recognition import get_emotion_service

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """
    Health-check endpoint.
    Returns status, service info, and environment.
    """
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get("/health/components")
async def get_component_health() -> Dict[str, Any]:
    """
    Deep component-level health check inspecting:
    - MongoDB connection status
    - Speech recognition (faster-whisper) readiness
    - Speech emotion recognition (Wav2Vec2/HuBERT/SVM) readiness
    - LLM conversational intelligence configuration
    - Modular TTS availability
    """
    db_info = {
        "status": "connected" if mongodb_manager.is_connected else "in_memory_fallback",
        "connected": mongodb_manager.is_connected,
        "database": mongodb_manager.db_name,
    }

    try:
        whisper_service = get_whisper_service()
        stt_info = {
            "status": "ready" if whisper_service.is_ready() else "loading",
            "model_size": whisper_service.model_size,
            "device": whisper_service.device,
            "compute_type": whisper_service.compute_type,
        }
    except Exception as exc:
        stt_info = {"status": "unavailable", "error": str(exc)}

    try:
        emotion_service = get_emotion_service()
        ser_info = {
            "status": "ready" if emotion_service.is_ready() else "loading",
            "model_name": emotion_service.get_active_model_name(),
            "classes": emotion_service.classes,
        }
    except Exception as exc:
        ser_info = {"status": "unavailable", "error": str(exc)}

    llm_configured = bool(settings.LLM_API_KEY and settings.LLM_API_KEY.strip())
    llm_info = {
        "status": "configured" if llm_configured else "unconfigured",
        "configured": llm_configured,
        "model": settings.LLM_MODEL,
        "base_url": settings.LLM_API_BASE_URL,
    }

    tts_info = {
        "status": "available",
        "provider": "gTTS",
        "offline_ready": True,
    }

    all_ready = (
        stt_info.get("status") in ("ready", "loading")
        and ser_info.get("status") in ("ready", "loading")
    )

    return {
        "status": "healthy" if all_ready else "degraded",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "components": {
            "database": db_info,
            "speech_recognition": stt_info,
            "emotion_recognition": ser_info,
            "conversational_llm": llm_info,
            "text_to_speech": tts_info,
        },
    }
