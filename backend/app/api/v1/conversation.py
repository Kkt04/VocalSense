"""
API router for Multilingual Emotion-Aware Conversation Intelligence.
Endpoints:
- POST /api/v1/conversation/respond: Generate language-matched companion response
- GET /api/v1/conversation/history/{conversation_id}: Retrieve message history from MongoDB
- POST /api/v1/conversation/user/preference: Update user's explicit language preference
- GET /api/v1/conversation/health: Check conversation service status
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status

from backend.app.schemas.conversation import (
    ConversationRespondRequest,
    ConversationRespondResponse,
)
from backend.app.schemas.mongodb import UserProfileSchema, MessageSchema
from backend.app.services.conversation.response_generator import ConversationResponseGenerator
from backend.app.db.mongodb import mongodb_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversation", tags=["Conversational Intelligence"])


@router.get("/health")
async def get_conversation_health() -> Dict[str, Any]:
    """Health check for the conversational intelligence service."""
    return {
        "status": "healthy",
        "service": "Multilingual Emotion-Aware Conversation Service",
        "supported_languages": [
            {"code": "en", "name": "English"},
            {"code": "hi", "name": "Hindi (हिंदी)"},
            {"code": "hinglish", "name": "Hinglish (Code-Switched)"},
            {"code": "es", "name": "Spanish (Español)"},
            {"code": "fr", "name": "French (Français)"},
            {"code": "de", "name": "German (Deutsch)"},
            {"code": "ja", "name": "Japanese (日本語)"},
            {"code": "bn", "name": "Bengali (বাংলা)"},
            {"code": "ta", "name": "Tamil (தமிழ்)"},
        ],
        "supported_styles": [
            "just_listen",
            "help_me_think",
            "practical_suggestions",
            "distract_me",
        ],
        "mongodb_connected": mongodb_manager.is_connected,
        "database": mongodb_manager.db_name,
    }


@router.post(
    "/respond",
    response_model=ConversationRespondResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate language-matched companion response",
)
async def respond_to_user(request: ConversationRespondRequest) -> ConversationRespondResponse:
    """
    Main conversational endpoint:
    1. Analyzes user transcript and detected language.
    2. Respects user language preference or auto-detected language.
    3. Guarantees companion response is generated in the matching language.
    4. Validates output language and safely falls back if necessary.
    5. Optionally synthesizes language-matched TTS audio.
    6. Persists conversation, messages, and emotion insights to MongoDB.
    """
    if not request.transcript or not request.transcript.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Transcript cannot be empty",
        )

    try:
        response = await ConversationResponseGenerator.respond(request)
        return response
    except Exception as exc:
        logger.error("Error generating conversation response: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate conversation response: {str(exc)}",
        )


@router.get(
    "/history/{conversation_id}",
    summary="Retrieve conversation history from MongoDB",
)
async def get_conversation_history(
    conversation_id: str,
    limit: int = Query(50, ge=1, le=200, description="Number of messages to retrieve"),
    skip: int = Query(0, ge=0, description="Offset for pagination"),
) -> Dict[str, Any]:
    """
    Retrieves chronological messages for a conversation from MongoDB.
    """
    try:
        conv = await mongodb_manager.get_conversation(conversation_id)
        messages = await mongodb_manager.get_messages(
            conversation_id=conversation_id,
            limit=limit,
            skip=skip,
        )
        latest_emotion = await mongodb_manager.get_latest_emotion_insight(conversation_id)

        return {
            "conversation_id": conversation_id,
            "conversation_metadata": conv,
            "messages": messages,
            "latest_emotion_insight": latest_emotion,
            "count": len(messages),
            "limit": limit,
            "skip": skip,
        }
    except Exception as exc:
        logger.error("Error retrieving conversation history: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch conversation history: {str(exc)}",
        )


@router.post(
    "/user/preference",
    summary="Set or update user language preference in MongoDB",
)
async def set_user_preference(user_profile: UserProfileSchema) -> Dict[str, Any]:
    """
    Stores or updates user display name, preferred response language, and settings in MongoDB.
    """
    try:
        saved = await mongodb_manager.save_user(user_profile.model_dump())
        return {
            "success": True,
            "user": saved,
        }
    except Exception as exc:
        logger.error("Error saving user preference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save user preference: {str(exc)}",
        )
