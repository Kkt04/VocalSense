"""
Unit tests for MongoDB persistence layer and collection operations.
Tests users, conversations, messages, emotion insights, model metadata,
in-memory fallback, chronological retrieval, and pagination.
"""

import pytest
from datetime import datetime, timezone
from backend.app.db.mongodb import MongoDBManager
from backend.app.schemas.mongodb import (
    UserProfileSchema,
    ConversationSchema,
    MessageSchema,
    EmotionInsightSchema,
    ModelMetadataSchema,
)


@pytest.fixture
def mock_db():
    # Fresh instance operating in fallback/mock mode
    mgr = MongoDBManager(uri="mongodb://localhost:27017", db_name="test_vocalsense")
    mgr.is_connected = False
    mgr.db = None
    mgr._in_memory_fallback = {
        "users": {},
        "conversations": {},
        "messages": {},
        "emotion_insights": {},
        "model_metadata": {},
    }
    return mgr


@pytest.mark.anyio
async def test_user_persistence_and_retrieval(mock_db):
    user_data = {
        "user_id": "usr_test_001",
        "display_name": "Aarav Sharma",
        "preferred_language": "hi",
        "preferences": {"voice_enabled": True, "style": "just_listen"},
    }
    saved = await mock_db.save_user(user_data)
    assert saved["user_id"] == "usr_test_001"
    assert "created_at" in saved

    fetched = await mock_db.get_user("usr_test_001")
    assert fetched is not None
    assert fetched["display_name"] == "Aarav Sharma"
    assert fetched["preferred_language"] == "hi"

    # Non-existent user
    non_existent = await mock_db.get_user("usr_non_existent")
    assert non_existent is None


@pytest.mark.anyio
async def test_conversation_persistence(mock_db):
    conv_data = {
        "conversation_id": "conv_test_101",
        "user_id": "usr_test_001",
        "style": "help_me_think",
        "language": "hinglish",
    }
    saved = await mock_db.save_conversation(conv_data)
    assert saved["conversation_id"] == "conv_test_101"

    fetched = await mock_db.get_conversation("conv_test_101")
    assert fetched is not None
    assert fetched["style"] == "help_me_think"
    assert fetched["language"] == "hinglish"


@pytest.mark.anyio
async def test_messages_chronological_ordering_and_pagination(mock_db):
    conv_id = "conv_chronological_test"

    # Add 5 messages with sequential timestamps
    for i in range(5):
        msg = {
            "message_id": f"msg_{i}",
            "conversation_id": conv_id,
            "role": "user" if i % 2 == 0 else "assistant",
            "text": f"Message turn {i}",
            "timestamp": datetime(2026, 11, 5, 12, i, 0, tzinfo=timezone.utc),
        }
        await mock_db.add_message(msg)

    # Fetch all
    messages = await mock_db.get_messages(conv_id, limit=10, skip=0)
    assert len(messages) == 5
    assert messages[0]["text"] == "Message turn 0"
    assert messages[4]["text"] == "Message turn 4"

    # Test pagination: limit=2, skip=2
    paginated = await mock_db.get_messages(conv_id, limit=2, skip=2)
    assert len(paginated) == 2
    assert paginated[0]["text"] == "Message turn 2"
    assert paginated[1]["text"] == "Message turn 3"


@pytest.mark.anyio
async def test_emotion_insights_persistence(mock_db):
    insight = {
        "insight_id": "ins_001",
        "conversation_id": "conv_insights",
        "message_id": "msg_001",
        "predicted_vocal_pattern": "calm",
        "probabilities": {"calm": 0.65, "neutral": 0.20, "sad": 0.15},
        "model_name": "wav2vec2_base",
        "model_version": "1.0.0",
        "is_reliable": True,
    }
    saved = await mock_db.save_emotion_insight(insight)
    assert saved["insight_id"] == "ins_001"

    latest = await mock_db.get_latest_emotion_insight("conv_insights")
    assert latest is not None
    assert latest["predicted_vocal_pattern"] == "calm"
    assert latest["probabilities"]["calm"] == 0.65


@pytest.mark.anyio
async def test_model_metadata_persistence(mock_db):
    metadata = {
        "model_name": "Wav2Vec2 Base",
        "model_version": "1.0.0",
        "model_type": "wav2vec2",
        "supported_languages": ["en", "cross-lingual acoustics"],
        "training_dataset": "RAVDESS",
        "evaluation_metrics": {"accuracy": 0.60, "macro_f1": 0.5914},
    }
    saved = await mock_db.save_model_metadata(metadata)
    assert saved["model_name"] == "Wav2Vec2 Base"

    fetched = await mock_db.get_model_metadata("Wav2Vec2 Base")
    assert fetched is not None
    assert fetched["evaluation_metrics"]["macro_f1"] == 0.5914
