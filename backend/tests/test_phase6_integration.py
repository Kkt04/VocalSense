"""
Phase 6 integration and production readiness tests.
Validates:
1. Deep component health check endpoint (/api/health/components)
2. Request correlation ID middleware (X-Request-ID & X-Response-Time-Ms)
3. Paginated conversation listing endpoint (/api/v1/conversation/list)
4. Conversation deletion endpoint (/api/v1/conversation/{conversation_id})
5. GDPR / DPDP user data purge endpoint (/api/v1/conversation/user/{user_id}/data)
6. Non-fabrication guardrail: 503 Service Unavailable when LLM is unconfigured
"""

import uuid
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.db.mongodb import mongodb_manager

client = TestClient(app)


def test_component_health_check():
    """Verify component-level health endpoint reports all subsystems."""
    response = client.get("/api/health/components")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "components" in data

    components = data["components"]
    assert "database" in components
    assert "speech_recognition" in components
    assert "emotion_recognition" in components
    assert "conversational_llm" in components
    assert "text_to_speech" in components

    assert "status" in components["database"]
    assert "status" in components["speech_recognition"]
    assert "status" in components["emotion_recognition"]
    assert "configured" in components["conversational_llm"]
    assert components["text_to_speech"]["provider"] == "gTTS"


def test_correlation_id_and_timing_headers():
    """Verify X-Request-ID and X-Response-Time-Ms headers are injected by middleware."""
    custom_id = "test-custom-req-12345"
    response = client.get("/api/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id
    assert "X-Response-Time-Ms" in response.headers

    # Auto-generated request ID when none provided
    auto_res = client.get("/api/health")
    assert auto_res.status_code == 200
    assert auto_res.headers.get("X-Request-ID", "").startswith("req_")


@pytest.mark.anyio
async def test_conversation_list_and_pagination():
    """Verify paginated listing of conversations."""
    test_user = f"user_p6_{uuid.uuid4().hex[:8]}"

    # Seed 3 conversations
    for i in range(3):
        cid = f"conv_p6_{i}_{uuid.uuid4().hex[:6]}"
        await mongodb_manager.save_conversation({
            "conversation_id": cid,
            "user_id": test_user,
            "style": "just_listen",
            "language": "en",
        })
        await mongodb_manager.add_message({
            "message_id": f"msg_seed_{i}",
            "conversation_id": cid,
            "role": "user",
            "text": f"Seed message {i}",
        })

    response = client.get(f"/api/v1/conversation/list?user_id={test_user}&limit=2&skip=0")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["conversations"]) <= 2
    assert data["user_id"] == test_user

    # Page 2
    res_page2 = client.get(f"/api/v1/conversation/list?user_id={test_user}&limit=2&skip=2")
    assert res_page2.status_code == 200
    assert len(res_page2.json()["conversations"]) <= 1


@pytest.mark.anyio
async def test_delete_conversation_endpoint():
    """Verify deleting a specific conversation purges it."""
    cid = f"conv_to_delete_{uuid.uuid4().hex[:8]}"
    await mongodb_manager.save_conversation({
        "conversation_id": cid,
        "user_id": "test_del_user",
    })
    await mongodb_manager.add_message({
        "message_id": "msg_to_del_1",
        "conversation_id": cid,
        "role": "user",
        "text": "Hello, please delete me",
    })

    # Delete existing
    del_res = client.delete(f"/api/v1/conversation/{cid}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # Delete non-existent should return 404
    del_res_404 = client.delete(f"/api/v1/conversation/non_existent_{uuid.uuid4().hex[:6]}")
    assert del_res_404.status_code == 404


@pytest.mark.anyio
async def test_purge_user_data_gdpr_endpoint():
    """Verify GDPR right-to-erasure endpoint purges all user data."""
    u_id = f"user_gdpr_{uuid.uuid4().hex[:8]}"
    cid1 = f"conv_gdpr_1_{uuid.uuid4().hex[:6]}"
    cid2 = f"conv_gdpr_2_{uuid.uuid4().hex[:6]}"

    await mongodb_manager.save_user({"user_id": u_id, "display_name": "GDPR User"})
    await mongodb_manager.save_conversation({"conversation_id": cid1, "user_id": u_id})
    await mongodb_manager.save_conversation({"conversation_id": cid2, "user_id": u_id})
    await mongodb_manager.add_message({"message_id": "m1", "conversation_id": cid1, "role": "user", "text": "M1"})
    await mongodb_manager.add_message({"message_id": "m2", "conversation_id": cid2, "role": "user", "text": "M2"})

    purge_res = client.delete(f"/api/v1/conversation/user/{u_id}/data")
    assert purge_res.status_code == 200
    pdata = purge_res.json()
    assert pdata["success"] is True
    assert pdata["user_id"] == u_id
    assert pdata["purged"]["user_deleted"] is True
    assert pdata["purged"]["conversations_deleted"] >= 2


def test_no_hardcoded_conversational_response_fallback_returns_503():
    """
    Phase 6 strict rule: Never return hardcoded/canned conversational replies.
    When the LLM client fails or returns None, the endpoint must return 503
    Service Unavailable with a clear technical error.
    """
    with patch("backend.app.services.conversation.response_generator.LLMClient.generate", return_value=None):
        payload = {
            "transcript": "I am feeling really tired and anxious.",
            "conversation_style": "just_listen",
        }
        res = client.post("/api/v1/conversation/respond", json=payload)
        assert res.status_code == 503
        data = res.json()
        assert "temporarily unavailable" in data["detail"].lower()
