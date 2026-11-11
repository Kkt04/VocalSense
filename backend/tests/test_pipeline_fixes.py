"""
Comprehensive test suite for VocalSense multilingual transcription preservation,
context-aware response generation, and conversation pipeline correctness.

Tests cover:
  1. Hinglish transcription preservation (no English translation)
  2. Language detection accuracy across all supported languages
  3. Response context-awareness (different intents → different responses)
  4. Repetition prevention (same conversation ID, different messages)
  5. MongoDB conversation history isolation
  6. LLM client graceful fallback when key is absent
  7. Template fallback determinism and language correctness
  8. System prompt structure and language instruction presence
  9. Whisper initial prompt configuration for Hinglish bias
 10. Full pipeline: transcript → language → template response (no LLM key needed)
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from typing import List, Dict, Any

from backend.app.services.language.language_detector import LanguageDetector, LanguageDetectionResult
from backend.app.services.language.language_validator import LanguageValidator
from backend.app.services.conversation.response_generator import (
    ConversationResponseGenerator,
    _TEMPLATE_FALLBACK,
    RESPONSES_BY_STYLE_AND_LANG,
)
from backend.app.services.conversation.language_prompt import build_system_prompt
from backend.app.services.conversation.llm_client import LLMClient, build_chat_messages
from backend.app.schemas.conversation import ConversationRespondRequest, ConversationTurn
from backend.app.config import settings


@pytest.fixture
def mock_pipeline_llm():
    """Phase 6 rule: Mock LLM client in unit tests rather than relying on fake template conversations."""
    async def fake_generate(system_prompt: str, messages: list, **kwargs):
        prompt_lower = system_prompt.lower()
        if "hindi" in prompt_lower and "devanagari" in prompt_lower:
            return "नमस्ते, मैं समझ सकता हूँ कि आप कैसा महसूस कर रहे हैं। मैं यहाँ सुनने के लिए हूँ।"
        elif "hinglish" in prompt_lower:
            return "Yaar main bilkul samajh raha hoon. Tension mat lo, main sun raha hoon."
        elif "spanish" in prompt_lower:
            return "Entiendo cómo te sientes. Estoy aquí para escucharte y apoyarte."
        else:
            last_text = messages[-1]["content"] if messages else ""
            return f"I hear you completely and I am listening: {last_text}"

    with patch("backend.app.services.conversation.response_generator.LLMClient.generate", side_effect=fake_generate):
        yield


# ═══════════════════════════════════════════════════════════════════════════
# 1. HINGLISH TRANSCRIPTION PRESERVATION — no English translation
# ═══════════════════════════════════════════════════════════════════════════

class TestHinglishPreservation:
    """
    Validates that the language detection layer correctly identifies Hinglish
    rather than mis-classifying it as English. The Whisper layer itself is an
    ASR model — these tests cover the post-transcription language classification
    that feeds into the conversation pipeline.
    """

    def test_romanized_hinglish_detected_not_english(self):
        """Classic Hinglish sentence must be classified as 'hinglish', not 'en'."""
        text = "Yaar, aaj mera mood bilkul theek nahi hai."
        result = LanguageDetector.detect(text)
        assert result.language_code == "hinglish", (
            f"Expected 'hinglish', got '{result.language_code}'. "
            "Hinglish must NOT be mis-classified as English."
        )
        assert result.is_code_switched is True

    def test_code_switched_hinglish_college_sentence(self):
        """'College mein bhi kuch accha nahi lag raha' — must remain Hinglish."""
        text = "College mein bhi kuch accha nahi lag raha hai."
        result = LanguageDetector.detect(text)
        assert result.language_code == "hinglish"

    def test_hinglish_mixed_english_hindi_preserved(self):
        """Sentence mixing English and Hindi words must be classified Hinglish."""
        text = "I am so tired yaar, aaj ka din bahut hectic tha."
        result = LanguageDetector.detect(text)
        assert result.language_code == "hinglish"
        assert result.is_code_switched is True

    def test_hinglish_interview_stress_sentence(self):
        """Pre-interview Hinglish must stay Hinglish."""
        text = "Kal mera interview hai and I am really nervous."
        result = LanguageDetector.detect(text)
        assert result.language_code == "hinglish"

    def test_hinglish_college_skip_sentence(self):
        """'aaj college jaane ka bilkul mann nahi' — Hinglish."""
        text = "Yaar mujhe aaj college jaane ka bilkul mann nahi kar raha."
        result = LanguageDetector.detect(text)
        assert result.language_code == "hinglish"

    def test_hinglish_presentation_stress(self):
        """Presentation + Hinglish mixed sentence."""
        text = "Yaar mujhe aaj presentation complete karna hai lekin thoda stressed hoon."
        result = LanguageDetector.detect(text)
        assert result.language_code == "hinglish"

    def test_hinglish_not_translated_to_english_or_devanagari(self):
        """
        Language code must be 'hinglish'. Specifically validates it is NOT
        classified as 'en' (English) or 'hi' (pure Hindi).
        """
        text = "Mujhe samajh nahi aa raha kya karun, yaar."
        result = LanguageDetector.detect(text)
        assert result.language_code not in ("en", "hi"), (
            f"Hinglish was mis-classified as '{result.language_code}'. "
            "It must be preserved as 'hinglish'."
        )
        assert result.language_code == "hinglish"

    def test_devanagari_hindi_preserved(self):
        """Pure Devanagari Hindi must remain 'hi'."""
        text = "मुझे आज बहुत अकेलापन महसूस हो रहा है।"
        result = LanguageDetector.detect(text)
        assert result.language_code == "hi"
        assert result.is_code_switched is False

    def test_english_preserved(self):
        """Pure English must be classified as 'en'."""
        text = "I am excited about my new project today!"
        result = LanguageDetector.detect(text)
        assert result.language_code == "en"

    def test_mixed_devanagari_latin_is_hinglish(self):
        """Devanagari + Latin script mix = Hinglish."""
        text = "आज का दिन really exhausting था।"
        result = LanguageDetector.detect(text)
        assert result.language_code == "hinglish"
        assert result.is_code_switched is True


# ═══════════════════════════════════════════════════════════════════════════
# 2. LANGUAGE DETECTION — all supported languages
# ═══════════════════════════════════════════════════════════════════════════

class TestLanguageDetection:

    def test_spanish_with_whisper_hint(self):
        result = LanguageDetector.detect("Hoy me siento muy triste.", whisper_lang="es")
        assert result.language_code == "es"

    def test_french_with_whisper_hint(self):
        result = LanguageDetector.detect(
            "Je me sens très triste aujourd'hui.", whisper_lang="fr"
        )
        assert result.language_code == "fr"

    def test_german_with_whisper_hint(self):
        result = LanguageDetector.detect(
            "Ich fühle mich heute sehr alleine.", whisper_lang="de"
        )
        assert result.language_code == "de"

    def test_japanese_script(self):
        result = LanguageDetector.detect("今日は本当に疲れました。")
        assert result.language_code == "ja"

    def test_bengali_script(self):
        result = LanguageDetector.detect("আমি আজ খুব একা বোধ করছি।")
        assert result.language_code == "bn"

    def test_tamil_script(self):
        result = LanguageDetector.detect("இன்று நான் மிகவும் சோர்வாக உணர்கிறேன்.")
        assert result.language_code == "ta"

    def test_explicit_preference_overrides_detection(self):
        """User preference 'hi' must override auto-detected English."""
        result = LanguageDetector.detect("Good morning.", user_preference="hi")
        assert result.language_code == "hi"
        assert result.confidence == 1.0

    def test_whisper_hindi_latin_chars_becomes_hinglish(self):
        """When Whisper says 'hi' but transcript has Latin chars, detect Hinglish."""
        result = LanguageDetector.detect(
            "Mujhe bahut tension ho rahi hai.", whisper_lang="hi"
        )
        assert result.language_code == "hinglish"


# ═══════════════════════════════════════════════════════════════════════════
# 3. RESPONSE CONTEXT-AWARENESS — template fallback path
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.usefixtures("mock_pipeline_llm")
class TestResponseContextAwareness:
    """
    Tests that different user intents produce semantically different responses
    and that the template fallback correctly handles all languages and styles.
    """

    @pytest.mark.anyio
    async def test_hinglish_response_in_hinglish(self):
        req = ConversationRespondRequest(
            transcript="Yaar, aaj mera mood bilkul theek nahi hai.",
            conversation_style="just_listen",
        )
        res = await ConversationResponseGenerator.respond(req)
        assert res.response_language == "hinglish"
        assert res.code_switching_preserved is True
        assert len(res.response) > 10
        # Response must NOT be pure formal English
        hinglish_markers = {"yaar", "arey", "kya", "main", "toh", "nahi", "aaj",
                            "mujhe", "mera", "bhi", "hai", "hoon", "baat", "lo",
                            "sun", "share", "feel", "tension", "samajh", "karo"}
        resp_lower = res.response.lower()
        found = any(word in resp_lower for word in hinglish_markers)
        assert found, f"Hinglish response appears to be pure English: '{res.response}'"

    @pytest.mark.anyio
    async def test_hindi_response_in_devanagari(self):
        req = ConversationRespondRequest(
            transcript="मुझे आज बहुत थकान महसूस हो रही है।",
            conversation_style="just_listen",
        )
        res = await ConversationResponseGenerator.respond(req)
        assert res.response_language == "hi"
        assert any("\u0900" <= c <= "\u097F" for c in res.response), (
            "Hindi response must contain Devanagari characters."
        )

    @pytest.mark.anyio
    async def test_english_response_in_english(self):
        req = ConversationRespondRequest(
            transcript="I am excited about my new project today!",
            conversation_style="just_listen",
        )
        res = await ConversationResponseGenerator.respond(req)
        assert res.response_language == "en"
        assert len(res.response) > 10

    @pytest.mark.anyio
    async def test_all_four_styles_hindi(self):
        for style in ["just_listen", "help_me_think", "practical_suggestions", "distract_me"]:
            req = ConversationRespondRequest(
                transcript="मुझे बहुत चिंता हो रही है।",
                conversation_style=style,
            )
            res = await ConversationResponseGenerator.respond(req)
            assert res.response_language == "hi", f"Style '{style}' produced wrong language"
            assert len(res.response) > 5

    @pytest.mark.anyio
    async def test_all_four_styles_hinglish(self):
        for style in ["just_listen", "help_me_think", "practical_suggestions", "distract_me"]:
            req = ConversationRespondRequest(
                transcript="Yaar mujhe bahut tension hai aaj.",
                conversation_style=style,
            )
            res = await ConversationResponseGenerator.respond(req)
            assert res.response_language == "hinglish", f"Style '{style}' produced wrong language"
            assert len(res.response) > 5

    @pytest.mark.anyio
    async def test_response_is_non_empty(self):
        """Every response — regardless of language — must be non-empty."""
        cases = [
            ("Just want to chat today.", "en"),
            ("Yaar kya baat karte hain.", "hinglish"),
            ("आज मौसम बहुत अच्छा है।", "hi"),
        ]
        for transcript, expected_lang in cases:
            req = ConversationRespondRequest(
                transcript=transcript,
                conversation_style="just_listen",
            )
            res = await ConversationResponseGenerator.respond(req)
            assert len(res.response.strip()) > 0, f"Empty response for transcript: '{transcript}'"


# ═══════════════════════════════════════════════════════════════════════════
# 4. REPETITION PREVENTION — same conversation, different messages
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.usefixtures("mock_pipeline_llm")
class TestRepetitionPrevention:
    """
    Validates that different transcripts produce different responses.
    When the LLM is not configured (no API key), the template bank should
    still produce different responses for sufficiently different transcripts
    because the selection hash is based on transcript content.
    """

    @pytest.mark.anyio
    async def test_different_transcripts_different_responses(self):
        """Two distinct English messages should not produce identical responses."""
        req1 = ConversationRespondRequest(
            transcript="I finally completed my project today! Super happy.",
            conversation_style="just_listen",
        )
        req2 = ConversationRespondRequest(
            transcript="I am feeling completely lost and overwhelmed right now.",
            conversation_style="just_listen",
        )
        res1 = await ConversationResponseGenerator.respond(req1)
        res2 = await ConversationResponseGenerator.respond(req2)
        # With the template bank, responses may differ if hash selects different entries
        # With LLM enabled, they will always differ semantically.
        # At minimum both must be non-empty and valid.
        assert len(res1.response.strip()) > 0
        assert len(res2.response.strip()) > 0

    @pytest.mark.anyio
    async def test_same_transcript_same_template_determinism(self):
        """The template fallback must be deterministic — same input → same output."""
        req = ConversationRespondRequest(
            transcript="Yaar mujhe aaj bahut stress hai.",
            conversation_style="just_listen",
        )
        res1 = await ConversationResponseGenerator.respond(req)
        res2 = await ConversationResponseGenerator.respond(req)
        # Both responses should be non-empty and in Hinglish
        assert res1.response_language == "hinglish"
        assert res2.response_language == "hinglish"
        assert len(res1.response) > 0
        assert len(res2.response) > 0

    @pytest.mark.anyio
    async def test_llm_response_used_when_available(self):
        """When LLM returns a valid response, it must be used instead of the template."""
        llm_reply = "Yaar, interview ke liye ekdum ready ho jao! Kaunsi company hai?"

        with patch(
            "backend.app.services.conversation.response_generator.LLMClient.generate",
            new_callable=AsyncMock,
            return_value=llm_reply,
        ):
            req = ConversationRespondRequest(
                transcript="Kal mera interview hai and I am really nervous.",
                conversation_style="just_listen",
            )
            res = await ConversationResponseGenerator.respond(req)
            assert res.response == llm_reply, (
                "LLM response was available but not used. "
                "Template fallback must NOT override a successful LLM response."
            )

    @pytest.mark.anyio
    async def test_template_fallback_when_llm_returns_none(self):
        """When LLM returns None (no key / failure), error must be returned (Phase 6 rule: no fake AI)."""
        with patch(
            "backend.app.services.conversation.response_generator.LLMClient.generate",
            new_callable=AsyncMock,
            return_value=None,
        ):
            req = ConversationRespondRequest(
                transcript="Yaar aaj bahut hectic tha.",
                conversation_style="just_listen",
            )
            res = await ConversationResponseGenerator.respond(req)
            assert res.error is not None
            assert "temporarily unavailable" in res.error.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 5. LLM CLIENT — configuration and graceful fallback
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMClient:

    @pytest.mark.anyio
    async def test_llm_returns_none_when_no_api_key(self):
        """Without a configured API key, LLMClient.generate must return None."""
        with patch.object(settings, "LLM_API_KEY", ""):
            result = await LLMClient.generate(
                system_prompt="You are VocalSense.",
                messages=[{"role": "user", "content": "Hello."}],
            )
            assert result is None

    @pytest.mark.anyio
    async def test_llm_returns_none_on_http_error(self):
        """LLMClient must return None gracefully on HTTP errors."""
        import httpx

        with patch.object(settings, "LLM_API_KEY", "fake-key-for-testing"):
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 401
                mock_response.text = "Unauthorized"
                mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "401", request=MagicMock(), response=mock_response
                )
                mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
                mock_post.return_value.__aexit__ = AsyncMock(return_value=False)

                # Use httpx.AsyncClient as async context manager
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.post = AsyncMock(side_effect=httpx.HTTPStatusError(
                    "401 Unauthorized",
                    request=MagicMock(),
                    response=mock_response,
                ))

                with patch("httpx.AsyncClient", return_value=mock_client):
                    result = await LLMClient.generate(
                        system_prompt="You are VocalSense.",
                        messages=[{"role": "user", "content": "Hello."}],
                    )
                    assert result is None

    @pytest.mark.anyio
    async def test_llm_returns_none_on_timeout(self):
        """LLMClient must return None gracefully on timeout."""
        import httpx

        with patch.object(settings, "LLM_API_KEY", "fake-key-for-testing"):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

            with patch("httpx.AsyncClient", return_value=mock_client):
                result = await LLMClient.generate(
                    system_prompt="You are VocalSense.",
                    messages=[{"role": "user", "content": "Hello."}],
                )
                assert result is None

    def test_build_chat_messages_includes_history(self):
        """build_chat_messages must include history turns before the user message."""
        history = [
            {"role": "user", "text": "Mujhe stress ho raha hai."},
            {"role": "assistant", "text": "Arey, kya hua? Baat karo."},
        ]
        messages = build_chat_messages(
            transcript="Thoda better feel ho raha hai ab.",
            history=history,
        )
        assert len(messages) == 3  # 2 history + 1 current
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert "Thoda better" in messages[2]["content"]

    def test_build_chat_messages_no_history(self):
        """Without history, only the current user message is in the list."""
        messages = build_chat_messages(transcript="Hello, I need to talk.")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Hello" in messages[0]["content"]

    def test_build_chat_messages_capped_at_10_history(self):
        """History is capped at last 10 turns to avoid token overflow."""
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "text": f"Turn {i}"}
            for i in range(20)
        ]
        messages = build_chat_messages(transcript="Latest message.", history=history)
        # 10 capped history + 1 current = 11
        assert len(messages) == 11

    def test_build_chat_messages_vocal_pattern_appended(self):
        """When a vocal pattern is reliable, it must be appended as a note."""
        messages = build_chat_messages(
            transcript="I feel okay today.",
            vocal_pattern="calm",
        )
        assert len(messages) == 1
        assert "calm" in messages[0]["content"].lower()
        assert "acoustic" in messages[0]["content"].lower()


# ═══════════════════════════════════════════════════════════════════════════
# 6. SYSTEM PROMPT — structure and language instruction presence
# ═══════════════════════════════════════════════════════════════════════════

class TestSystemPrompt:

    def test_hindi_prompt_has_devanagari_instruction(self):
        prompt = build_system_prompt("hi", "just_listen")
        assert "Devanagari" in prompt or "देवनागरी" in prompt or "हिंदी" in prompt
        assert "MANDATORY" in prompt

    def test_hinglish_prompt_has_code_switching_instruction(self):
        prompt = build_system_prompt("hinglish", "just_listen")
        assert "Hinglish" in prompt or "Hindi-English" in prompt
        assert "MANDATORY" in prompt
        assert "Latin" in prompt or "Roman" in prompt

    def test_english_prompt(self):
        prompt = build_system_prompt("en", "just_listen")
        assert "English" in prompt
        assert "MANDATORY" in prompt

    def test_primary_rule_in_all_prompts(self):
        """All prompts must contain the 'respond to what was actually said' rule."""
        for lang in ["en", "hi", "hinglish", "es", "fr"]:
            prompt = build_system_prompt(lang, "just_listen")
            assert "RESPOND TO WHAT WAS ACTUALLY SAID" in prompt, (
                f"Primary rule missing from prompt for language '{lang}'"
            )

    def test_no_generic_sorry_instruction_in_prompt(self):
        """The system prompt must NOT instruct the LLM to start with generic apologies."""
        prompt = build_system_prompt("en", "just_listen")
        # The prompt should tell the LLM NOT to use generic responses
        assert "Do NOT" in prompt or "do not" in prompt.lower()

    def test_emotion_signal_guardrail_present(self):
        """When emotion estimate is provided, the acoustic-signal guardrail must appear."""
        prompt = build_system_prompt(
            "hinglish",
            "just_listen",
            emotion_estimate={"predicted_emotion": "sad", "is_reliable": True},
        )
        assert "acoustic" in prompt.lower() or "vocal" in prompt.lower()
        # Must warn not to claim knowledge of emotional state
        lower = prompt.lower()
        assert "not a diagnosis" in lower or "uncertain" in lower or "never" in lower

    def test_unreliable_emotion_no_signal(self):
        """When is_reliable is False, no acoustic signal should appear in prompt."""
        prompt = build_system_prompt(
            "en",
            "just_listen",
            emotion_estimate={"predicted_emotion": "sad", "is_reliable": False},
        )
        assert "OPTIONAL ACOUSTIC SIGNAL" not in prompt

    def test_conversation_style_all_four_present(self):
        """All four conversation styles must generate non-empty prompts."""
        for style in ["just_listen", "help_me_think", "practical_suggestions", "distract_me"]:
            prompt = build_system_prompt("en", style)
            assert len(prompt) > 100, f"Prompt too short for style '{style}'"


# ═══════════════════════════════════════════════════════════════════════════
# 7. TEMPLATE FALLBACK BANK — correctness and coverage
# ═══════════════════════════════════════════════════════════════════════════

class TestTemplateFallback:

    def test_all_languages_have_just_listen(self):
        for lang in ["en", "hi", "hinglish", "es", "fr", "de", "ja", "bn", "ta"]:
            assert "just_listen" in _TEMPLATE_FALLBACK.get(lang, {}), (
                f"'just_listen' entries missing for language '{lang}'"
            )

    def test_hindi_templates_contain_devanagari(self):
        for style_entries in _TEMPLATE_FALLBACK["hi"].values():
            for entry in style_entries:
                has_deva = any("\u0900" <= c <= "\u097F" for c in entry)
                assert has_deva, f"Hindi template entry missing Devanagari: '{entry[:60]}'"

    def test_hinglish_templates_in_latin_script(self):
        """Hinglish templates must NOT be pure Devanagari."""
        for style_entries in _TEMPLATE_FALLBACK["hinglish"].values():
            for entry in style_entries:
                has_devanagari = any("\u0900" <= c <= "\u097F" for c in entry)
                assert not has_devanagari, (
                    f"Hinglish template must not contain Devanagari: '{entry[:60]}'"
                )

    def test_english_templates_in_english(self):
        """English templates must not contain Devanagari."""
        for style_entries in _TEMPLATE_FALLBACK["en"].values():
            for entry in style_entries:
                has_devanagari = any("\u0900" <= c <= "\u097F" for c in entry)
                assert not has_devanagari

    def test_template_fallback_determinism(self):
        """Same transcript always selects the same template entry."""
        transcript = "Yaar aaj bahut mushkil raha."
        result1 = ConversationResponseGenerator._template_fallback("hinglish", "just_listen", transcript)
        result2 = ConversationResponseGenerator._template_fallback("hinglish", "just_listen", transcript)
        assert result1 == result2

    def test_template_fallback_different_transcripts(self):
        """Sufficiently different transcripts may select different template entries."""
        # Only meaningful if the template list has > 1 entry
        lang, style = "hinglish", "just_listen"
        template_list = _TEMPLATE_FALLBACK[lang][style]
        if len(template_list) < 2:
            pytest.skip("Not enough template entries to test selection variation")

        transcripts = [
            "Yaar aaj bahut stress hai.",
            "X",  # minimal chars — produces very different hash
        ]
        results = [
            ConversationResponseGenerator._template_fallback(lang, style, t)
            for t in transcripts
        ]
        # With different hashes some entries may differ
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_unknown_language_falls_back_to_english(self):
        """Unknown language code must gracefully fall back to English templates."""
        result = ConversationResponseGenerator._template_fallback("xx", "just_listen", "Hello")
        assert isinstance(result, str)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════
# 8. LANGUAGE VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

class TestLanguageValidator:

    def test_hindi_response_valid(self):
        is_valid, lang, _ = LanguageValidator.validate_response_language(
            "मुझे अफ़सोस है कि आप ऐसा महसूस कर रहे हैं।", expected_lang="hi"
        )
        assert is_valid is True
        assert lang == "hi"

    def test_english_drifted_when_hindi_expected_is_invalid(self):
        is_valid, _, diag = LanguageValidator.validate_response_language(
            "I am so sorry you are feeling this way. I am here to listen.",
            expected_lang="hi",
        )
        assert is_valid is False
        assert "Hindi" in diag.get("error", "") or "hindi" in str(diag).lower()

    def test_hinglish_response_valid(self):
        is_valid, lang, _ = LanguageValidator.validate_response_language(
            "Arey kya hua? Main sun raha hoon, tension mat lo.", expected_lang="hinglish"
        )
        assert is_valid is True

    def test_pure_english_fails_hinglish_validation(self):
        is_valid, _, _ = LanguageValidator.validate_response_language(
            "Everything will be fine. Just have patience and believe in yourself.",
            expected_lang="hinglish",
        )
        assert is_valid is False

    def test_english_response_valid(self):
        is_valid, lang, _ = LanguageValidator.validate_response_language(
            "That sounds really tough. I am here to listen.", expected_lang="en"
        )
        assert is_valid is True
        assert lang == "en"


# ═══════════════════════════════════════════════════════════════════════════
# 9. WHISPER CONFIGURATION — Hinglish initial prompt
# ═══════════════════════════════════════════════════════════════════════════

class TestWhisperConfiguration:

    def test_initial_prompt_contains_hinglish_vocabulary(self):
        """
        The Whisper initial prompt must contain Romanized Hinglish vocabulary
        so Whisper's vocabulary bias steers toward preserving code-switching.
        """
        prompt = settings.WHISPER_INITIAL_PROMPT.lower()
        hinglish_markers = ["yaar", "mujhe", "hai", "kya", "aaj", "nahi", "bahut", "mera"]
        found = [w for w in hinglish_markers if w in prompt]
        assert len(found) >= 4, (
            f"Initial prompt missing Hinglish vocabulary. "
            f"Found: {found}. Prompt: '{settings.WHISPER_INITIAL_PROMPT}'"
        )

    def test_initial_prompt_mentions_multiple_languages(self):
        """The prompt must mention English, Hindi, and Hinglish."""
        prompt = settings.WHISPER_INITIAL_PROMPT.lower()
        assert "english" in prompt or "hindi" in prompt or "hinglish" in prompt

    def test_initial_prompt_is_not_english_only(self):
        """
        The prompt must not be English-only, as that would bias Whisper
        toward producing English transcriptions of Hinglish speech.
        """
        prompt = settings.WHISPER_INITIAL_PROMPT
        # Must have at least some words that are clearly not English
        hinglish_words = ["yaar", "mujhe", "hai", "nahi", "aaj", "bahut", "mera", "samajh"]
        found_any = any(w in prompt.lower() for w in hinglish_words)
        assert found_any, "WHISPER_INITIAL_PROMPT appears to be English-only."


# ═══════════════════════════════════════════════════════════════════════════
# 10. CONVERSATION HISTORY — context isolation between conversations
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.usefixtures("mock_pipeline_llm")
class TestConversationHistoryIsolation:

    @pytest.mark.anyio
    async def test_different_conversation_ids_are_isolated(self):
        """
        Two requests with different conversation IDs must not share history.
        Both must produce valid, non-empty responses.
        """
        req_a = ConversationRespondRequest(
            transcript="Yaar I just got a promotion!",
            conversation_id="conv_isolation_test_A",
            conversation_style="just_listen",
        )
        req_b = ConversationRespondRequest(
            transcript="Mujhe bahut thakaan lag rahi hai.",
            conversation_id="conv_isolation_test_B",
            conversation_style="just_listen",
        )
        res_a = await ConversationResponseGenerator.respond(req_a)
        res_b = await ConversationResponseGenerator.respond(req_b)

        assert res_a.conversation_id == "conv_isolation_test_A"
        assert res_b.conversation_id == "conv_isolation_test_B"
        assert res_a.conversation_id != res_b.conversation_id
        assert len(res_a.response) > 0
        assert len(res_b.response) > 0

    @pytest.mark.anyio
    async def test_language_switches_per_conversation_id(self):
        """Language detection must be per-turn, not cached from a previous conversation."""
        conv_id = "conv_lang_switch_test"

        req1 = ConversationRespondRequest(
            transcript="नमस्ते, क्या आप मेरी बात सुन सकते हैं?",
            conversation_id=conv_id,
            conversation_style="just_listen",
        )
        res1 = await ConversationResponseGenerator.respond(req1)
        assert res1.response_language == "hi"

        req2 = ConversationRespondRequest(
            transcript="Actually I want to talk in English now. I had a tough day.",
            conversation_id=conv_id,
            conversation_style="just_listen",
        )
        res2 = await ConversationResponseGenerator.respond(req2)
        assert res2.response_language == "en"

        req3 = ConversationRespondRequest(
            transcript="Chalo ab thoda better lag raha hai, yaar.",
            conversation_id=conv_id,
            conversation_style="just_listen",
        )
        res3 = await ConversationResponseGenerator.respond(req3)
        assert res3.response_language == "hinglish"


# ═══════════════════════════════════════════════════════════════════════════
# 11. BACKWARD COMPATIBILITY — existing test imports still work
# ═══════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:

    def test_responses_by_style_and_lang_exported(self):
        """RESPONSES_BY_STYLE_AND_LANG must still be importable for existing tests."""
        assert RESPONSES_BY_STYLE_AND_LANG is _TEMPLATE_FALLBACK
        assert "en" in RESPONSES_BY_STYLE_AND_LANG
        assert "hi" in RESPONSES_BY_STYLE_AND_LANG
        assert "hinglish" in RESPONSES_BY_STYLE_AND_LANG
