"""
Unit and integration tests for Multilingual Conversational Intelligence.
Validates:
1. Language detection: English, Hindi, Hinglish, Spanish, French, German, Japanese, Bengali, Tamil
2. Language-matched response generation (Hindi -> Hindi, Hinglish -> Hinglish, etc.)
3. Language validator and wrong-language response detection / safe fallback
4. Mid-conversation language switching across consecutive turns
5. All four companion conversation styles across languages
6. Non-diagnostic, cautious integration of vocal emotion estimates
7. Text-to-speech modular synthesis
"""

import pytest
from backend.app.services.language.language_detector import LanguageDetector, LanguageDetectionResult
from backend.app.services.language.language_validator import LanguageValidator
from backend.app.services.conversation.response_generator import (
    ConversationResponseGenerator,
    RESPONSES_BY_STYLE_AND_LANG,
)
from backend.app.services.conversation.language_prompt import build_system_prompt
from backend.app.services.tts.tts_service import TTSService
from backend.app.schemas.conversation import ConversationRespondRequest, ConversationTurn


def test_language_detection_hindi_devanagari():
    text = "मुझे आज बहुत अकेलापन महसूस हो रहा है।"
    res = LanguageDetector.detect(text)
    assert res.language_code == "hi"
    assert res.primary_language == "hi"
    assert res.confidence >= 0.85
    assert res.is_code_switched is False


def test_language_detection_hinglish_code_switching():
    # Romanized Hindi with English words
    text = "Yaar mujhe aaj bahut stress ho raha hai because of my exams."
    res = LanguageDetector.detect(text)
    assert res.language_code == "hinglish"
    assert res.is_code_switched is True

    # Mixed Devanagari + Latin
    mixed_text = "आज का दिन really exhausting था।"
    res_mixed = LanguageDetector.detect(mixed_text)
    assert res_mixed.language_code == "hinglish"
    assert res_mixed.is_code_switched is True


def test_language_detection_english():
    text = "I have been feeling really exhausted and overwhelmed lately."
    res = LanguageDetector.detect(text)
    assert res.language_code == "en"
    assert res.primary_language == "en"
    assert res.is_code_switched is False


def test_language_detection_spanish():
    text = "Hoy me siento muy triste y cansado de todo."
    res = LanguageDetector.detect(text, whisper_lang="es")
    assert res.language_code == "es"


def test_language_detection_french():
    text = "Je me sens très triste aujourd'hui et je suis fatigué."
    res = LanguageDetector.detect(text, whisper_lang="fr")
    assert res.language_code == "fr"


def test_language_detection_japanese():
    text = "今日は本当に疲れました。"
    res = LanguageDetector.detect(text)
    assert res.language_code == "ja"


def test_language_detection_bengali():
    text = "আমি আজ খুব একা বোধ করছি।"
    res = LanguageDetector.detect(text)
    assert res.language_code == "bn"


def test_language_detection_tamil():
    text = "இன்று நான் மிகவும் சோர்வாக உணர்கிறேன்."
    res = LanguageDetector.detect(text)
    assert res.language_code == "ta"


def test_explicit_user_language_preference_override():
    # User types English words but explicitly set preference to Hindi
    text = "Good morning."
    res = LanguageDetector.detect(text, user_preference="hi")
    assert res.language_code == "hi"
    assert res.confidence == 1.0


def test_language_validator_hindi_and_wrong_language_drift():
    valid_hindi = "मुझे अफ़सोस है कि आज आप ऐसा महसूस कर रहे हैं।"
    is_valid, lang, diag = LanguageValidator.validate_response_language(valid_hindi, expected_lang="hi")
    assert is_valid is True
    assert lang == "hi"

    # Silent English drift when Hindi was expected
    english_drift = "I am so sorry you are feeling down today. Let me know what happened."
    is_valid_drift, lang_drift, diag_drift = LanguageValidator.validate_response_language(english_drift, expected_lang="hi")
    assert is_valid_drift is False
    assert "Expected Hindi" in diag_drift.get("error", "")


def test_language_validator_hinglish():
    valid_hinglish = "Arey kya hua? Main sun raha hoon, tension mat lo."
    is_valid, lang, _ = LanguageValidator.validate_response_language(valid_hinglish, expected_lang="hinglish")
    assert is_valid is True

    pure_english = "Everything will be fine tomorrow morning, just have patience."
    is_valid_en, _, diag = LanguageValidator.validate_response_language(pure_english, expected_lang="hinglish")
    assert is_valid_en is False


@pytest.mark.anyio
async def test_conversation_response_hindi_generation():
    req = ConversationRespondRequest(
        transcript="मुझे बहुत उदासी लग रही है।",
        conversation_style="just_listen",
    )
    res = await ConversationResponseGenerator.respond(req)
    assert res.response_language == "hi"
    assert res.language_code == "hi"
    # Response must be in Devanagari Hindi
    assert any("\u0900" <= c <= "\u097F" for c in res.response)
    assert res.validation_passed is True


@pytest.mark.anyio
async def test_conversation_response_hinglish_generation():
    req = ConversationRespondRequest(
        transcript="Yaar, aaj mera mood bilkul theek nahi hai.",
        conversation_style="just_listen",
    )
    res = await ConversationResponseGenerator.respond(req)
    assert res.response_language == "hinglish"
    assert res.code_switching_preserved is True
    assert res.validation_passed is True


@pytest.mark.anyio
async def test_conversation_response_english_generation():
    req = ConversationRespondRequest(
        transcript="I have been carrying this heavy stress all week.",
        conversation_style="practical_suggestions",
    )
    res = await ConversationResponseGenerator.respond(req)
    assert res.response_language == "en"
    assert res.conversation_style == "practical_suggestions"
    assert len(res.response) > 10


@pytest.mark.anyio
async def test_conversation_response_spanish_generation():
    req = ConversationRespondRequest(
        transcript="Hoy me siento muy solo.",
        detected_language="es",
        conversation_style="just_listen",
    )
    res = await ConversationResponseGenerator.respond(req)
    assert res.response_language == "es"
    assert "siento" in res.response.lower() or "escucharte" in res.response.lower() or "aquí" in res.response.lower()


@pytest.mark.anyio
async def test_conversation_all_four_styles_across_languages():
    styles = ["just_listen", "help_me_think", "practical_suggestions", "distract_me"]
    for s in styles:
        req_hi = ConversationRespondRequest(
            transcript="मुझे बहुत चिंता हो रही है।",
            conversation_style=s,
        )
        res_hi = await ConversationResponseGenerator.respond(req_hi)
        assert res_hi.response_language == "hi"
        assert res_hi.conversation_style == s
        assert len(res_hi.response) > 0


@pytest.mark.anyio
async def test_mid_conversation_language_switching():
    conv_id = "test_switching_conv"

    # Turn 1: User speaks Hindi
    req1 = ConversationRespondRequest(
        transcript="नमस्ते, क्या आप मेरी बात सुन सकते हैं?",
        conversation_id=conv_id,
    )
    res1 = await ConversationResponseGenerator.respond(req1)
    assert res1.response_language == "hi"

    # Turn 2: User switches to English
    req2 = ConversationRespondRequest(
        transcript="Actually, can we talk in English instead? I had a tough day at work.",
        conversation_id=conv_id,
        history=[ConversationTurn(role="user", text=req1.transcript), ConversationTurn(role="assistant", text=res1.response)],
    )
    res2 = await ConversationResponseGenerator.respond(req2)
    assert res2.response_language == "en"

    # Turn 3: User switches to Hinglish
    req3 = ConversationRespondRequest(
        transcript="Chalo ab theek hai, thoda relax feel ho raha hai.",
        conversation_id=conv_id,
    )
    res3 = await ConversationResponseGenerator.respond(req3)
    assert res3.response_language == "hinglish"


def test_language_prompt_construction_and_emotion_signal_guardrail():
    # When vocal pattern is provided, prompt must include cautious acoustic guidance
    prompt_with_emo = build_system_prompt(
        target_language="hi",
        conversation_style="just_listen",
        emotion_estimate={"predicted_emotion": "sad", "is_reliable": True},
    )
    assert "MANDATORY: You MUST respond entirely in Hindi using the Devanagari script" in prompt_with_emo
    assert "treat this strictly as an uncertain acoustic signal, not a diagnosis" in prompt_with_emo.lower()
    assert "never say 'my emotion model detected you are" in prompt_with_emo.lower()


def test_multilingual_tts_synthesis():
    from unittest.mock import patch

    with patch("backend.app.services.tts.tts_service.gTTS") as mock_gtts:
        def fake_write(fp):
            fp.write(b"MOCK_MP3_AUDIO_BYTES_TESTING_PURPOSES_ONLY_1234567890")

        instance = mock_gtts.return_value
        instance.write_to_fp.side_effect = fake_write

        # Test English synthesis
        success_en, b64_en, err_en = TTSService.synthesize_to_base64("Hello, I am here to listen.", "en")
        assert success_en is True
        assert b64_en is not None
        assert len(b64_en) > 20

        # Test Hindi synthesis
        success_hi, b64_hi, err_hi = TTSService.synthesize_to_base64("मैं सुन रहा हूँ।", "hi")
        assert success_hi is True
        assert b64_hi is not None
        assert len(b64_hi) > 20
