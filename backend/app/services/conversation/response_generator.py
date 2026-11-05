"""
Multilingual, emotion-aware conversational response generator for VocalSense.

Architecture:
1. Detect language from transcript + Whisper acoustic hints.
2. Build a rich, context-aware system prompt (language instruction + style + emotion signal).
3. Attempt LLM generation with the user's actual transcript and MongoDB conversation history.
4. If LLM is unconfigured or fails, fall back to the language-matched template bank.
5. Validate that the response is in the expected language; if not, apply template fallback.
6. Optionally synthesize TTS audio.
7. Persist user message, assistant message, and emotion insight to MongoDB.

Root cause of the previous bugs:
- Bug 1 (repetitive responses): _generate_text() was a pure hash → template picker that
  completely ignored the transcript content and never called any LLM.
- Bug 2 (English translation): The initial Whisper prompt was too sparse for Whisper-base
  to reliably preserve Hinglish code-switching; fixed in config.py.
"""

import uuid
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from backend.app.schemas.conversation import (
    ConversationRespondRequest,
    ConversationRespondResponse,
    ConversationTurn,
)
from backend.app.services.language.language_detector import LanguageDetector, LanguageDetectionResult
from backend.app.services.language.language_validator import LanguageValidator
from backend.app.services.conversation.language_prompt import build_system_prompt
from backend.app.services.conversation.llm_client import LLMClient, build_chat_messages
from backend.app.services.tts.tts_service import TTSService
from backend.app.db.mongodb import mongodb_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template fallback bank (used when LLM is unconfigured or returns an error)
# These are the ONLY place hardcoded text lives. They serve as a safety net,
# NOT as the primary response path. The LLM generates real responses when
# LLM_API_KEY is configured.
# ---------------------------------------------------------------------------
_TEMPLATE_FALLBACK: Dict[str, Dict[str, List[str]]] = {
    "hi": {
        "just_listen": [
            "मुझे अफ़सोस है कि आज आप ऐसा महसूस कर रहे हैं। अगर आप चाहें, तो मुझे विस्तार से बता सकते हैं कि क्या हुआ। मैं पूरी तरह से सुन रहा हूँ।",
            "आपकी बात सुनकर समझ आ रहा है कि यह समय आपके लिए आसान नहीं है। मैं यहीं हूँ, आप जितना साझा करना चाहें, कर सकते हैं।",
            "कभी-कभी बस किसी का सुनना ही बहुत मायने रखता है। आप अपने विचार बिना किसी झिझक के रख सकते हैं, मैं सुन रहा हूँ।",
        ],
        "help_me_think": [
            "यह स्थिति सच में सोचने पर मजबूर करने वाली लगती है। इस पूरी बात में सबसे ज़्यादा परेशानी आपको किस हिस्से से महसूस हो रही है?",
            "आइए इसे मिलकर समझने की कोशिश करते हैं। अगर आप पीछे मुड़कर देखें, तो क्या आपको कोई ऐसा रास्ता दिखता है जिसे पहले आज़माया जा सकता था?",
        ],
        "practical_suggestions": [
            "इस समय सब कुछ एक साथ सुलझाने के बजाय, क्या आप सबसे पहले बस एक छोटा सा कदम उठा सकते हैं? थोड़ा पानी पीजिए और पाँच मिनट का गहरा ब्रेक लीजिए।",
            "एक व्यावहारिक तरीका यह हो सकता है कि आज की सबसे ज़रूरी एक चीज़ चुन लें, और बाकी चीज़ों को कल के लिए टाल दें।",
        ],
        "distract_me": [
            "आइए थोड़ी देर के लिए दिमाग को हल्का करते हैं। क्या आप जानते हैं कि अंतरिक्ष में पूरी तरह से सन्नाटा होता है क्योंकि वहाँ ध्वनि ले जाने के लिए कोई हवा नहीं है? क्या आप किसी पसंदीदा संगीत या जगह के बारे में बात करना चाहेंगे?",
            "तनाव से ध्यान हटाने के लिए, मुझे अपनी किसी ऐसी यात्रा या याद के बारे में बताइए जिसने आपको बहुत शांति दी थी।",
        ],
    },
    "hinglish": {
        "just_listen": [
            "Arey, main samajh sakta hoon ki aaj ka din challenging raha hai. Agar comfortable ho toh share karo, main sun raha hoon.",
            "I hear you, yaar. Kabhi kabhi bas kisi ko dil ki baat bata dena zaroori hota hai. Take your time, main yahin hoon.",
            "Aapki baat sunkar lagta hai kaafi stress ho gaya hai. Feel free to vent out whatever is on your mind.",
        ],
        "help_me_think": [
            "Yeh situation thodi complicated lag rahi hai. Is poore matter mein sabse zyada tension kis cheez ki hai?",
            "Chalo step by step sochte hain. Agar options dekhein, toh sabse pehla move kya ho sakta hai?",
        ],
        "practical_suggestions": [
            "Pehle toh ek deep breath lo aur thoda paani piyo. Saari problem ek saath solve karne ki zaroorat nahi hai, pehle ek chota sa step lete hain.",
            "Mera suggestion hai ki aaj ke liye bas priority task complete karo, baaki cheezein fresh dimag se kal dekhenge.",
        ],
        "distract_me": [
            "Chalo thoda mind distract karte hain! Fun fact: Sea otters sote waqt ek doosre ka haath pakad ke sote hain taaki beh na jaayein. Tumhara favourite comfort show kaunsa hai?",
            "Stress bhoolne ke liye, mujhe batao tumhara all-time favourite song ya street food kya hai?",
        ],
    },
    "en": {
        "just_listen": [
            "I'm really sorry you're feeling this way today. If you'd like, you can tell me what's been going on. I'm here to listen.",
            "That sounds like a heavy burden to carry. Take all the time you need; I'm listening with an open mind.",
            "Sometimes having someone just be there and listen makes all the difference. I'm right here with you.",
        ],
        "help_me_think": [
            "It sounds like there are several moving parts here. What feels like the most pressing piece of this puzzle for you right now?",
            "Let's look at this together. When you think about the ideal outcome, what does the first realistic step look like?",
        ],
        "practical_suggestions": [
            "Rather than tackling everything at once, try focusing on just one micro-step today. Start by taking a deep breath and a short 5-minute break.",
            "A practical approach could be writing down just two things within your control today, and letting the rest wait until tomorrow.",
        ],
        "distract_me": [
            "Let's give your mind a gentle break. Did you know that honey never truly spoils? Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still edible. What is your go-to comfort book or movie?",
            "To take your mind off things, what is one place in the world that brings you total peace whenever you picture it?",
        ],
    },
    "es": {
        "just_listen": [
            "Siento mucho que te sientas así hoy. Si quieres, puedes contarme lo que te preocupa. Estoy aquí para escucharte.",
            "Parece que estás pasando por un momento complicado. Tómate tu tiempo, te escucho con toda atención.",
        ],
        "help_me_think": [
            "Parece una situación compleja. ¿Qué parte de todo esto es la que más incertidumbre te genera?",
        ],
        "practical_suggestions": [
            "En lugar de resolverlo todo hoy, te sugiero dar un paso muy pequeño. Bebe un poco de agua y descansa unos minutos.",
        ],
        "distract_me": [
            "Tomemos un respiro. ¿Sabías que los pulpos tienen tres corazones? ¿Cuál es tu canción favorita para relajarte?",
        ],
    },
    "fr": {
        "just_listen": [
            "Je suis désolé que tu te sentes comme ça aujourd'hui. Si tu veux en parler, je suis là pour t'écouter sans aucun jugement.",
            "Cela semble difficile à porter. Prends ton temps, je suis présent et je t'écoute.",
        ],
        "help_me_think": [
            "C'est une situation qui demande réflexion. Selon toi, quel est l'élément le plus important à clarifier en premier ?",
        ],
        "practical_suggestions": [
            "Ne cherche pas à tout régler d'un coup. Fais une pause de cinq minutes et concentre-toi sur une seule petite chose aujourd'hui.",
        ],
        "distract_me": [
            "Changeons-nous un peu les idées. Savais-tu que les flamants roses dorment sur une seule patte ? Quel est ton film réconfortant préféré ?",
        ],
    },
    "de": {
        "just_listen": [
            "Es tut mir leid, dass du dich heute so fühlst. Wenn du möchtest, kannst du mir alles erzählen. Ich bin hier, um zuzuhören.",
        ],
        "help_me_think": [
            "Das wirkt wie eine vielschichtige Situation. Was bereitet dir dabei im Moment die größten Sorgen?",
        ],
        "practical_suggestions": [
            "Versuche nicht alles auf einmal zu lösen. Nimm dir heute nur einen einzigen kleinen Schritt vor.",
        ],
        "distract_me": [
            "Lass uns die Gedanken kurz ablenken. Was ist dein liebster Ort, um zur Ruhe zu kommen?",
        ],
    },
    "ja": {
        "just_listen": [
            "今日はそういった気持ちなんですね。もしよろしければ、何があったのか話してみてください。私はここで耳を傾けています。",
        ],
        "help_me_think": [
            "状況を整理してみましょう。いま一番気がかりな点はどの部分ですか？",
        ],
        "practical_suggestions": [
            "一度にすべてを抱え込まず、まずは温かいお茶を飲んで、少し深呼吸してみませんか。",
        ],
        "distract_me": [
            "少し気分転換をしましょう。あなたの好きなリラックスできる音楽や趣味について教えていただけますか？",
        ],
    },
    "bn": {
        "just_listen": [
            "আমি বুঝতে পারছি আজকের দিনটি আপনার জন্য সহজ নয়। আপনি চাইলে আমাকে মন খুলে বলতে পারেন, আমি শুনছি।",
        ],
        "help_me_think": [
            "আসুন বিষয়টি একসাথে ভেবে দেখি। এই পরিস্থিতির কোন অংশটি আপনাকে সবচেয়ে বেশি ভাবাচ্ছে?",
        ],
        "practical_suggestions": [
            "সবকিছু একসাথে ঠিক করতে যাবেন না। প্রথমে একটি ছোট পদক্ষেপে মনোযোগ দিন।",
        ],
        "distract_me": [
            "মনকে একটু শান্ত করা যাক। আপনার প্রিয় কোনো স্মৃতি বা গান সম্পর্কে কিছু বলবেন?",
        ],
    },
    "ta": {
        "just_listen": [
            "நீங்கள் இப்படி உணர்வது வருத்தமளிக்கிறது. உங்கள் மனதில் உள்ளதை நீங்கள் என்னிடம் பகிர்ந்து கொள்ளலாம், நான் கேட்கிறேன்.",
        ],
        "help_me_think": [
            "நிலைமையை யோசிப்போம். இதில் எந்த விஷயம் உங்களுக்கு அதிக மன அழுத்தத்தை தருகிறது?",
        ],
        "practical_suggestions": [
            "எல்லாவற்றையும் ஒரே நேரத்தில் செய்யாதீர்கள். முதலில் ஒரு சிறிய படியை மட்டும் எடுத்துக் கொள்ளுங்கள்.",
        ],
        "distract_me": [
            "மனதை அமைதிப்படுத்துவோம். உங்களுக்கு மிகவும் பிடித்த பாடல் அல்லது இடம் பற்றி கூற முடியுமா?",
        ],
    },
}

# Keep the old name so existing tests that import RESPONSES_BY_STYLE_AND_LANG still work
RESPONSES_BY_STYLE_AND_LANG = _TEMPLATE_FALLBACK


class ConversationResponseGenerator:
    """
    Multilingual, emotion-aware conversational response generator.

    Primary path  : LLM generation (dynamic, context-aware, language-matched).
    Fallback path : Language-matched template bank (static, used only on LLM failure).
    """

    @classmethod
    async def respond(
        cls, request: ConversationRespondRequest
    ) -> ConversationRespondResponse:
        # ── 1. Detect language ────────────────────────────────────────────────
        detected = LanguageDetector.detect(
            transcript=request.transcript,
            whisper_lang=request.detected_language,
            user_preference=request.language_preference,
        )
        target_lang = detected.language_code
        style = request.conversation_style or "just_listen"
        conv_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        user_id = request.user_id

        # ── 2. Retrieve MongoDB conversation history ───────────────────────────
        db_history: List[Dict[str, Any]] = []
        try:
            raw_msgs = await mongodb_manager.get_messages(
                conversation_id=conv_id,
                limit=20,  # last 10 turns (user + assistant pairs)
            )
            db_history = [
                {"role": m.get("role", "user"), "text": m.get("text", "")}
                for m in raw_msgs
                if m.get("text", "").strip()
            ]
        except Exception as db_err:
            logger.debug("Could not retrieve MongoDB history: %s", db_err)

        # Merge request-provided history with DB history (request history wins if both present)
        merged_history: List[Dict[str, Any]] = db_history
        if request.history:
            merged_history = [
                {"role": t.role, "text": t.text}
                for t in request.history
                if t.text.strip()
            ]

        # ── 3. Generate response via LLM (primary path) ───────────────────────
        vocal_pattern: Optional[str] = None
        if request.emotion_estimate and request.emotion_estimate.get("is_reliable"):
            vocal_pattern = (
                request.emotion_estimate.get("predicted_emotion")
                or request.emotion_estimate.get("emotion_label")
            )

        response_text: Optional[str] = await cls._generate_llm_response(
            transcript=request.transcript,
            target_lang=target_lang,
            style=style,
            history=merged_history,
            is_code_switched=detected.is_code_switched,
            vocal_pattern=vocal_pattern,
            emotion_estimate=request.emotion_estimate,
        )

        # ── 4. Fall back to template bank if LLM unavailable or failed ────────
        llm_used = response_text is not None
        if not llm_used:
            response_text = cls._template_fallback(target_lang, style, request.transcript)
            logger.debug(
                "LLM unavailable — using template fallback for lang='%s' style='%s'.",
                target_lang, style,
            )

        assert response_text  # invariant: always a non-empty string here

        # ── 5. Validate response language ─────────────────────────────────────
        is_valid, detected_res_lang, diag = LanguageValidator.validate_response_language(
            response_text=response_text,
            expected_lang=target_lang,
        )

        if not is_valid:
            logger.warning(
                "Response language validation failed — expected '%s', detected '%s'. "
                "LLM used: %s. Applying template fallback.",
                target_lang, detected_res_lang, llm_used,
            )
            response_text = cls._template_fallback(target_lang, style, request.transcript)
            is_valid = True

        # ── 6. TTS synthesis ──────────────────────────────────────────────────
        tts_base64 = None
        tts_available = False
        mime_type = None
        if request.synthesize_speech:
            success, audio_b64, tts_err = TTSService.synthesize_to_base64(
                text=response_text,
                language_code=target_lang,
            )
            if success and audio_b64:
                tts_available = True
                tts_base64 = audio_b64
                mime_type = "audio/mpeg"

        # ── 7. Persist to MongoDB ─────────────────────────────────────────────
        try:
            now = datetime.now(timezone.utc)
            await mongodb_manager.save_conversation({
                "conversation_id": conv_id,
                "user_id": user_id,
                "style": style,
                "language": target_lang,
                "updated_at": now,
            })

            user_msg_id = f"msg_{uuid.uuid4().hex[:12]}"
            await mongodb_manager.add_message({
                "message_id": user_msg_id,
                "conversation_id": conv_id,
                "role": "user",
                "text": request.transcript,
                "detected_language": target_lang,
                "timestamp": now,
                "metadata": {"is_code_switched": detected.is_code_switched},
            })

            asst_msg_id = f"msg_{uuid.uuid4().hex[:12]}"
            await mongodb_manager.add_message({
                "message_id": asst_msg_id,
                "conversation_id": conv_id,
                "role": "assistant",
                "text": response_text,
                "detected_language": target_lang,
                "timestamp": datetime.now(timezone.utc),
                "metadata": {"conversation_style": style, "llm_used": llm_used},
            })

            if request.emotion_estimate:
                await mongodb_manager.save_emotion_insight({
                    "insight_id": f"ins_{uuid.uuid4().hex[:12]}",
                    "conversation_id": conv_id,
                    "message_id": user_msg_id,
                    "predicted_vocal_pattern": request.emotion_estimate.get("predicted_emotion", "uncertain"),
                    "probabilities": request.emotion_estimate.get("probabilities", {}),
                    "model_name": request.emotion_estimate.get("model_name", "speech_emotion_classifier"),
                    "model_version": request.emotion_estimate.get("model_version", "1.0.0"),
                    "is_reliable": request.emotion_estimate.get("is_reliable", True),
                    "timestamp": now,
                })
        except Exception as db_exc:
            logger.warning("Database persistence warning: %s", db_exc)

        return ConversationRespondResponse(
            response=response_text,
            response_language=target_lang,
            language_code=target_lang,
            conversation_style=style,
            conversation_id=conv_id,
            user_id=user_id,
            tts_available=tts_available,
            tts_audio_base64=tts_base64,
            audio_mime_type=mime_type,
            validation_passed=is_valid,
            code_switching_preserved=detected.is_code_switched,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    @classmethod
    async def _generate_llm_response(
        cls,
        transcript: str,
        target_lang: str,
        style: str,
        history: List[Dict[str, Any]],
        is_code_switched: bool,
        vocal_pattern: Optional[str],
        emotion_estimate: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Build a rich system prompt and send the actual transcript + history to
        the configured LLM. Returns None when LLM is unavailable or fails.
        """
        system_prompt = build_system_prompt(
            target_language=target_lang,
            conversation_style=style,
            emotion_estimate=emotion_estimate,
        )

        chat_messages = build_chat_messages(
            transcript=transcript,
            history=history,
            is_code_switched=is_code_switched,
            vocal_pattern=vocal_pattern,
        )

        return await LLMClient.generate(
            system_prompt=system_prompt,
            messages=chat_messages,
        )

    @classmethod
    def _template_fallback(
        cls,
        target_lang: str,
        style: str,
        transcript: str,
    ) -> str:
        """
        Select a language-and-style-matched template entry.
        Uses transcript character-sum for deterministic (not random) selection
        so the same input always maps to the same template, which is predictable
        for testing.
        """
        lang_dict = _TEMPLATE_FALLBACK.get(target_lang, _TEMPLATE_FALLBACK["en"])
        style_list = lang_dict.get(style, lang_dict.get("just_listen", ["I am here to listen."]))
        h = sum(ord(c) for c in transcript)
        return style_list[h % len(style_list)]
