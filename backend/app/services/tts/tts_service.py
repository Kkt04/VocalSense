import io
import base64
import logging
from typing import Optional, Tuple
from gtts import gTTS

logger = logging.getLogger(__name__)

# Map internal language codes to gTTS language codes
GTTS_LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "hinglish": "hi",  # Hinglish phonetics synthesize well with Hindi engine
    "es": "es",
    "fr": "fr",
    "de": "de",
    "ja": "ja",
    "bn": "bn",
    "ta": "ta",
}


class TTSService:
    """
    Multilingual Text-to-Speech synthesis service.
    Generates audio in the user's spoken language without external paid API keys.
    Returns base64-encoded audio for direct frontend playback.
    """

    @classmethod
    def synthesize_to_base64(
        cls,
        text: str,
        language_code: str = "en",
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Synthesizes speech for the provided text.

        Returns:
            Tuple[bool, Optional[str], Optional[str]]:
                - success: True if audio was generated
                - audio_base64: Base64 data string of the audio
                - error_message: None if successful, error message otherwise
        """
        if not text or not text.strip():
            return False, None, "Empty text provided for TTS"

        gtts_lang = GTTS_LANG_MAP.get(language_code.lower(), "en")

        try:
            tts = gTTS(text=text.strip(), lang=gtts_lang, slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            encoded = base64.b64encode(mp3_fp.read()).decode("utf-8")
            return True, encoded, None
        except Exception as exc:
            logger.warning(
                "TTS synthesis failed for lang '%s' (%s): %s",
                language_code,
                gtts_lang,
                exc,
            )
            # Safe fallback attempt with English if regional TTS service had an issue
            if gtts_lang != "en":
                try:
                    tts = gTTS(text=text.strip(), lang="en", slow=False)
                    mp3_fp = io.BytesIO()
                    tts.write_to_fp(mp3_fp)
                    mp3_fp.seek(0)
                    encoded = base64.b64encode(mp3_fp.read()).decode("utf-8")
                    return True, encoded, None
                except Exception as fallback_exc:
                    return False, None, f"TTS fallback failed: {fallback_exc}"
            return False, None, str(exc)
