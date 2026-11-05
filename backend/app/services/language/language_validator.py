import re
from typing import Tuple, Dict, Any
from backend.app.services.language.language_detector import (
    LanguageDetector,
    DEVANAGARI_REGEX,
    LATIN_REGEX,
    BENGALI_REGEX,
    TAMIL_REGEX,
    JAPANESE_REGEX,
    HINGLISH_VOCABULARY,
    SPANISH_MARKERS,
    FRENCH_MARKERS,
    GERMAN_MARKERS,
)


class LanguageValidator:
    """
    Validates whether a generated companion response adheres to the expected
    user language, preventing silent English drift when the user is speaking
    Hindi, Hinglish, Spanish, French, etc.
    """

    @classmethod
    def validate_response_language(
        cls,
        response_text: str,
        expected_lang: str,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates the generated response text against the expected language.

        Returns:
            Tuple[bool, str, Dict[str, Any]]:
                - is_valid: True if response language matches expected language
                - detected_response_lang: Detected language of the response
                - diagnostics: Metadata detailing validation reasoning
        """
        clean_text = response_text.strip()
        if not clean_text:
            return False, "empty", {"reason": "empty_response"}

        norm_expected = expected_lang.lower()

        has_devanagari = bool(DEVANAGARI_REGEX.search(clean_text))
        has_latin = bool(LATIN_REGEX.search(clean_text))
        has_bengali = bool(BENGALI_REGEX.search(clean_text))
        has_tamil = bool(TAMIL_REGEX.search(clean_text))
        has_japanese = bool(JAPANESE_REGEX.search(clean_text))

        words = set(re.findall(r"\b[a-zA-Z]+\b", clean_text.lower()))
        matching_hinglish = words.intersection(HINGLISH_VOCABULARY)

        # 1. Hindi expectation
        if norm_expected in ["hi", "hindi"]:
            if has_devanagari:
                return True, "hi", {"devanagari_present": True}
            # If expected Hindi but response is entirely English Latin with no Hindi
            if has_latin and not has_devanagari and len(matching_hinglish) < 2:
                return False, "en", {"error": "Expected Hindi in Devanagari, got English text"}
            # If Romanized Hindi was produced
            if len(matching_hinglish) >= 2:
                return True, "hinglish", {"note": "Romanized Hindi accepted for Hindi"}
            return False, "unknown", {"error": "Missing Devanagari script for Hindi response"}

        # 2. Hinglish expectation (mixed code-switching)
        if norm_expected in ["hinglish"]:
            # Hinglish can be Romanized Hindi + English words, or Devanagari + English words
            if (has_devanagari and has_latin) or len(matching_hinglish) >= 1:
                return True, "hinglish", {"hinglish_markers": list(matching_hinglish)}
            if has_devanagari and not has_latin:
                # Accept pure Hindi as valid companion response for Hinglish
                return True, "hi", {"note": "Devanagari accepted for Hinglish"}
            # If purely English without any Hindi markers or phrasing
            if len(matching_hinglish) == 0 and has_latin:
                return False, "en", {"error": "Expected Hinglish with code-switching, got pure English"}
            return True, "hinglish", {}

        # 3. Bengali expectation
        if norm_expected in ["bn", "bengali"]:
            if has_bengali:
                return True, "bn", {"bengali_script": True}
            return False, "non_bengali", {"error": "Expected Bengali script"}

        # 4. Tamil expectation
        if norm_expected in ["ta", "tamil"]:
            if has_tamil:
                return True, "ta", {"tamil_script": True}
            return False, "non_tamil", {"error": "Expected Tamil script"}

        # 5. Japanese expectation
        if norm_expected in ["ja", "japanese"]:
            if has_japanese:
                return True, "ja", {"japanese_script": True}
            return False, "non_japanese", {"error": "Expected Japanese characters"}

        # 6. Spanish expectation
        if norm_expected in ["es", "spanish"]:
            spanish_matches = words.intersection(SPANISH_MARKERS)
            if len(spanish_matches) >= 2 or any(c in clean_text for c in "áéíóúñ¿¡"):
                return True, "es", {"spanish_markers": list(spanish_matches)}
            return False, "non_spanish", {"error": "Expected Spanish response"}

        # 7. French expectation
        if norm_expected in ["fr", "french"]:
            french_matches = words.intersection(FRENCH_MARKERS)
            if len(french_matches) >= 2 or any(c in clean_text for c in "éèêàçùœ"):
                return True, "fr", {"french_markers": list(french_matches)}
            return False, "non_french", {"error": "Expected French response"}

        # 8. German expectation
        if norm_expected in ["de", "german"]:
            german_matches = words.intersection(GERMAN_MARKERS)
            if len(german_matches) >= 2 or any(c in clean_text for c in "äöüß"):
                return True, "de", {"german_markers": list(german_matches)}
            return False, "non_german", {"error": "Expected German response"}

        # 9. English expectation
        if norm_expected in ["en", "english"]:
            # If response contains non-Latin scripts, it failed
            if has_devanagari or has_bengali or has_tamil or has_japanese:
                return False, "non_english", {"error": "Expected English, got non-Latin script"}
            return True, "en", {}

        # Fallback: assume valid if no specific language rule is broken
        return True, norm_expected, {"fallback_validation": True}
