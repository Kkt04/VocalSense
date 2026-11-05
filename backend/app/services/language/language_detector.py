import re
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# Unicode script regexes
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
BENGALI_REGEX = re.compile(r"[\u0980-\u09FF]")
TAMIL_REGEX = re.compile(r"[\u0B80-\u0BFF]")
JAPANESE_REGEX = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]")
LATIN_REGEX = re.compile(r"[a-zA-Z]")

# Romanized Hindi markers for Hinglish detection
HINGLISH_VOCABULARY = {
    "yaar", "mujhe", "mera", "meri", "mere", "hai", "hain", "karna", "karni", "karte",
    "nahi", "nahin", "accha", "achha", "theek", "thik", "kya", "kyun", "kaise",
    "toh", "aur", "lekin", "bhi", "bhai", "aap", "hum", "tum", "baat", "karo", "karenge",
    "soch", "lagta", "raha", "rahi", "rahe", "hoga", "hogi", "samajh", "pata", "chalo",
    "bilkul", "aaj", "ab", "kuch", "bohot", "bahut", "sirf", "zindagi", "tension", "dimag"
}

# Distinctive word sets for European languages when Whisper is ambiguous
SPANISH_MARKERS = {
    "hola", "gracias", "por", "favor", "bueno", "días", "tardes", "amigo", "siento",
    "hoy", "muy", "solo", "estoy", "estás", "tengo", "quiero", "vida", "tiempo",
    "nada", "como", "pero", "para", "porque", "puedes", "decirme", "triste"
}

FRENCH_MARKERS = {
    "bonjour", "merci", "oui", "non", "triste", "aujourd'hui", "suis", "désolé",
    "sens", "veux", "parler", "écoute", "pourquoi", "très", "avec", "tout", "faire",
    "comment", "jamais", "peux", "aide", "besoin"
}

GERMAN_MARKERS = {
    "hallo", "danke", "bitte", "guten", "tag", "wie", "geht", "traurig", "fühle",
    "mich", "alleine", "heute", "nicht", "zeit", "leben", "kann", "helfen", "höre"
}

LANGUAGE_CODE_MAP = {
    "english": "en",
    "en": "en",
    "hindi": "hi",
    "hi": "hi",
    "hinglish": "hinglish",
    "spanish": "es",
    "es": "es",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
    "japanese": "ja",
    "ja": "ja",
    "bengali": "bn",
    "bn": "bn",
    "tamil": "ta",
    "ta": "ta",
}


class LanguageDetectionResult(BaseModel):
    primary_language: str
    language_code: str
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    is_code_switched: bool = False
    is_uncertain: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class LanguageDetector:
    """
    Multilingual and code-switching detector.
    Analyzes Whisper acoustic detection probabilities together with transcript
    Unicode scripts and lexical patterns to reliably classify English, Hindi,
    Hinglish, Spanish, French, German, Japanese, Bengali, and Tamil.
    """

    @classmethod
    def detect(
        cls,
        transcript: str,
        whisper_lang: Optional[str] = None,
        whisper_prob: Optional[float] = None,
        user_preference: Optional[str] = None,
    ) -> LanguageDetectionResult:
        # If user specified an explicit valid override, respect it
        if user_preference and user_preference.lower() in LANGUAGE_CODE_MAP:
            pref_code = LANGUAGE_CODE_MAP[user_preference.lower()]
            return LanguageDetectionResult(
                primary_language=pref_code,
                language_code=pref_code,
                confidence=1.0,
                is_code_switched=(pref_code == "hinglish"),
                is_uncertain=False,
                details={"source": "explicit_user_preference"},
            )

        clean_text = transcript.strip()
        if not clean_text:
            return LanguageDetectionResult(
                primary_language="en",
                language_code="en",
                confidence=0.5,
                is_code_switched=False,
                is_uncertain=True,
                details={"reason": "empty_transcript"},
            )

        # 1. Script checks (Direct Unicode ranges)
        has_devanagari = bool(DEVANAGARI_REGEX.search(clean_text))
        has_latin = bool(LATIN_REGEX.search(clean_text))
        has_bengali = bool(BENGALI_REGEX.search(clean_text))
        has_tamil = bool(TAMIL_REGEX.search(clean_text))
        has_japanese = bool(JAPANESE_REGEX.search(clean_text))

        if has_bengali:
            return LanguageDetectionResult(
                primary_language="bn",
                language_code="bn",
                confidence=0.95,
                details={"script": "bengali"},
            )
        if has_tamil:
            return LanguageDetectionResult(
                primary_language="ta",
                language_code="ta",
                confidence=0.95,
                details={"script": "tamil"},
            )
        if has_japanese:
            return LanguageDetectionResult(
                primary_language="ja",
                language_code="ja",
                confidence=0.95,
                details={"script": "japanese"},
            )

        # 2. Hindi and Hinglish checks
        words = set(re.findall(r"\b[a-zA-Z]+\b", clean_text.lower()))
        matching_hinglish = words.intersection(HINGLISH_VOCABULARY)

        # Mixed Devanagari + Latin words (code-switching)
        if has_devanagari and has_latin and len(words) >= 1:
            return LanguageDetectionResult(
                primary_language="hinglish",
                language_code="hinglish",
                confidence=0.92,
                is_code_switched=True,
                details={"markers": list(matching_hinglish), "mixed_scripts": True},
            )

        # Pure Devanagari (Hindi)
        if has_devanagari and not has_latin:
            return LanguageDetectionResult(
                primary_language="hi",
                language_code="hi",
                confidence=0.95,
                is_code_switched=False,
                details={"script": "devanagari"},
            )

        # Romanized Hinglish (Latin characters with 2+ Hindi vocabulary words)
        if len(matching_hinglish) >= 2 or (
            len(matching_hinglish) == 1 and (whisper_lang == "hi" or whisper_lang == "ur")
        ):
            return LanguageDetectionResult(
                primary_language="hinglish",
                language_code="hinglish",
                confidence=0.90,
                is_code_switched=True,
                details={"markers": list(matching_hinglish), "romanized_hindi": True},
            )

        # 3. Whisper acoustic detection hints
        norm_whisper = (whisper_lang or "").lower()
        if norm_whisper in ["fr", "french"]:
            return LanguageDetectionResult(
                primary_language="fr",
                language_code="fr",
                confidence=whisper_prob or 0.88,
                details={"whisper_lang": whisper_lang},
            )
        if norm_whisper in ["es", "spanish"]:
            return LanguageDetectionResult(
                primary_language="es",
                language_code="es",
                confidence=whisper_prob or 0.88,
                details={"whisper_lang": whisper_lang},
            )
        if norm_whisper in ["de", "german"]:
            return LanguageDetectionResult(
                primary_language="de",
                language_code="de",
                confidence=whisper_prob or 0.88,
                details={"whisper_lang": whisper_lang},
            )

        # 4. Lexical vocabulary disambiguation when Whisper language is unstated or ambiguous
        fr_matches = len(words.intersection(FRENCH_MARKERS))
        es_matches = len(words.intersection(SPANISH_MARKERS))
        de_matches = len(words.intersection(GERMAN_MARKERS))

        if fr_matches > 0 and fr_matches >= es_matches and fr_matches >= de_matches:
            return LanguageDetectionResult(
                primary_language="fr",
                language_code="fr",
                confidence=0.88,
                details={"lexical": "french"},
            )
        if es_matches > 0 and es_matches >= de_matches:
            return LanguageDetectionResult(
                primary_language="es",
                language_code="es",
                confidence=0.88,
                details={"lexical": "spanish"},
            )
        if de_matches > 0:
            return LanguageDetectionResult(
                primary_language="de",
                language_code="de",
                confidence=0.88,
                details={"lexical": "german"},
            )

        if norm_whisper in ["hi", "hindi"]:
            # Whisper detected Hindi: if has Latin characters, it's Hinglish; otherwise Hindi
            if has_latin:
                return LanguageDetectionResult(
                    primary_language="hinglish",
                    language_code="hinglish",
                    confidence=0.88,
                    is_code_switched=True,
                    details={"whisper_detected_hindi_with_latin": True},
                )
            return LanguageDetectionResult(
                primary_language="hi",
                language_code="hi",
                confidence=whisper_prob or 0.85,
            )

        # Default to English if Latin script and no foreign markers found
        conf = whisper_prob if whisper_prob and norm_whisper in ["en", "english"] else 0.85
        return LanguageDetectionResult(
            primary_language="en",
            language_code="en",
            confidence=conf,
            is_code_switched=False,
            is_uncertain=conf < 0.60,
            details={"whisper_lang": whisper_lang},
        )
