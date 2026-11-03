import re
from typing import Tuple, Dict, Any, Optional

# Common Hinglish Romanized vocabulary markers
COMMON_HINGLISH_WORDS = {
    "yaar", "mujhe", "mera", "meri", "mere", "hai", "hain", "karna", "karni", "karte",
    "nahi", "nahin", "accha", "achha", "theek", "thik", "kya", "kyun", "kaise",
    "toh", "aur", "lekin", "bhi", "bhai", "aap", "hum", "tum", "baat", "karo", "karenge",
    "soch", "lagta", "raha", "rahi", "rahe", "hoga", "hogi", "samajh", "pata", "chalo",
}

DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
LATIN_RANGE = re.compile(r"[a-zA-Z]")


def detect_language_characteristics(
    whisper_detected_lang: str,
    language_prob: float,
    transcript_text: str,
    all_language_probs: Optional[Dict[str, float]] = None,
) -> Tuple[str, bool, Dict[str, Any]]:
    """
    Analyzes transcript content and Whisper language probabilities to identify
    whether the speech is English, Hindi, or Hindi-English code-switching (Hinglish).

    Returns:
        Tuple[str, bool, Dict[str, Any]]:
            - final_lang: 'en', 'hi', or 'hinglish'
            - is_hinglish: True if code-switching is detected
            - details: diagnostic metadata for research and explainability
    """
    clean_text = transcript_text.lower().strip()
    words = set(re.findall(r"\b[a-zA-Z]+\b", clean_text))

    has_devanagari = bool(DEVANAGARI_RANGE.search(transcript_text))
    has_latin = bool(LATIN_RANGE.search(transcript_text))

    # Count matching common romanized Hindi words
    matching_hinglish_words = words.intersection(COMMON_HINGLISH_WORDS)
    hinglish_word_count = len(matching_hinglish_words)

    is_hinglish = False
    final_lang = whisper_detected_lang

    # Case 1: Text has both Devanagari script and Latin English words
    # Example: "कल मेरा presentation complete करना है"
    if has_devanagari and has_latin and len(words) >= 1:
        is_hinglish = True
        final_lang = "hinglish"

    # Case 2: Whisper detected English or Hindi, but text has multiple romanized Hindi words alongside English
    # Example: "Yaar, mujhe kal presentation complete karna hai"
    elif hinglish_word_count >= 2:
        is_hinglish = True
        final_lang = "hinglish"

    # Case 3: Whisper detected Hindi ('hi'), but Latin characters are predominant
    elif whisper_detected_lang == "hi" and not has_devanagari and has_latin:
        is_hinglish = True
        final_lang = "hinglish"

    # Case 4: Language probabilities check if available
    elif all_language_probs:
        hi_prob = all_language_probs.get("hi", 0.0)
        en_prob = all_language_probs.get("en", 0.0)
        # If both Hindi and English have significant probabilities
        if hi_prob > 0.25 and en_prob > 0.25 and hinglish_word_count >= 1:
            is_hinglish = True
            final_lang = "hinglish"

    details = {
        "whisper_raw_lang": whisper_detected_lang,
        "whisper_prob": language_prob,
        "hinglish_markers_found": list(matching_hinglish_words),
        "has_devanagari": has_devanagari,
        "has_latin": has_latin,
        "code_switching_detected": is_hinglish,
    }

    return final_lang, is_hinglish, details
