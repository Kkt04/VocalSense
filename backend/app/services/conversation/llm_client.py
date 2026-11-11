"""
LLM client for VocalSense conversational response generation.

Uses an OpenAI-compatible chat completions API so any provider works:
  - Google Gemini (via OpenAI-compat endpoint)
  - Groq
  - OpenAI / Azure OpenAI
  - Anthropic (via proxy)
  - Local Ollama

Configuration (via .env or environment variables):
  LLM_API_KEY      — provider API key (leave empty to disable LLM)
  LLM_API_BASE_URL — provider base URL (default: Gemini OpenAI-compat)
  LLM_MODEL        — model identifier (default: gemini-2.0-flash)
  LLM_MAX_TOKENS   — max completion tokens (default: 400)
  LLM_TEMPERATURE  — sampling temperature (default: 0.85)
  LLM_TIMEOUT_SECONDS — request timeout (default: 20.0)

When LLM_API_KEY is empty, generate() returns None so the caller can
fall back to the template bank without crashing.
"""

import json
import logging
from typing import Optional, List, Dict, Any

import httpx

from backend.app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin async wrapper around any OpenAI-compatible chat completions endpoint.
    Handles timeouts, error logging, and clean None-returns on failure so the
    caller can fall back gracefully.
    """

    @classmethod
    async def generate(
        cls,
        system_prompt: str,
        messages: List[Dict[str, str]],
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
    ) -> Optional[str]:
        """
        Call the configured LLM and return the assistant reply text.

        Args:
            system_prompt: The system instruction for personality and language.
            messages: Chronological chat messages as [{"role": ..., "content": ...}].
            api_key_override: Optional user-supplied API key overriding environment.
            model_override: Optional user-supplied model name.
            base_url_override: Optional user-supplied base URL.

        Returns:
            Generated text string, or None if LLM is unconfigured or the call fails.
        """
        api_key = (api_key_override or settings.LLM_API_KEY or "").strip()
        if not api_key:
            logger.debug(
                "LLM_API_KEY not configured — skipping LLM call. "
                "Set LLM_API_KEY in .env or provide via client settings to enable dynamic response generation."
            )
            return None

        model = (model_override or settings.LLM_MODEL).strip()
        base_url = (base_url_override or settings.LLM_API_BASE_URL).rstrip('/')

        # Auto-detect Groq keys (always start with 'gsk_')
        if api_key.startswith("gsk_") and "generativelanguage.googleapis.com" in base_url:
            logger.info("Auto-adapting Groq key with Groq endpoint and qwen/qwen3.8-27b model.")
            base_url = "https://api.groq.com/openai/v1"
            if model in ("gemini-2.0-flash", "llama-3.3-70b-versatile", "llama3-8b-8192"):
                model = "qwen/qwen3.8-27b"

        # Auto-detect OpenAI keys (start with 'sk-' and not 'gsk_')
        if api_key.startswith("sk-") and "generativelanguage.googleapis.com" in base_url and not api_key.startswith("gsk_"):
            logger.info("Auto-adapting OpenAI key with OpenAI endpoint and gpt-4o-mini model.")
            base_url = "https://api.openai.com/v1"
            if model == "gemini-2.0-flash":
                model = "gpt-4o-mini"

        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": settings.LLM_TEMPERATURE,
        }

        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            choices = data.get("choices") or []
            if not choices:
                logger.warning("LLM returned no choices: %s", data)
                return None

            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                logger.warning("LLM returned empty content in first choice.")
                return None

            logger.debug("LLM response (%d chars): %.120s…", len(content), content)
            return content

        except httpx.TimeoutException:
            logger.warning(
                "LLM request timed out after %.1fs for model '%s'.",
                settings.LLM_TIMEOUT_SECONDS,
                settings.LLM_MODEL,
            )
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "LLM HTTP error %s — %s",
                exc.response.status_code,
                exc.response.text[:300],
            )
            return None
        except Exception as exc:
            logger.warning("LLM call failed unexpectedly: %s", exc, exc_info=False)
            return None


def build_chat_messages(
    transcript: str,
    history: Optional[List[Dict[str, Any]]] = None,
    is_code_switched: bool = False,
    vocal_pattern: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build the ordered list of chat messages sent to the LLM.

    Includes:
    - Previous conversation turns from MongoDB (chronological order, capped at 10)
    - The current user transcript as the final human message
    - A light acoustic note when a reliable vocal pattern is available

    Args:
        transcript: The raw transcription of what the user said this turn.
        history: Prior conversation turns as list of dicts with 'role' and 'text'.
        is_code_switched: Whether Hinglish / mixed-language was detected.
        vocal_pattern: Optional acoustic vocal pattern label (e.g. "calm", "happy").

    Returns:
        List of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    messages: List[Dict[str, str]] = []

    # Include up to the last 10 turns of conversation history for context
    if history:
        for turn in history[-10:]:
            role = turn.get("role", "user")
            text = turn.get("text", "").strip()
            if text:
                messages.append({
                    "role": "assistant" if role == "assistant" else "user",
                    "content": text,
                })

    # Compose the user's current message, optionally annotating acoustic info
    user_content = transcript.strip()
    if vocal_pattern and vocal_pattern not in ("uncertain", "neutral"):
        user_content += (
            f"\n\n[Note: Acoustic analysis detected a '{vocal_pattern}'-like vocal pattern. "
            "Treat this as a subtle signal only — do not state or assume the user's emotional state.]"
        )

    messages.append({"role": "user", "content": user_content})
    return messages
