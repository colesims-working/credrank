"""Thin wrapper around the Google Gemini API."""

import logging
import time

from google import genai

from credrank.config import get_gemini_key

_client: genai.Client | None = None
log = logging.getLogger(__name__)


def get_client() -> genai.Client:
    """Return a reusable Gemini client instance."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=get_gemini_key())
    return _client


def complete(
    prompt: str,
    system: str = "",
    max_tokens: int = 300,
    thinking: bool = False,
) -> str:
    """Send a prompt to Gemini Flash and return the text response.

    Args:
        prompt: The user prompt.
        system: Optional system instruction.
        max_tokens: Maximum output tokens.
        thinking: Whether to allow the model to use thinking tokens.
            Disable for straightforward extraction/summarization tasks.
    """
    client = get_client()
    started_at = time.perf_counter()

    config = genai.types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=0.3,
        thinking_config=genai.types.ThinkingConfig(
            thinking_budget=0 if not thinking else None,
        ),
    )
    if system:
        config.system_instruction = system

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=config,
        )
    except TimeoutError:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        log.warning(
            "llm.complete.timeout duration_ms=%d prompt_chars=%d",
            duration_ms,
            len(prompt),
        )
        return ""
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        log.warning(
            "llm.complete.error duration_ms=%d prompt_chars=%d error=%s",
            duration_ms,
            len(prompt),
            exc,
        )
        raise

    text = response.text or ""
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    log.info(
        "llm.complete.done duration_ms=%d prompt_chars=%d output_chars=%d",
        duration_ms,
        len(prompt),
        len(text),
    )
    return text
