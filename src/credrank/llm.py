"""Thin wrapper around the Google Gemini API."""

from google import genai

from credrank.config import get_gemini_key

_client: genai.Client | None = None


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

    config = genai.types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=0.3,
        thinking_config=genai.types.ThinkingConfig(
            thinking_budget=0 if not thinking else None,
        ),
    )
    if system:
        config.system_instruction = system

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=config,
    )

    return response.text
