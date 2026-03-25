import os

from dotenv import load_dotenv

load_dotenv()


def _get_env(key: str) -> str:
    """Return an environment variable, raising a clear error if missing."""
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"{key} is not set. Copy .env.example to .env and add your key."
        )
    return value


def get_api_key() -> str:
    """Return the Google Maps API key."""
    return _get_env("GOOGLE_MAPS_API_KEY")


def get_gemini_key() -> str:
    """Return the Google Gemini API key."""
    return _get_env("GOOGLE_GEMINI_API_KEY")
