import os

from dotenv import load_dotenv


def get_api_key() -> str:
    """Load and return the Google Maps API key from environment."""
    load_dotenv()
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    return key
