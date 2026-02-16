"""Deep review analysis for item recommendations at a specific place."""

import json

from credrank.llm import complete
from credrank.models import PlaceResult

SYSTEM_PROMPT = (
    "You are a food and experience critic. Analyze all available information about "
    "a place to identify the specific items, dishes, or experiences that are most loved. "
    "Be concrete — name specific dishes, not categories. "
    "Respond ONLY with valid JSON, no markdown, no preamble."
)


def _build_review_context(place: PlaceResult) -> str:
    """Assemble all available text signal for deep analysis."""
    parts = []

    if place.reviews:
        review_text = "\n".join(
            f"- [{r.rating}/5] ({r.author}): {r.text[:500]}" for r in place.reviews
        )
        parts.append(f"Individual reviews:\n{review_text}")

    if place.review_summary:
        parts.append(f"Aggregated review summary: {place.review_summary}")

    if place.generative_summary:
        parts.append(f"Place overview: {place.generative_summary}")

    if place.editorial_summary:
        parts.append(f"Editorial: {place.editorial_summary}")

    return "\n\n".join(parts)


def recommend_items(place: PlaceResult, top_n: int = 3) -> dict:
    """Analyze reviews to find the best items at a place.

    Returns a dict with:
        - place_name: str
        - place_id: str
        - recommendations: list of {item, why, mentions}
        - avoid: list of {item, why}
        - summary: str (2-3 sentence deep summary)
    """
    context = _build_review_context(place)

    if not context:
        return {
            "place_name": place.name,
            "place_id": place.place_id,
            "recommendations": [],
            "avoid": [],
            "summary": "No reviews available for analysis.",
        }

    prompt = f"""Place: {place.name}
Type: {", ".join(place.types[:5]) if place.types else "Unknown"}
Rating: {place.rating} ({place.user_rating_count} reviews)

{context}

Analyze all available information and return JSON with this exact structure:
{{
    "recommendations": [
        {{"item": "specific dish name", "why": "brief reason from reviews", "mentions": number_of_sources_mentioning_it}}
    ],
    "avoid": [
        {{"item": "specific item", "why": "brief reason"}}
    ],
    "summary": "2-3 sentence summary of the dining experience, vibe, and standout qualities"
}}

Return up to {top_n} recommendations and up to 2 items to avoid.
Only include items specifically mentioned in the available information.
If no specific items are mentioned, return an empty recommendations list."""

    raw = complete(prompt, system=SYSTEM_PROMPT, max_tokens=2000)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = {"recommendations": [], "avoid": [], "summary": raw}

    return {
        "place_name": place.name,
        "place_id": place.place_id,
        **parsed,
    }
