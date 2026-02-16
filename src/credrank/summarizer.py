"""Lightweight one-line summaries for ranked place results."""

from credrank.llm import complete
from credrank.models import PlaceResult

SYSTEM_PROMPT = (
    "You generate ultra-concise one-sentence place summaries. "
    "Be specific and opinionated — mention what makes this place stand out. "
    "Never start with the place name. Never use filler like 'This place offers'. "
    "Maximum 20 words."
)


def _build_context(place: PlaceResult) -> str:
    """Assemble all available text signal for a place."""
    parts = []

    if place.reviews:
        review_text = "\n".join(
            f"- [{r.rating}/5] {r.text[:300]}" for r in place.reviews[:5]
        )
        parts.append(f"Recent reviews:\n{review_text}")

    if place.review_summary:
        parts.append(f"Review summary: {place.review_summary}")

    if place.generative_summary:
        parts.append(f"Overview: {place.generative_summary}")

    if place.editorial_summary:
        parts.append(f"Editorial: {place.editorial_summary}")

    return "\n\n".join(parts)


def summarize_place(place: PlaceResult) -> str:
    """Generate a one-line summary of a place from all available text signal."""
    context = _build_context(place)

    if not context:
        return "No reviews available."

    prompt = f"""Place: {place.name}
Address: {place.address or "Unknown"}
Google Rating: {place.rating} ({place.user_rating_count} reviews)

{context}

Write a single sentence summarizing this place's vibe and standout quality."""

    return complete(prompt, system=SYSTEM_PROMPT, max_tokens=200)


def summarize_top_n(ranked_results: list[dict], n: int = 5) -> list[dict]:
    """Add LLM summaries to the top N ranked results.

    Modifies entries in place by adding a 'summary' key.
    """
    for entry in ranked_results[:n]:
        entry["summary"] = summarize_place(entry["place"])

    return ranked_results
