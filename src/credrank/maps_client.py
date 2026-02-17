"""Google Places API (New) client."""

import time

import httpx

from credrank.config import get_api_key
from credrank.models import PlaceResult, PlaceSearchParams, Review

BASE_URL = "https://places.googleapis.com/v1/places:searchText"
DETAIL_URL = "https://places.googleapis.com/v1/places"

# Lightweight mask for search results — no reviews, no summaries.
# Billed at Enterprise SKU (rating + userRatingCount) but skips Atmosphere.
SEARCH_FIELD_MASK = ",".join(
    [
        "places.id",
        "nextPageToken",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.types",
        "places.rating",
        "places.userRatingCount",
    ]
)

# Full mask for place details — includes reviews and summaries for LLM analysis.
# Billed at Enterprise + Atmosphere SKU.
DETAIL_FIELD_MASK = ",".join(
    [
        "id",
        "displayName",
        "formattedAddress",
        "location",
        "types",
        "rating",
        "userRatingCount",
        "reviews",
        "editorialSummary",
        "reviewSummary",
        "generativeSummary",
    ]
)


def _circle_to_bounding_box(lat: float, lng: float, radius_m: int) -> dict:
    """Convert a circle (center + radius) to a rectangular locationRestriction."""
    d_lat = radius_m / 111_320
    d_lng = radius_m / (
        111_320 * __import__("math").cos(__import__("math").radians(lat))
    )

    return {
        "rectangle": {
            "low": {"latitude": lat - d_lat, "longitude": lng - d_lng},
            "high": {"latitude": lat + d_lat, "longitude": lng + d_lng},
        }
    }


def _parse_place(place: dict) -> PlaceResult:
    """Map a raw Google Places API response object to our PlaceResult model."""
    location = place.get("location", {})
    display_name = place.get("displayName", {})

    reviews = []
    for r in place.get("reviews", []):
        text = r.get("text", {}).get("text", "")
        if text:
            reviews.append(
                Review(
                    rating=r.get("rating", 0),
                    text=text,
                    author=r.get("authorAttribution", {}).get("displayName", ""),
                )
            )

    editorial = place.get("editorialSummary", {})
    review_summary = place.get("reviewSummary", {})
    generative = place.get("generativeSummary", {})

    return PlaceResult(
        place_id=place.get("id", ""),
        name=display_name.get("text", "Unknown"),
        rating=place.get("rating"),
        user_rating_count=place.get("userRatingCount"),
        address=place.get("formattedAddress"),
        lat=location.get("latitude"),
        lng=location.get("longitude"),
        types=place.get("types", []),
        reviews=reviews,
        editorial_summary=editorial.get("text")
        if isinstance(editorial, dict)
        else None,
        review_summary=review_summary.get("text")
        if isinstance(review_summary, dict)
        else None,
        generative_summary=generative.get("text")
        if isinstance(generative, dict)
        else None,
    )


def search_places(params: PlaceSearchParams) -> list[PlaceResult]:
    """Search for places using the lightweight field mask (no reviews)."""
    lat, lng = params.location

    body = {
        "textQuery": params.query,
        "locationRestriction": _circle_to_bounding_box(lat, lng, params.radius_m),
        "maxResultCount": min(params.max_results, 20),
    }

    headers = {
        "X-Goog-Api-Key": get_api_key(),
        "X-Goog-FieldMask": SEARCH_FIELD_MASK,
        "Content-Type": "application/json",
    }

    results: list[PlaceResult] = []

    with httpx.Client() as client:
        while True:
            resp = client.post(BASE_URL, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            for place in data.get("places", []):
                results.append(_parse_place(place))

            next_token = data.get("nextPageToken")
            if not next_token or len(results) >= params.max_results:
                break

            body["pageToken"] = next_token
            time.sleep(2)

    return results[: params.max_results]


def get_place_details(place_id: str) -> PlaceResult | None:
    """Fetch full details for a single place by ID, including reviews and summaries.

    This uses the Atmosphere SKU — only call it when you need LLM input.
    """
    url = f"{DETAIL_URL}/{place_id}"

    headers = {
        "X-Goog-Api-Key": get_api_key(),
        "X-Goog-FieldMask": DETAIL_FIELD_MASK,
        "Content-Type": "application/json",
    }

    with httpx.Client() as client:
        resp = client.get(url, headers=headers)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return _parse_place(resp.json())
