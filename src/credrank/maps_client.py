import math
import time

import httpx

from credrank.config import get_api_key
from credrank.models import PlaceResult, PlaceSearchParams, Review

BASE_URL = "https://places.googleapis.com/v1/places:searchText"

FIELD_MASK = ",".join(
    [
        # Essentials ID Only
        "places.id",
        "nextPageToken",
        # Pro
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.types",
        # Enterprise
        "places.rating",
        "places.userRatingCount",
        # Enterprise + Atmosphere
        "places.reviews",
        "places.editorialSummary",
        "places.reviewSummary",
        "places.generativeSummary",
    ]
)


def _circle_to_bounding_box(lat: float, lng: float, radius_m: float) -> dict:
    """Convert a center point + radius into a rectangular bounding box.

    Approximation using the fact that:
        1 degree latitude  ≈ 111,320 meters (constant)
        1 degree longitude ≈ 111,320 * cos(latitude) meters (varies)
    """
    delta_lat = radius_m / 111_320
    delta_lng = radius_m / (111_320 * math.cos(math.radians(lat)))

    return {
        "rectangle": {
            "low": {
                "latitude": lat - delta_lat,
                "longitude": lng - delta_lng,
            },
            "high": {
                "latitude": lat + delta_lat,
                "longitude": lng + delta_lng,
            },
        }
    }


def search_places(params: PlaceSearchParams) -> list[PlaceResult]:
    """Search Google Places and return normalized results.

    Uses the Places API (New) Text Search endpoint. Handles pagination
    automatically up to params.max_results.
    """
    api_key = get_api_key()
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    lat, lng = params.location
    body = {
        "textQuery": params.query,
        "locationRestriction": _circle_to_bounding_box(
            lat, lng, float(params.radius_m)
        ),
        "pageSize": min(params.max_results, 20),
    }

    results: list[PlaceResult] = []
    page_token: str | None = None

    with httpx.Client(timeout=10) as client:
        while len(results) < params.max_results:
            if page_token:
                body["pageToken"] = page_token

            response = client.post(BASE_URL, headers=headers, json=body)

            if response.status_code != 200:
                raise RuntimeError(
                    f"Google Places request failed: {response.status_code} - {response.text}"
                )

            data = response.json()

            # Empty response means no results
            if "places" not in data:
                break

            for place in data["places"]:
                if len(results) >= params.max_results:
                    break
                results.append(_parse_place(place))

            page_token = data.get("nextPageToken")
            if not page_token:
                break  # No more pages

            time.sleep(2)  # Required delay before using nextPageToken

    return results


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
