"""Google Places API (New) client."""

import logging
import math
import time

import httpx

from credrank.config import get_api_key
from credrank.models import PlaceResult, PlaceSearchParams, Review

BASE_URL = "https://places.googleapis.com/v1/places:searchText"
DETAIL_URL = "https://places.googleapis.com/v1/places"
log = logging.getLogger(__name__)

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


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return the great-circle distance in meters between two points."""
    R = 6_371_000  # Earth radius in meters
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _circle_to_bounding_box(lat: float, lng: float, radius_m: int) -> dict:
    """Convert a circle (center + radius) to a rectangular locationRestriction."""
    d_lat = radius_m / 111_320
    d_lng = radius_m / (111_320 * math.cos(math.radians(lat)))

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
    """Search for places within a circular area.

    Searches the bounding rectangle (API requirement) then filters to
    places within the actual circle radius for honest boundaries.
    """
    lat, lng = params.location
    started_at = time.perf_counter()
    log.info(
        "maps.search.start query=%r lat=%.6f lng=%.6f radius_m=%d max_results=%d",
        params.query,
        lat,
        lng,
        params.radius_m,
        params.max_results,
    )

    body = {
        "textQuery": params.query,
        "locationRestriction": _circle_to_bounding_box(lat, lng, params.radius_m),
        "maxResultCount": min(params.max_results, 60),
    }

    headers = {
        "X-Goog-Api-Key": get_api_key(),
        "X-Goog-FieldMask": SEARCH_FIELD_MASK,
        "Content-Type": "application/json",
    }

    results: list[PlaceResult] = []

    with httpx.Client(timeout=10) as client:
        while True:
            try:
                resp = client.post(BASE_URL, json=body, headers=headers)
                resp.raise_for_status()
            except httpx.TimeoutException as exc:
                log.warning(
                    "maps.search.timeout query=%r page_token=%r",
                    params.query,
                    body.get("pageToken"),
                )
                raise exc
            except httpx.HTTPError as exc:
                log.warning(
                    "maps.search.error query=%r page_token=%r error=%s",
                    params.query,
                    body.get("pageToken"),
                    exc,
                )
                raise exc
            data = resp.json()

            for place in data.get("places", []):
                parsed = _parse_place(place)
                results.append(parsed)
                log.debug(
                    "maps.search.place place_id=%s name=%r rating=%s reviews=%s",
                    parsed.place_id,
                    parsed.name,
                    parsed.rating,
                    parsed.user_rating_count,
                )

            next_token = data.get("nextPageToken")
            if not next_token or len(results) >= params.max_results:
                break

            body["pageToken"] = next_token
            time.sleep(2)

    # Filter to places within the actual circle radius
    filtered = []
    for p in results:
        if p.lat is not None and p.lng is not None:
            dist = haversine_m(lat, lng, p.lat, p.lng)
            if dist <= params.radius_m:
                filtered.append(p)
        else:
            filtered.append(p)  # Keep places without coords (rare edge case)

    final = filtered[: params.max_results]
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    log.info(
        "maps.search.done query=%r raw_count=%d in_radius=%d returned=%d duration_ms=%d",
        params.query,
        len(results),
        len(filtered),
        len(final),
        duration_ms,
    )
    return final


def get_place_details(place_id: str) -> PlaceResult | None:
    """Fetch full details for a single place by ID, including reviews and summaries.

    This uses the Atmosphere SKU — only call it when you need LLM input.
    """
    url = f"{DETAIL_URL}/{place_id}"
    started_at = time.perf_counter()
    log.info("maps.details.start place_id=%s", place_id)

    headers = {
        "X-Goog-Api-Key": get_api_key(),
        "X-Goog-FieldMask": DETAIL_FIELD_MASK,
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=10) as client:
        try:
            resp = client.get(url, headers=headers)
            if resp.status_code == 404:
                log.info("maps.details.not_found place_id=%s", place_id)
                return None
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            log.warning("maps.details.timeout place_id=%s", place_id)
            raise exc
        except httpx.HTTPError as exc:
            log.warning("maps.details.error place_id=%s error=%s", place_id, exc)
            raise exc

        parsed = _parse_place(resp.json())
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        log.debug(
            "maps.details.place place_id=%s name=%r rating=%s reviews=%s",
            parsed.place_id,
            parsed.name,
            parsed.rating,
            parsed.user_rating_count,
        )
        log.info(
            "maps.details.done place_id=%s duration_ms=%d",
            place_id,
            duration_ms,
        )
        return parsed
