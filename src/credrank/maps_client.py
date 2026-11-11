"""Google Places API (New) client."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import httpx

from credrank.cache import MISS, details_cache, search_cache
from credrank.config import get_api_key
from credrank.models import (
    PlaceResult,
    PlaceSearchParams,
    PlacesSearchResult,
    Review,
    SearchIntent,
)

BASE_URL = "https://places.googleapis.com/v1/places:searchText"
DETAIL_URL = "https://places.googleapis.com/v1/places"
log = logging.getLogger(__name__)

# Text Search (New) pagination constants.
# - pageSize max is 20 per Google's spec (maxResultCount was deprecated).
# - 3 pages * 20 = 60 results, which matches Google's practical cap per query.
# - pageToken in the New API is usable immediately; the 2-second delay
#   the old API required was a legacy carryover. Months of production use
#   at 0.5s showed no issues, so it's now 0 — set it back if Google ever
#   starts returning INVALID_ARGUMENT on fresh tokens.
_PAGE_SIZE = 20
_MAX_PAGES = 3
_PAGE_DELAY_S = 0.0

# Every Text Search page and Place Details call is billed individually.
# Rough Enterprise-SKU prices, used only for log-line cost estimates.
_TEXT_SEARCH_COST_USD = 0.035
_DETAILS_COST_USD = 0.025

# One shared client for the whole process: connection pooling avoids a
# fresh TCP+TLS handshake per request (2 RTTs each), which dominated
# latency for multi-page and grid searches. httpx.Client is thread-safe.
_http = httpx.Client(timeout=10)

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
        "places.primaryType",
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
        "primaryType",
        "rating",
        "userRatingCount",
        "reviews",
        "editorialSummary",
        "reviewSummary",
        "generativeSummary",
    ]
)


# ── LocalizedText helpers ──
#
# Google returns several user-facing strings as LocalizedText objects shaped
# like {"text": "...", "languageCode": "en"}. Some fields wrap one of those
# directly (e.g. editorialSummary); others nest it under another key
# (reviewSummary.text, generativeSummary.overview). These helpers pull the
# actual string out while tolerating missing/malformed structure.


def localized_text(value: Any) -> str | None:
    """Return the `.text` string from a LocalizedText dict, or None."""
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text:
            return text
    return None


def review_summary_text(place_json: dict) -> str | None:
    """Extract human-readable text from a Place's `reviewSummary`.

    Google's shape is `reviewSummary.text` = LocalizedText, so the real
    string lives two levels deep at `reviewSummary.text.text`.
    """
    review_summary = place_json.get("reviewSummary")
    if not isinstance(review_summary, dict):
        return None
    return localized_text(review_summary.get("text"))


def generative_summary_text(place_json: dict) -> str | None:
    """Extract overview text from a Place's `generativeSummary`.

    Google's shape is `generativeSummary.overview` = LocalizedText, so the
    real string lives at `generativeSummary.overview.text`. (The older
    `.text` field is deprecated and not what we want.)
    """
    generative = place_json.get("generativeSummary")
    if not isinstance(generative, dict):
        return None
    return localized_text(generative.get("overview"))


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
    location = place.get("location") or {}
    display_name = place.get("displayName") or {}

    reviews = []
    for r in place.get("reviews", []):
        text = (r.get("text") or {}).get("text", "")
        if text:
            reviews.append(
                Review(
                    rating=r.get("rating", 0),
                    text=text,
                    author=(r.get("authorAttribution") or {}).get("displayName", ""),
                )
            )

    return PlaceResult(
        place_id=place.get("id", ""),
        name=display_name.get("text", "Unknown"),
        rating=place.get("rating"),
        user_rating_count=place.get("userRatingCount"),
        address=place.get("formattedAddress"),
        lat=location.get("latitude"),
        lng=location.get("longitude"),
        types=place.get("types", []),
        primary_type=place.get("primaryType"),
        reviews=reviews,
        editorial_summary=localized_text(place.get("editorialSummary")),
        review_summary=review_summary_text(place),
        generative_summary=generative_summary_text(place),
    )


def _search_cache_key(params: PlaceSearchParams) -> tuple:
    """Cache key for a text search. Coordinates are rounded to 5 decimal
    places (~1m) so float noise doesn't defeat the cache while distinct
    grid cells still key separately."""
    lat, lng = params.location
    return (
        params.query.strip().lower(),
        round(lat, 5),
        round(lng, 5),
        params.radius_m,
        params.max_results,
        params.included_type,
        params.strict_type_filtering,
    )


def search_places(params: PlaceSearchParams) -> PlacesSearchResult:
    """Search for places within a circular area.

    Uses Text Search (New) with `pageSize`/`pageToken` pagination. Google
    deprecated `maxResultCount`; pageSize is capped at 20, so reaching the
    60-result practical cap requires up to 3 requests.

    Searches the bounding rectangle Google requires, then filters to
    places within the actual circle radius for honest boundaries.

    Results are cached (see `credrank.cache`) — every page is a billed
    Enterprise-SKU request, and users repeat searches far more often
    than the underlying ratings change. A cache hit reports
    `page_count=0` since no HTTP request was made.

    Returns a `PlacesSearchResult` which behaves like `list[PlaceResult]`
    but also exposes `.saturated` (True if more results were available
    when we stopped) and `.page_count` (number of HTTP requests made).
    """
    cache_key = _search_cache_key(params)
    cached = search_cache.get(cache_key)
    if cached is not MISS:
        places, saturated = cached
        log.info(
            "maps.search.cache_hit query=%r results=%d saturated=%s",
            params.query,
            len(places),
            saturated,
        )
        return PlacesSearchResult(places, saturated=saturated, page_count=0)

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

    body: dict[str, Any] = {
        "textQuery": params.query,
        "locationRestriction": _circle_to_bounding_box(lat, lng, params.radius_m),
        "pageSize": _PAGE_SIZE,
    }
    if params.included_type:
        body["includedType"] = params.included_type
        if params.strict_type_filtering:
            body["strictTypeFiltering"] = True

    headers = {
        "X-Goog-Api-Key": get_api_key(),
        "X-Goog-FieldMask": SEARCH_FIELD_MASK,
        "Content-Type": "application/json",
    }

    collected: list[PlaceResult] = []
    saturated = False
    page_count = 0

    while True:
        try:
            resp = _http.post(BASE_URL, json=body, headers=headers)
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            log.warning(
                "maps.search.timeout query=%r page=%d page_token=%r",
                params.query,
                page_count,
                body.get("pageToken"),
            )
            raise exc
        except httpx.HTTPError as exc:
            log.warning(
                "maps.search.error query=%r page=%d page_token=%r error=%s",
                params.query,
                page_count,
                body.get("pageToken"),
                exc,
            )
            raise exc

        data = resp.json()
        page_count += 1

        for place in data.get("places", []):
            parsed = _parse_place(place)
            collected.append(parsed)
            log.debug(
                "maps.search.place place_id=%s name=%r rating=%s reviews=%s",
                parsed.place_id,
                parsed.name,
                parsed.rating,
                parsed.user_rating_count,
            )

        next_token = data.get("nextPageToken")

        # Google silently caps Text Search (New) at roughly 60 results
        # across 3 pages of 20 and often returns NO nextPageToken on
        # that final page even when more results exist in the area.
        # Treat a fully-maxed response (every page of every fetched
        # page used up to the theoretical cap) as saturated regardless
        # of token state — otherwise dense cells look "complete" and
        # the adaptive grid never subdivides them.
        hit_google_cap = (
            page_count >= _MAX_PAGES
            and len(collected) >= _PAGE_SIZE * _MAX_PAGES
        )

        # Stop if we have enough. More may still be available — flag that.
        if len(collected) >= params.max_results:
            saturated = bool(next_token) or hit_google_cap
            break

        # Google has nothing more.
        if not next_token:
            break

        # Page cap reached but Google still has more — mark saturated.
        if page_count >= _MAX_PAGES:
            saturated = True
            break

        body["pageToken"] = next_token
        if _PAGE_DELAY_S:
            time.sleep(_PAGE_DELAY_S)

    # Filter to places within the actual circle radius
    filtered: list[PlaceResult] = []
    for p in collected:
        if p.lat is not None and p.lng is not None:
            dist = haversine_m(lat, lng, p.lat, p.lng)
            if dist <= params.radius_m:
                filtered.append(p)
        else:
            filtered.append(p)  # Keep places without coords (rare edge case)

    final = filtered[: params.max_results]
    duration_ms = int((time.perf_counter() - started_at) * 1000)
    log.info(
        "maps.search.done query=%r raw_count=%d in_radius=%d returned=%d "
        "page_count=%d saturated=%s est_cost_usd=%.3f duration_ms=%d",
        params.query,
        len(collected),
        len(filtered),
        len(final),
        page_count,
        saturated,
        page_count * _TEXT_SEARCH_COST_USD,
        duration_ms,
    )
    search_cache.set(cache_key, (tuple(final), saturated))
    return PlacesSearchResult(final, saturated=saturated, page_count=page_count)


def get_place_details(place_id: str) -> PlaceResult | None:
    """Fetch full details for a single place by ID, including reviews and summaries.

    This uses the Atmosphere SKU — only call it when you need LLM input.
    Results (including "not found") are cached so e.g. clicking both
    "Summarize" and "What to order" on the same place bills one fetch.
    """
    cached = details_cache.get(place_id)
    if cached is not MISS:
        log.info("maps.details.cache_hit place_id=%s", place_id)
        return cached

    url = f"{DETAIL_URL}/{place_id}"
    started_at = time.perf_counter()
    log.info("maps.details.start place_id=%s", place_id)

    headers = {
        "X-Goog-Api-Key": get_api_key(),
        "X-Goog-FieldMask": DETAIL_FIELD_MASK,
        "Content-Type": "application/json",
    }

    try:
        resp = _http.get(url, headers=headers)
        if resp.status_code == 404:
            log.info("maps.details.not_found place_id=%s", place_id)
            details_cache.set(place_id, None)
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
        "maps.details.done place_id=%s est_cost_usd=%.3f duration_ms=%d",
        place_id,
        _DETAILS_COST_USD,
        duration_ms,
    )
    details_cache.set(place_id, parsed)
    return parsed


def search_places_with_intent(
    params: PlaceSearchParams,
    intent: SearchIntent,
) -> PlacesSearchResult:
    """Search with SearchIntent-aware server-side type filtering.

    Google's `includedType` only accepts Table A types, so we first
    narrow `intent.included_types` to `TABLE_A_FILTERABLE`. Any Table B
    types (like `pizza_delivery`) stay in the intent and will be caught
    in the client-side `filter_by_intent` pass — we just can't push
    them down to Google.

    Dispatches on how many Table A types remain:

    - 0: a single broad `search_places` call (no `includedType`).
    - 1: a single `search_places` call with `includedType` +
      `strictTypeFiltering=True`, so Google drops unrelated results.
    - N: one request per type, merged on `place_id` with first-seen
      winning. Aggregate `saturated` is True if any sub-search was
      saturated; `page_count` is summed across sub-searches.

    Post-filtering (name-term evidence, excluded types, strict/broad
    mode) lives in `relevance.filter_by_intent`. This function is
    only about fetching the right superset.
    """
    # Avoid a circular import at module load: relevance imports models,
    # models is safe, but maps_client <-> relevance could loop if
    # relevance ever imported from here.
    from credrank.relevance import TABLE_A_FILTERABLE

    server_types = [
        t for t in (intent.included_types or []) if t in TABLE_A_FILTERABLE
    ]

    if not server_types:
        return search_places(params)

    if len(server_types) == 1:
        return search_places(_with_type(params, server_types[0], strict=True))

    seen: dict[str, PlaceResult] = {}
    saturated = False
    page_count = 0

    for t in server_types:
        try:
            sub_result = search_places(_with_type(params, t, strict=True))
        except Exception as exc:
            log.warning(
                "maps.multitype.error query=%r type=%s error=%s",
                params.query,
                t,
                exc,
            )
            continue

        for place in sub_result:
            if place.place_id and place.place_id not in seen:
                seen[place.place_id] = place

        saturated = saturated or sub_result.saturated
        page_count += sub_result.page_count

        if len(seen) >= params.max_results:
            break

    merged = list(seen.values())[: params.max_results]
    log.info(
        "maps.multitype.done query=%r types=%s returned=%d page_count=%d saturated=%s",
        params.query,
        server_types,
        len(merged),
        page_count,
        saturated,
    )
    return PlacesSearchResult(merged, saturated=saturated, page_count=page_count)


def _with_type(
    params: PlaceSearchParams,
    included_type: str,
    *,
    strict: bool,
) -> PlaceSearchParams:
    """Return a copy of `params` with an `includedType` bound."""
    return PlaceSearchParams(
        query=params.query,
        location=params.location,
        radius_m=params.radius_m,
        max_results=params.max_results,
        included_type=included_type,
        strict_type_filtering=strict,
    )
