"""FastAPI backend for CredRank."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from credrank.auth import BasicAuthMiddleware, validate_auth_config
from credrank.cache import MISS, all_stats, geocode_cache
from credrank.config import get_api_key
from credrank.grid_search import grid_search
from credrank.maps_client import (
    get_place_details,
    search_places_with_intent,
)
from credrank.models import GridCoverage, PlaceSearchParams
from credrank.ranking import rank_places
from credrank.recommender import recommend_items
from credrank.relevance import TABLE_A_FILTERABLE, build_intent, filter_by_intent
from credrank.summarizer import summarize_place

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

validate_auth_config()

app = FastAPI(title="CredRank", version="0.3.0")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware order note: last added = outermost = runs first. We want
# CORS preflight (OPTIONS) to be handled before auth rejects it, so
# auth is added first (inner) and CORS second (outer, only if enabled).
app.add_middleware(BasicAuthMiddleware)

_cors_origins_env = os.environ.get("CORS_ORIGINS", "").strip()
if _cors_origins_env:
    _origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )


# ── Request / Response Models ──


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    radius_m: int = Field(default=8000, gt=0, le=80000)
    max_results: int = Field(default=60, gt=0, le=500)
    deep_search: bool = Field(default=False)


class PlaceResponse(BaseModel):
    rank: int
    credscore: float
    bayesian_avg: float
    place_id: str
    name: str
    rating: float | None
    user_rating_count: int | None
    address: str | None
    lat: float | None
    lng: float | None
    types: list[str]


class CoverageResponse(BaseModel):
    """Grid-search coverage metadata surfaced to the frontend.

    `complete` is False when the server couldn't guarantee full recall
    (a cell failed, a saturated cell couldn't subdivide further, or the
    safety cap tripped). The frontend shows a warning banner in that case.
    """

    complete: bool
    reason: str | None = None
    cells_searched: int
    cells_failed: int
    cells_saturated: int
    raw_count: int
    deduped_count: int
    in_radius_count: int


class SearchResponse(BaseModel):
    query: str
    result_count: int
    filtered_count: int = 0
    deep_search: bool = False
    coverage: CoverageResponse | None = None
    places: list[PlaceResponse]


class PlaceIdRequest(BaseModel):
    place_id: str = Field(min_length=1, max_length=500, pattern=r"^[a-zA-Z0-9_-]+$")


class SummaryResponse(BaseModel):
    place_id: str
    name: str
    summary: str


class RecommendationItem(BaseModel):
    item: str
    why: str
    mentions: int | None = None


class AvoidItem(BaseModel):
    item: str
    why: str


class RecommendResponse(BaseModel):
    place_name: str
    place_id: str
    summary: str
    recommendations: list[RecommendationItem]
    avoid: list[AvoidItem]


class GeocodeResponse(BaseModel):
    lat: float | None
    lng: float | None
    name: str | None


class ReverseGeocodeResponse(BaseModel):
    name: str


# ── Endpoints ──


@app.post("/api/search", response_model=SearchResponse)
@limiter.limit("30/minute")
def search(request: Request, req: SearchRequest):
    """Search, filter by relevance, rank, and return places with CredScores."""
    _ = request
    started_at = time.perf_counter()
    log.info(
        "api.search.start query=%r lat=%.6f lng=%.6f radius_m=%d max_results=%d deep_search=%s",
        req.query,
        req.lat,
        req.lng,
        req.radius_m,
        req.max_results,
        req.deep_search,
    )

    # 1) Extract structured intent — LLM first, deterministic fallback.
    intent = build_intent(req.query)
    log.debug(
        "api.search.intent query=%r kind=%s included=%s required=%s",
        req.query,
        intent.query_kind,
        intent.included_types,
        intent.required_terms,
    )

    # 2) Search — grid search for deep, otherwise intent-aware standard search.
    grid_coverage: GridCoverage | None = None
    if req.deep_search:
        # Grid search can push a single Table A type to the server side.
        # If the intent spans multiple types we let grid search run broad
        # and rely on post-filter to clean up — avoids exploding cost to
        # num_cells * num_types requests.
        server_types = [
            t for t in intent.included_types if t in TABLE_A_FILTERABLE
        ]
        grid_included_type = server_types[0] if len(server_types) == 1 else None
        grid_result = grid_search(
            query=req.query,
            center_lat=req.lat,
            center_lng=req.lng,
            radius_m=req.radius_m,
            included_type=grid_included_type,
        )
        places = list(grid_result)
        grid_coverage = grid_result.coverage
        log.debug(
            "api.search.grid query=%r cells=%d saturated=%d complete=%s",
            req.query,
            grid_coverage.cells_searched,
            grid_coverage.cells_saturated,
            grid_coverage.complete,
        )
    else:
        params = PlaceSearchParams(
            query=req.query,
            location=(req.lat, req.lng),
            radius_m=req.radius_m,
            max_results=min(req.max_results, 60),
        )
        places = search_places_with_intent(params, intent)

    coverage_response = _coverage_to_response(grid_coverage)

    if not places:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(
            "api.search.done query=%r result_count=0 filtered_count=0 duration_ms=%d",
            req.query,
            duration_ms,
        )
        return SearchResponse(
            query=req.query,
            result_count=0,
            deep_search=req.deep_search,
            coverage=coverage_response,
            places=[],
        )

    # 3) Client-side post-filter by intent (catches Google false positives
    #    and keeps ambiguous matches with name/summary evidence).
    raw_count = len(places)
    places = filter_by_intent(places, intent)
    filtered_count = raw_count - len(places)
    log.info(
        "api.search.filter query=%r raw_count=%d filtered_count=%d remaining=%d",
        req.query,
        raw_count,
        filtered_count,
        len(places),
    )

    if not places:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        log.info(
            "api.search.done query=%r result_count=0 filtered_count=%d duration_ms=%d",
            req.query,
            filtered_count,
            duration_ms,
        )
        return SearchResponse(
            query=req.query,
            result_count=0,
            filtered_count=filtered_count,
            deep_search=req.deep_search,
            coverage=coverage_response,
            places=[],
        )

    # 4) Rank — then truncate to the user's requested display count.
    ranked = rank_places(places)
    ranked_top = ranked[: req.max_results]

    results = []
    for entry in ranked_top:
        p = entry["place"]
        log.debug(
            "api.search.result rank=%d place_id=%s name=%r credscore=%.1f bayesian_avg=%.3f",
            entry["rank"],
            p.place_id,
            p.name,
            entry["credscore"],
            entry["bayesian_avg"],
        )
        results.append(
            PlaceResponse(
                rank=entry["rank"],
                credscore=round(entry["credscore"], 1),
                bayesian_avg=round(entry["bayesian_avg"], 3),
                place_id=p.place_id,
                name=p.name,
                rating=p.rating,
                user_rating_count=p.user_rating_count,
                address=p.address,
                lat=p.lat,
                lng=p.lng,
                types=p.types,
            )
        )

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    log.info(
        "api.search.done query=%r result_count=%d filtered_count=%d "
        "deep_search=%s coverage_complete=%s duration_ms=%d",
        req.query,
        len(results),
        filtered_count,
        req.deep_search,
        grid_coverage.complete if grid_coverage else None,
        duration_ms,
    )
    return SearchResponse(
        query=req.query,
        result_count=len(results),
        filtered_count=filtered_count,
        deep_search=req.deep_search,
        coverage=coverage_response,
        places=results,
    )


def _coverage_to_response(
    coverage: GridCoverage | None,
) -> CoverageResponse | None:
    """Convert a GridCoverage dataclass to the pydantic response model."""
    if coverage is None:
        return None
    return CoverageResponse(
        complete=coverage.complete,
        reason=coverage.reason,
        cells_searched=coverage.cells_searched,
        cells_failed=coverage.cells_failed,
        cells_saturated=coverage.cells_saturated,
        raw_count=coverage.raw_count,
        deduped_count=coverage.deduped_count,
        in_radius_count=coverage.in_radius_count,
    )


@app.post("/api/summarize", response_model=SummaryResponse)
@limiter.limit("10/minute")
def summarize(request: Request, req: PlaceIdRequest):
    """Generate an LLM summary for a specific place by ID."""
    _ = request
    log.info("api.summarize.start place_id=%s", req.place_id)
    place = get_place_details(req.place_id)
    if not place:
        log.warning("api.summarize.not_found place_id=%s", req.place_id)
        raise HTTPException(status_code=404, detail="Place not found")
    summary = summarize_place(place)
    log.info("api.summarize.done place_id=%s name=%r", place.place_id, place.name)
    return SummaryResponse(place_id=place.place_id, name=place.name, summary=summary)


@app.post("/api/recommend", response_model=RecommendResponse)
@limiter.limit("10/minute")
def recommend(request: Request, req: PlaceIdRequest):
    """Get item recommendations for a specific place by ID."""
    _ = request
    log.info("api.recommend.start place_id=%s", req.place_id)
    place = get_place_details(req.place_id)
    if not place:
        log.warning("api.recommend.not_found place_id=%s", req.place_id)
        raise HTTPException(status_code=404, detail="Place not found")
    result = recommend_items(place)
    log.info(
        "api.recommend.done place_id=%s recommendations=%d avoid=%d",
        place.place_id,
        len(result.get("recommendations", [])),
        len(result.get("avoid", [])),
    )
    return RecommendResponse(**result)


@app.get("/api/geocode", response_model=GeocodeResponse)
@limiter.limit("30/minute")
def geocode(request: Request, address: str = Query(min_length=1, max_length=500)):
    """Server-side geocoding to avoid exposing Maps API key to browser.

    Results (including misses) are cached — geocoding is billed per
    request and "Richmond, VA" doesn't move.
    """
    _ = request
    cache_key = ("fwd", address.strip().lower())
    cached = geocode_cache.get(cache_key)
    if cached is not MISS:
        log.info("api.geocode.cache_hit address=%r", address)
        return cached

    log.info("api.geocode.start address=%r", address)
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    try:
        resp = httpx.get(
            url,
            params={"address": address, "key": get_api_key()},
            timeout=5,
        )
        resp.raise_for_status()
    except httpx.TimeoutException as exc:
        log.warning("api.geocode.timeout address=%r", address)
        raise HTTPException(status_code=504, detail="Geocoding timed out") from exc
    except httpx.HTTPError as exc:
        log.warning("api.geocode.error address=%r error=%s", address, exc)
        raise HTTPException(status_code=502, detail="Geocoding failed") from exc

    data = resp.json()
    if data.get("results"):
        loc = data["results"][0]["geometry"]["location"]
        result = {
            "lat": loc["lat"],
            "lng": loc["lng"],
            "name": data["results"][0]["formatted_address"],
        }
        log.info("api.geocode.done address=%r found=true", address)
        geocode_cache.set(cache_key, result)
        return result

    log.info("api.geocode.done address=%r found=false", address)
    result = {"lat": None, "lng": None, "name": None}
    geocode_cache.set(cache_key, result)
    return result


@app.get("/api/reverse-geocode", response_model=ReverseGeocodeResponse)
@limiter.limit("30/minute")
def reverse_geocode(request: Request, lat: float, lng: float):
    """Reverse geocode coordinates to a human-readable name.

    Cached with coordinates rounded to 3 decimals (~110m) — the result
    is a locality/neighborhood label, so nearby points share it. This
    also collapses the billed call fired on every page load when the
    browser grants geolocation.
    """
    _ = request
    cache_key = ("rev", round(lat, 3), round(lng, 3))
    cached = geocode_cache.get(cache_key)
    if cached is not MISS:
        log.info("api.reverse_geocode.cache_hit lat=%.6f lng=%.6f", lat, lng)
        return cached

    log.info("api.reverse_geocode.start lat=%.6f lng=%.6f", lat, lng)
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    try:
        resp = httpx.get(
            url,
            params={"latlng": f"{lat},{lng}", "key": get_api_key()},
            timeout=5,
        )
        resp.raise_for_status()
    except httpx.TimeoutException as exc:
        log.warning("api.reverse_geocode.timeout lat=%.6f lng=%.6f", lat, lng)
        raise HTTPException(
            status_code=504, detail="Reverse geocoding timed out"
        ) from exc
    except httpx.HTTPError as exc:
        log.warning(
            "api.reverse_geocode.error lat=%.6f lng=%.6f error=%s", lat, lng, exc
        )
        raise HTTPException(status_code=502, detail="Reverse geocoding failed") from exc

    data = resp.json()
    if data.get("results"):
        short = next(
            (
                r
                for r in data["results"]
                if any(
                    t in r.get("types", [])
                    for t in ["locality", "neighborhood", "sublocality"]
                )
            ),
            data["results"][0],
        )
        log.info("api.reverse_geocode.done lat=%.6f lng=%.6f found=true", lat, lng)
        result = {"name": short["formatted_address"]}
        geocode_cache.set(cache_key, result)
        return result

    fallback = f"{lat:.3f}, {lng:.3f}"
    log.info("api.reverse_geocode.done lat=%.6f lng=%.6f found=false", lat, lng)
    result = {"name": fallback}
    geocode_cache.set(cache_key, result)
    return result


# ── Health ──


@app.get("/health")
def health():
    """Unauthenticated liveness probe for Railway."""
    return {"status": "ok"}


@app.get("/api/stats")
def cache_stats():
    """Cache hit/miss counters — a quick read on how much API spend the
    caches are absorbing. Auth-protected like every other endpoint."""
    return {"caches": all_stats()}


# ── Static files ──

# Resolve paths relative to project root, not cwd
_project_root = Path(__file__).resolve().parent.parent.parent
_frontend_dir = _project_root / "frontend"

if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(_frontend_dir / "index.html"))
