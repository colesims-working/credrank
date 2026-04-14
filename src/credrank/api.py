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
from credrank.config import get_api_key
from credrank.grid_search import generate_subcircles, grid_search
from credrank.maps_client import get_place_details, search_places
from credrank.models import PlaceSearchParams
from credrank.ranking import rank_places
from credrank.recommender import recommend_items
from credrank.relevance import filter_by_relevance, get_relevant_types
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


class SearchResponse(BaseModel):
    query: str
    result_count: int
    filtered_count: int = 0
    deep_search: bool = False
    subcircle_count: int = 0
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

    # 1) LLM relevance filter (one call per search)
    relevant_types = get_relevant_types(req.query)
    log.debug(
        "api.search.relevance query=%r relevant_type_count=%d",
        req.query,
        len(relevant_types or []),
    )

    # 2) Search — standard or grid
    subcircle_count = 0
    if req.deep_search:
        subcircles = generate_subcircles(req.lat, req.lng, req.radius_m)
        subcircle_count = len(subcircles)
        log.debug(
            "api.search.subcircles query=%r count=%d",
            req.query,
            subcircle_count,
        )
        places = grid_search(
            query=req.query,
            center_lat=req.lat,
            center_lng=req.lng,
            radius_m=req.radius_m,
            max_results=req.max_results,
        )
    else:
        params = PlaceSearchParams(
            query=req.query,
            location=(req.lat, req.lng),
            radius_m=req.radius_m,
            max_results=min(req.max_results, 60),
        )
        places = search_places(params)

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
            subcircle_count=subcircle_count,
            places=[],
        )

    # 3) Filter by relevance
    raw_count = len(places)
    if relevant_types:
        places = filter_by_relevance(places, relevant_types)
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
            subcircle_count=subcircle_count,
            places=[],
        )

    # 4) Rank
    ranked = rank_places(places)

    results = []
    for entry in ranked:
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
        "api.search.done query=%r result_count=%d filtered_count=%d deep_search=%s subcircle_count=%d duration_ms=%d",
        req.query,
        len(results),
        filtered_count,
        req.deep_search,
        subcircle_count,
        duration_ms,
    )
    return SearchResponse(
        query=req.query,
        result_count=len(results),
        filtered_count=filtered_count,
        deep_search=req.deep_search,
        subcircle_count=subcircle_count,
        places=results,
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
    """Server-side geocoding to avoid exposing Maps API key to browser."""
    _ = request
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
        return result

    log.info("api.geocode.done address=%r found=false", address)
    return {"lat": None, "lng": None, "name": None}


@app.get("/api/reverse-geocode", response_model=ReverseGeocodeResponse)
@limiter.limit("30/minute")
def reverse_geocode(request: Request, lat: float, lng: float):
    """Reverse geocode coordinates to a human-readable name."""
    _ = request
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
        return {"name": short["formatted_address"]}

    fallback = f"{lat:.3f}, {lng:.3f}"
    log.info("api.reverse_geocode.done lat=%.6f lng=%.6f found=false", lat, lng)
    return {"name": fallback}


# ── Health ──


@app.get("/health")
def health():
    """Unauthenticated liveness probe for Railway."""
    return {"status": "ok"}


# ── Static files ──

# Resolve paths relative to project root, not cwd
_project_root = Path(__file__).resolve().parent.parent.parent
_frontend_dir = _project_root / "frontend"

if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(str(_frontend_dir / "index.html"))
