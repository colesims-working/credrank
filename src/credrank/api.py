"""FastAPI backend for CredRank."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from credrank.maps_client import get_place_details, search_places
from credrank.models import PlaceSearchParams
from credrank.ranking import rank_places
from credrank.recommender import recommend_items
from credrank.summarizer import summarize_place

app = FastAPI(title="CredRank", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    radius_m: int = Field(default=8000, gt=0, le=50000)
    max_results: int = Field(default=20, gt=0, le=60)


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
    summary: str | None = None
    credible: bool = True


class SearchResponse(BaseModel):
    query: str
    result_count: int
    places: list[PlaceResponse]


class PlaceIdRequest(BaseModel):
    place_id: str = Field(min_length=1, max_length=500)


class SummaryResponse(BaseModel):
    place_id: str
    name: str
    summary: str


class RecommendResponse(BaseModel):
    place_name: str
    place_id: str
    summary: str
    recommendations: list[dict]
    avoid: list[dict]


# --- Endpoints ---


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Search, rank, and return places with CredScores.

    Uses the lightweight field mask — no reviews fetched, lower API cost.
    """
    params = PlaceSearchParams(
        query=req.query,
        location=(req.lat, req.lng),
        radius_m=req.radius_m,
        max_results=req.max_results,
    )

    places = search_places(params)
    if not places:
        return SearchResponse(query=req.query, result_count=0, places=[])

    ranked = rank_places(places)

    results = []
    for entry in ranked:
        p = entry["place"]
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

    return SearchResponse(
        query=req.query,
        result_count=len(results),
        places=results,
    )


@app.post("/api/summarize", response_model=SummaryResponse)
def summarize(req: PlaceIdRequest):
    """Generate an LLM summary for a specific place by ID.

    Fetches full details (reviews + summaries) for just this one place.
    """
    place = get_place_details(req.place_id)
    if not place:
        raise HTTPException(status_code=404, detail="Place not found")

    summary = summarize_place(place)
    return SummaryResponse(place_id=place.place_id, name=place.name, summary=summary)


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: PlaceIdRequest):
    """Get item recommendations for a specific place by ID.

    Fetches full details (reviews + summaries) for deep analysis.
    """
    place = get_place_details(req.place_id)
    if not place:
        raise HTTPException(status_code=404, detail="Place not found")

    result = recommend_items(place)
    return RecommendResponse(**result)


# --- Static files (frontend) ---

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")
