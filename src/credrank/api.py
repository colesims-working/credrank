"""FastAPI backend for CredRank."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from credrank.maps_client import search_places
from credrank.models import PlaceSearchParams
from credrank.ranking import rank_places
from credrank.recommender import recommend_items
from credrank.summarizer import summarize_place

app = FastAPI(title="CredRank", version="0.1.0")

# Allow the frontend to talk to the backend during local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---


class SearchRequest(BaseModel):
    query: str
    lat: float
    lng: float
    radius_m: int = 8000
    max_results: int = 60


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


class SearchResponse(BaseModel):
    query: str
    result_count: int
    places: list[PlaceResponse]


class RecommendResponse(BaseModel):
    place_name: str
    place_id: str
    summary: str
    recommendations: list[dict]
    avoid: list[dict]


# --- Endpoints ---


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Search, rank, and return places with CredScores."""
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


@app.post("/api/summarize/{place_rank}")
def summarize(place_rank: int, req: SearchRequest):
    """Generate an LLM summary for a specific ranked place."""
    params = PlaceSearchParams(
        query=req.query,
        location=(req.lat, req.lng),
        radius_m=req.radius_m,
        max_results=req.max_results,
    )

    places = search_places(params)
    ranked = rank_places(places)

    for entry in ranked:
        if entry["rank"] == place_rank:
            summary = summarize_place(entry["place"])
            return {"rank": place_rank, "summary": summary}

    raise HTTPException(status_code=404, detail=f"No place at rank {place_rank}")


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: SearchRequest):
    """Get item recommendations for the top result of a search."""
    params = PlaceSearchParams(
        query=req.query,
        location=(req.lat, req.lng),
        radius_m=req.radius_m,
        max_results=1,
    )

    places = search_places(params)
    if not places:
        raise HTTPException(status_code=404, detail="Place not found")

    result = recommend_items(places[0])
    return RecommendResponse(**result)


# --- Static files (frontend) ---

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")
