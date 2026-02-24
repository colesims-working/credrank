"""Tests for the FastAPI endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from credrank.api import app
from credrank.models import PlaceResult, Review

client = TestClient(app)
FAKE_RECOMMEND = {
    "place_name": "Test",
    "place_id": "123",
    "summary": "Good",
    "recommendations": [],
    "avoid": [],
}

FAKE_PLACES = [
    PlaceResult(
        place_id="ChIJ_1",
        name="Top Spot",
        rating=4.7,
        user_rating_count=500,
        address="123 Main St",
        lat=37.54,
        lng=-77.43,
        types=["restaurant"],
    ),
    PlaceResult(
        place_id="ChIJ_2",
        name="Mid Spot",
        rating=4.3,
        user_rating_count=200,
        address="456 Broad St",
        lat=37.55,
        lng=-77.44,
        types=["restaurant"],
    ),
    PlaceResult(
        place_id="ChIJ_3",
        name="Low Spot",
        rating=3.8,
        user_rating_count=50,
        address="789 Oak Ave",
        lat=37.56,
        lng=-77.45,
        types=["restaurant"],
    ),
]

FAKE_DETAIL = PlaceResult(
    place_id="ChIJ_1",
    name="Top Spot",
    rating=4.7,
    user_rating_count=500,
    address="123 Main St",
    lat=37.54,
    lng=-77.43,
    types=["restaurant"],
    reviews=[Review(rating=5, text="Amazing pizza!", author="Alice")],
    editorial_summary="A local favorite.",
)


@patch("credrank.api.get_relevant_types", return_value=None)
@patch("credrank.api.search_places", return_value=FAKE_PLACES)
def test_search_returns_ranked_results(mock_search, mock_relevant):
    _ = (mock_search, mock_relevant)
    resp = client.post(
        "/api/search",
        json={
            "query": "pizza",
            "lat": 37.54,
            "lng": -77.43,
            "radius_m": 5000,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["result_count"] == 3
    assert data["places"][0]["rank"] == 1

    scores = [p["credscore"] for p in data["places"]]
    assert scores == sorted(scores, reverse=True)

    for p in data["places"]:
        assert 0.0 <= p["credscore"] <= 100.0


@patch("credrank.api.get_relevant_types", return_value=None)
@patch("credrank.api.search_places", return_value=[])
def test_search_empty_results(mock_search, mock_relevant):
    _ = (mock_search, mock_relevant)
    resp = client.post(
        "/api/search",
        json={
            "query": "nonexistent",
            "lat": 37.54,
            "lng": -77.43,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["result_count"] == 0
    assert data["places"] == []


@patch("credrank.api.get_relevant_types", return_value=None)
def test_search_validates_required_fields(mock_relevant):
    _ = mock_relevant
    resp = client.post("/api/search", json={"lat": 37.54, "lng": -77.43})
    assert resp.status_code == 422


@patch("credrank.api.get_relevant_types", return_value=None)
def test_search_validates_lat_bounds(mock_relevant):
    _ = mock_relevant
    resp = client.post(
        "/api/search",
        json={
            "query": "pizza",
            "lat": 91.0,
            "lng": -77.43,
        },
    )
    assert resp.status_code == 422


@patch("credrank.api.get_relevant_types", return_value=None)
def test_search_validates_radius_cap(mock_relevant):
    _ = mock_relevant
    resp = client.post(
        "/api/search",
        json={
            "query": "pizza",
            "lat": 37.54,
            "lng": -77.43,
            "radius_m": 100000,
        },
    )
    assert resp.status_code == 422


@patch("credrank.api.summarize_place", return_value="Great place")
@patch("credrank.api.get_place_details", return_value=FAKE_DETAIL)
def test_summarize_returns_summary(mock_details, mock_summarize):
    _ = (mock_details, mock_summarize)
    resp = client.post("/api/summarize", json={"place_id": "ChIJ_1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["place_id"] == "ChIJ_1"
    assert data["name"] == "Top Spot"
    assert data["summary"] == "Great place"


@patch("credrank.api.summarize_place", return_value="Great place")
@patch("credrank.api.get_place_details", return_value=None)
def test_summarize_place_not_found(mock_details, mock_summarize):
    _ = (mock_details, mock_summarize)
    resp = client.post("/api/summarize", json={"place_id": "nonexistent"})
    assert resp.status_code == 404


@patch("credrank.api.recommend_items", return_value=FAKE_RECOMMEND)
@patch("credrank.api.get_place_details", return_value=FAKE_DETAIL)
def test_recommend_returns_analysis(mock_details, mock_recommend):
    _ = (mock_details, mock_recommend)
    resp = client.post("/api/recommend", json={"place_id": "ChIJ_1"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["place_id"] == "123"
    assert "recommendations" in data
    assert "avoid" in data
    assert data["summary"] == "Good"


@patch("credrank.api.recommend_items", return_value=FAKE_RECOMMEND)
@patch("credrank.api.get_place_details", return_value=None)
def test_recommend_place_not_found(mock_details, mock_recommend):
    _ = (mock_details, mock_recommend)
    resp = client.post("/api/recommend", json={"place_id": "nonexistent"})
    assert resp.status_code == 404
