"""Tests for the FastAPI endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from credrank.api import app
from credrank.models import PlaceResult

client = TestClient(app)

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


@patch("credrank.api.search_places", return_value=FAKE_PLACES)
def test_search_returns_ranked_results(mock_search):
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

    # Verify ranked by CredScore descending
    scores = [p["credscore"] for p in data["places"]]
    assert scores == sorted(scores, reverse=True)

    # Verify all CredScores are bounded
    for p in data["places"]:
        assert 0.0 <= p["credscore"] <= 100.0


@patch("credrank.api.search_places", return_value=[])
def test_search_empty_results(mock_search):
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


def test_search_validates_input():
    resp = client.post("/api/search", json={"lat": 37.54, "lng": -77.43})
    assert resp.status_code == 422  # Missing required field 'query'
