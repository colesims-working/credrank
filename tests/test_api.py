"""Tests for the FastAPI endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from credrank.api import app
from credrank.models import (
    GridCoverage,
    GridSearchResult,
    PlaceResult,
    Review,
    SearchIntent,
)

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


@patch("credrank.api.build_intent", return_value=SearchIntent())
@patch("credrank.api.search_places_with_intent", return_value=FAKE_PLACES)
def test_search_returns_ranked_results(mock_search, mock_intent):
    _ = (mock_search, mock_intent)
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


@patch("credrank.api.build_intent", return_value=SearchIntent())
@patch("credrank.api.search_places_with_intent", return_value=[])
def test_search_empty_results(mock_search, mock_intent):
    _ = (mock_search, mock_intent)
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


@patch("credrank.api.build_intent", return_value=SearchIntent())
def test_search_validates_required_fields(mock_intent):
    _ = mock_intent
    resp = client.post("/api/search", json={"lat": 37.54, "lng": -77.43})
    assert resp.status_code == 422


@patch("credrank.api.build_intent", return_value=SearchIntent())
def test_search_validates_lat_bounds(mock_intent):
    _ = mock_intent
    resp = client.post(
        "/api/search",
        json={
            "query": "pizza",
            "lat": 91.0,
            "lng": -77.43,
        },
    )
    assert resp.status_code == 422


@patch("credrank.api.build_intent", return_value=SearchIntent())
def test_search_validates_radius_cap(mock_intent):
    _ = mock_intent
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


# ── Place ID Validation ──


def test_place_id_rejects_path_traversal():
    resp = client.post("/api/summarize", json={"place_id": "../../etc/passwd"})
    assert resp.status_code == 422


def test_place_id_rejects_empty():
    resp = client.post("/api/summarize", json={"place_id": ""})
    assert resp.status_code == 422


# ── Deep Search ──


FAKE_COMPLETE_COVERAGE = GridCoverage(
    complete=True,
    reason=None,
    cells_searched=7,
    cells_failed=0,
    cells_saturated=0,
    raw_count=3,
    deduped_count=3,
    in_radius_count=3,
)

FAKE_INCOMPLETE_COVERAGE = GridCoverage(
    complete=False,
    reason="2 saturated cell(s) reached minimum radius",
    cells_searched=15,
    cells_failed=0,
    cells_saturated=2,
    raw_count=50,
    deduped_count=3,
    in_radius_count=3,
)


@patch("credrank.api.build_intent", return_value=SearchIntent())
@patch(
    "credrank.api.grid_search",
    return_value=GridSearchResult(FAKE_PLACES, coverage=FAKE_COMPLETE_COVERAGE),
)
def test_deep_search_returns_complete_coverage(mock_grid, mock_intent):
    _ = (mock_grid, mock_intent)
    resp = client.post(
        "/api/search",
        json={
            "query": "pizza",
            "lat": 37.54,
            "lng": -77.43,
            "radius_m": 8000,
            "deep_search": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["deep_search"] is True
    assert data["result_count"] == 3
    assert data["coverage"] is not None
    assert data["coverage"]["complete"] is True
    assert data["coverage"]["cells_searched"] == 7
    assert data["coverage"]["cells_saturated"] == 0
    assert data["coverage"]["reason"] is None


@patch("credrank.api.build_intent", return_value=SearchIntent())
@patch(
    "credrank.api.grid_search",
    return_value=GridSearchResult(FAKE_PLACES, coverage=FAKE_INCOMPLETE_COVERAGE),
)
def test_deep_search_surfaces_incomplete_coverage(mock_grid, mock_intent):
    _ = (mock_grid, mock_intent)
    resp = client.post(
        "/api/search",
        json={
            "query": "pizza",
            "lat": 37.54,
            "lng": -77.43,
            "radius_m": 8000,
            "deep_search": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["coverage"]["complete"] is False
    assert "minimum radius" in data["coverage"]["reason"]
    assert data["coverage"]["cells_saturated"] == 2
    assert data["coverage"]["cells_searched"] == 15


@patch("credrank.api.build_intent", return_value=SearchIntent())
@patch("credrank.api.search_places_with_intent", return_value=FAKE_PLACES)
def test_non_deep_search_has_no_coverage(mock_search, mock_intent):
    _ = (mock_search, mock_intent)
    resp = client.post(
        "/api/search",
        json={
            "query": "pizza",
            "lat": 37.54,
            "lng": -77.43,
            "radius_m": 8000,
            "deep_search": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["deep_search"] is False
    assert data["coverage"] is None


# ── Geocode ──


@patch("credrank.api.httpx.get")
def test_geocode_returns_location(mock_get):
    mock_get.return_value = type(
        "Resp",
        (),
        {
            "json": lambda self: {
                "results": [
                    {
                        "geometry": {"location": {"lat": 37.54, "lng": -77.43}},
                        "formatted_address": "Richmond, VA",
                    }
                ]
            },
            "raise_for_status": lambda self: None,
        },
    )()
    resp = client.get("/api/geocode", params={"address": "Richmond"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["lat"] == 37.54
    assert data["lng"] == -77.43
    assert data["name"] == "Richmond, VA"


@patch("credrank.api.httpx.get")
def test_geocode_returns_null_on_empty_results(mock_get):
    mock_get.return_value = type(
        "Resp",
        (),
        {
            "json": lambda self: {"results": []},
            "raise_for_status": lambda self: None,
        },
    )()
    resp = client.get("/api/geocode", params={"address": "xyzzy nowhere"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["lat"] is None


def test_geocode_validates_address_length():
    resp = client.get("/api/geocode", params={"address": ""})
    assert resp.status_code == 422


@patch("credrank.api.httpx.get")
def test_reverse_geocode_returns_name(mock_get):
    mock_get.return_value = type(
        "Resp",
        (),
        {
            "json": lambda self: {
                "results": [
                    {"formatted_address": "Richmond, VA", "types": ["locality"]}
                ]
            },
            "raise_for_status": lambda self: None,
        },
    )()
    resp = client.get("/api/reverse-geocode", params={"lat": 37.54, "lng": -77.43})
    assert resp.status_code == 200
    assert resp.json()["name"] == "Richmond, VA"


@patch("credrank.api.httpx.get")
def test_reverse_geocode_falls_back_to_coords(mock_get):
    mock_get.return_value = type(
        "Resp",
        (),
        {
            "json": lambda self: {"results": []},
            "raise_for_status": lambda self: None,
        },
    )()
    resp = client.get("/api/reverse-geocode", params={"lat": 37.54, "lng": -77.43})
    assert resp.status_code == 200
    assert resp.json()["name"] == "37.540, -77.430"
