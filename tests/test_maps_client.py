import httpx
import respx

from credrank.maps_client import haversine_m, search_places
from credrank.models import PlaceSearchParams


def test_haversine_known_distance() -> None:
    """Verify haversine against a known distance (NYC to LA ≈ 3,936 km)."""
    dist = haversine_m(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3_930_000 < dist < 3_945_000


def test_haversine_same_point() -> None:
    assert haversine_m(37.54, -77.43, 37.54, -77.43) == 0.0


FAKE_RESPONSE = {
    "places": [
        {
            "id": "ChIJ_fake_1",
            "displayName": {"text": "Pizza Palace", "languageCode": "en"},
            "rating": 4.5,
            "userRatingCount": 312,
            "formattedAddress": "123 Main St, Richmond, VA 23220",
            "location": {"latitude": 37.54, "longitude": -77.43},
            "types": ["pizza_restaurant", "restaurant"],
        },
        {
            "id": "ChIJ_fake_2",
            "displayName": {"text": "Slice House", "languageCode": "en"},
            "rating": 4.2,
            "userRatingCount": 189,
            "formattedAddress": "456 Broad St, Richmond, VA 23220",
            "location": {"latitude": 37.55, "longitude": -77.44},
            "types": ["pizza_restaurant", "meal_delivery"],
        },
    ]
}

PARAMS = PlaceSearchParams(
    query="pizza",
    location=(37.54, -77.43),
    radius_m=5000,
    max_results=10,
)


@respx.mock
def test_search_places_maps_fields_correctly() -> None:
    """Verify that raw API response fields map to our PlaceResult model."""
    respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        return_value=httpx.Response(200, json=FAKE_RESPONSE)
    )

    results = search_places(PARAMS)

    assert len(results) == 2

    first = results[0]
    assert first.place_id == "ChIJ_fake_1"
    assert first.name == "Pizza Palace"
    assert first.rating == 4.5
    assert first.user_rating_count == 312
    assert first.address == "123 Main St, Richmond, VA 23220"
    assert first.lat == 37.54
    assert first.lng == -77.43
    assert "pizza_restaurant" in first.types

    second = results[1]
    assert second.place_id == "ChIJ_fake_2"
    assert second.name == "Slice House"


@respx.mock
def test_search_places_handles_empty_response() -> None:
    """Verify that an empty API response returns an empty list."""
    respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        return_value=httpx.Response(200, json={})
    )

    results = search_places(PARAMS)

    assert results == []


@respx.mock
def test_search_places_handles_pagination() -> None:
    """Verify that pagination aggregates results across pages."""
    page_1 = {
        "places": [
            {
                "id": "ChIJ_page1",
                "displayName": {"text": "Page One Pizza"},
                "formattedAddress": "111 First St",
            }
        ],
        "nextPageToken": "fake_token_abc",
    }
    page_2 = {
        "places": [
            {
                "id": "ChIJ_page2",
                "displayName": {"text": "Page Two Pizza"},
                "formattedAddress": "222 Second St",
            }
        ],
    }

    route = respx.post("https://places.googleapis.com/v1/places:searchText")
    route.side_effect = [
        httpx.Response(200, json=page_1),
        httpx.Response(200, json=page_2),
    ]

    results = search_places(PARAMS)

    assert len(results) == 2
    assert results[0].place_id == "ChIJ_page1"
    assert results[1].place_id == "ChIJ_page2"
