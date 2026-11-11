import json
from unittest.mock import patch

import httpx
import respx

from credrank.maps_client import (
    generative_summary_text,
    haversine_m,
    localized_text,
    review_summary_text,
    search_places,
    search_places_with_intent,
)
from credrank.models import PlaceSearchParams, PlacesSearchResult, SearchIntent


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
            "primaryType": "pizza_restaurant",
        },
        {
            "id": "ChIJ_fake_2",
            "displayName": {"text": "Slice House", "languageCode": "en"},
            "rating": 4.2,
            "userRatingCount": 189,
            "formattedAddress": "456 Broad St, Richmond, VA 23220",
            "location": {"latitude": 37.55, "longitude": -77.44},
            "types": ["pizza_restaurant", "meal_delivery"],
            "primaryType": "pizza_restaurant",
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

    result = search_places(PARAMS)

    assert isinstance(result, PlacesSearchResult)
    assert len(result) == 2
    assert result.saturated is False
    assert result.page_count == 1

    first = result[0]
    assert first.place_id == "ChIJ_fake_1"
    assert first.name == "Pizza Palace"
    assert first.rating == 4.5
    assert first.user_rating_count == 312
    assert first.address == "123 Main St, Richmond, VA 23220"
    assert first.lat == 37.54
    assert first.lng == -77.43
    assert "pizza_restaurant" in first.types
    assert first.primary_type == "pizza_restaurant"

    second = result[1]
    assert second.place_id == "ChIJ_fake_2"
    assert second.name == "Slice House"
    assert second.primary_type == "pizza_restaurant"


@respx.mock
def test_search_places_request_body_uses_pagesize() -> None:
    """Request body must use pageSize=20 and not the deprecated maxResultCount."""
    route = respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        return_value=httpx.Response(200, json={"places": []})
    )

    search_places(PARAMS)

    assert route.called
    request_body = json.loads(route.calls[0].request.content)
    assert request_body["pageSize"] == 20
    assert "maxResultCount" not in request_body
    assert request_body["textQuery"] == "pizza"


@respx.mock
def test_search_places_handles_empty_response() -> None:
    """Verify that an empty API response returns an empty result."""
    respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        return_value=httpx.Response(200, json={})
    )

    result = search_places(PARAMS)
    assert len(result) == 0
    assert result.saturated is False
    assert result.page_count == 1


@respx.mock
@patch("credrank.maps_client.time.sleep")
def test_search_places_follows_next_page_token(mock_sleep) -> None:
    """Pagination walks nextPageToken until exhausted and aggregates across pages."""
    _ = mock_sleep
    page_1 = {
        "places": [
            {
                "id": "ChIJ_page1",
                "displayName": {"text": "Page One Pizza"},
                "formattedAddress": "111 First St",
                "location": {"latitude": 37.54, "longitude": -77.43},
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
                "location": {"latitude": 37.54, "longitude": -77.43},
            }
        ],
    }

    route = respx.post("https://places.googleapis.com/v1/places:searchText")
    route.side_effect = [
        httpx.Response(200, json=page_1),
        httpx.Response(200, json=page_2),
    ]

    result = search_places(PARAMS)

    assert len(result) == 2
    assert result[0].place_id == "ChIJ_page1"
    assert result[1].place_id == "ChIJ_page2"
    assert result.page_count == 2
    assert result.saturated is False

    # The second request must carry the pageToken from the first response.
    second_body = json.loads(route.calls[1].request.content)
    assert second_body["pageToken"] == "fake_token_abc"


@respx.mock
@patch("credrank.maps_client.time.sleep")
def test_search_saturated_when_page_cap_reached(mock_sleep) -> None:
    """Three pages, each still advertising a nextPageToken -> saturated=True."""
    _ = mock_sleep

    def make_page(i: int) -> dict:
        return {
            "places": [
                {
                    "id": f"ChIJ_pg{i}",
                    "displayName": {"text": f"Place {i}"},
                    "location": {"latitude": 37.54, "longitude": -77.43},
                }
            ],
            "nextPageToken": f"token_{i}",
        }

    route = respx.post("https://places.googleapis.com/v1/places:searchText")
    route.side_effect = [
        httpx.Response(200, json=make_page(1)),
        httpx.Response(200, json=make_page(2)),
        httpx.Response(200, json=make_page(3)),  # page cap hits after this
    ]

    result = search_places(PARAMS)

    assert len(result) == 3
    assert result.page_count == 3
    assert result.saturated is True


@respx.mock
@patch("credrank.maps_client.time.sleep")
def test_search_saturated_when_google_silently_caps_at_60(mock_sleep) -> None:
    """Regression for the Aruba 'Breakfast' bug.

    Google Text Search (New) caps at roughly 60 results (3 pages of 20)
    and often returns NO nextPageToken on the final page even when more
    results exist — we observed cells returning raw_count=60 page_count=3
    with saturated=False in production logs, which prevented the adaptive
    grid from subdividing dense cells.

    A fully-maxed response (every page used, 60 total, no final token)
    must now be treated as saturated.
    """
    _ = mock_sleep

    def make_page(idx: int, has_next: bool) -> dict:
        return {
            "places": [
                {
                    "id": f"ChIJ_p{idx}_{j}",
                    "displayName": {"text": f"Place {idx}.{j}"},
                    "location": {"latitude": 12.5, "longitude": -70.0},
                }
                for j in range(20)
            ],
            **({"nextPageToken": f"token_{idx}"} if has_next else {}),
        }

    route = respx.post("https://places.googleapis.com/v1/places:searchText")
    route.side_effect = [
        httpx.Response(200, json=make_page(0, has_next=True)),
        httpx.Response(200, json=make_page(1, has_next=True)),
        # Google's 3rd page: 20 more places but NO nextPageToken, even
        # though there are still more places in the area.
        httpx.Response(200, json=make_page(2, has_next=False)),
    ]

    params = PlaceSearchParams(
        query="breakfast",
        location=(12.5, -70.0),
        radius_m=8000,
        max_results=60,
    )
    result = search_places(params)

    assert len(result) == 60
    assert result.page_count == 3
    # The regression: saturated must be True even though the final page
    # had no token.
    assert result.saturated is True


@respx.mock
@patch("credrank.maps_client.time.sleep")
def test_search_saturated_when_max_results_reached(mock_sleep) -> None:
    """Hit max_results while a nextPageToken is still outstanding -> saturated=True."""
    _ = mock_sleep
    params = PlaceSearchParams(
        query="pizza",
        location=(37.54, -77.43),
        radius_m=5000,
        max_results=2,  # small cap
    )
    page = {
        "places": [
            {
                "id": "ChIJ_a",
                "displayName": {"text": "A"},
                "location": {"latitude": 37.54, "longitude": -77.43},
            },
            {
                "id": "ChIJ_b",
                "displayName": {"text": "B"},
                "location": {"latitude": 37.54, "longitude": -77.43},
            },
            {
                "id": "ChIJ_c",
                "displayName": {"text": "C"},
                "location": {"latitude": 37.54, "longitude": -77.43},
            },
        ],
        "nextPageToken": "token_xyz",
    }

    respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        return_value=httpx.Response(200, json=page)
    )

    result = search_places(params)

    assert len(result) == 2
    assert result.page_count == 1
    assert result.saturated is True


# ── LocalizedText helpers ──


def test_localized_text_extracts_string():
    assert localized_text({"text": "hello", "languageCode": "en"}) == "hello"


def test_localized_text_returns_none_for_malformed_input():
    assert localized_text({}) is None
    assert localized_text({"text": ""}) is None
    assert localized_text({"text": 42}) is None
    assert localized_text(None) is None
    assert localized_text("raw string") is None


def test_review_summary_text_extracts_nested_localized_text():
    place = {
        "reviewSummary": {
            "text": {"text": "A summary of reviews.", "languageCode": "en"}
        }
    }
    assert review_summary_text(place) == "A summary of reviews."


def test_review_summary_text_returns_none_on_missing_or_malformed():
    assert review_summary_text({}) is None
    assert review_summary_text({"reviewSummary": None}) is None
    assert review_summary_text({"reviewSummary": {}}) is None
    # A bare string at .text (instead of nested LocalizedText) is rejected —
    # this was the original bug.
    assert review_summary_text({"reviewSummary": {"text": "plain string"}}) is None


def test_generative_summary_text_extracts_from_overview():
    place = {
        "generativeSummary": {
            "overview": {"text": "Generated overview.", "languageCode": "en"}
        }
    }
    assert generative_summary_text(place) == "Generated overview."


def test_generative_summary_text_rejects_deprecated_text_field():
    """The old code read generativeSummary.text — that shape is wrong."""
    place = {"generativeSummary": {"text": "old shape"}}
    assert generative_summary_text(place) is None


def test_generative_summary_text_returns_none_on_missing():
    assert generative_summary_text({}) is None
    assert generative_summary_text({"generativeSummary": None}) is None
    assert generative_summary_text({"generativeSummary": {}}) is None


# ── Integration: nested LocalizedText through search_places -> _parse_place ──


# ── includedType + strictTypeFiltering server-side filtering ──


@respx.mock
def test_search_passes_included_type_and_strict_filter() -> None:
    """When PlaceSearchParams carries includedType + strict flag, both
    must appear in the outgoing request body."""
    route = respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        return_value=httpx.Response(200, json={"places": []})
    )
    params = PlaceSearchParams(
        query="pizza",
        location=(37.54, -77.43),
        radius_m=5000,
        max_results=10,
        included_type="pizza_restaurant",
        strict_type_filtering=True,
    )
    search_places(params)

    body = json.loads(route.calls[0].request.content)
    assert body["includedType"] == "pizza_restaurant"
    assert body["strictTypeFiltering"] is True


@respx.mock
def test_search_omits_included_type_when_not_set() -> None:
    """Default PlaceSearchParams must NOT include includedType."""
    route = respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        return_value=httpx.Response(200, json={"places": []})
    )
    params = PlaceSearchParams(
        query="food near me",
        location=(37.54, -77.43),
        radius_m=5000,
        max_results=10,
    )
    search_places(params)

    body = json.loads(route.calls[0].request.content)
    assert "includedType" not in body
    assert "strictTypeFiltering" not in body


# ── search_places_with_intent dispatch ──


@respx.mock
def test_with_intent_no_types_does_broad_search() -> None:
    """Intent with empty included_types -> broad search, no includedType."""
    route = respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        return_value=httpx.Response(200, json={"places": []})
    )
    params = PlaceSearchParams(
        query="best food",
        location=(37.54, -77.43),
        radius_m=5000,
        max_results=10,
    )
    intent = SearchIntent(query_kind="general")
    search_places_with_intent(params, intent)

    assert route.call_count == 1
    body = json.loads(route.calls[0].request.content)
    assert "includedType" not in body


@respx.mock
def test_with_intent_single_type_pushes_to_server() -> None:
    """Intent with exactly one Table A type -> one search with includedType."""
    route = respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        return_value=httpx.Response(200, json={"places": []})
    )
    params = PlaceSearchParams(
        query="pizza",
        location=(37.54, -77.43),
        radius_m=5000,
        max_results=10,
    )
    intent = SearchIntent(
        query_kind="specific_food",
        included_types=["pizza_restaurant"],
    )
    search_places_with_intent(params, intent)

    assert route.call_count == 1
    body = json.loads(route.calls[0].request.content)
    assert body["includedType"] == "pizza_restaurant"
    assert body["strictTypeFiltering"] is True


@respx.mock
@patch("credrank.maps_client.time.sleep")
def test_with_intent_multiple_types_runs_one_request_per_type(mock_sleep) -> None:
    """Intent with >1 Table A types -> one search per type, merged by place_id."""
    _ = mock_sleep
    call_bodies: list[dict] = []

    def _responder(request):
        call_bodies.append(json.loads(request.content))
        body = json.loads(request.content)
        included = body.get("includedType", "")
        place_id = f"ChIJ_{included}"
        return httpx.Response(
            200,
            json={
                "places": [
                    {
                        "id": place_id,
                        "displayName": {"text": f"Place for {included}"},
                        "location": {"latitude": 37.54, "longitude": -77.43},
                        "primaryType": included,
                    }
                ]
            },
        )

    respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        side_effect=_responder
    )

    params = PlaceSearchParams(
        query="coffee",
        location=(37.54, -77.43),
        radius_m=5000,
        max_results=10,
    )
    intent = SearchIntent(
        query_kind="specific_food",
        included_types=["coffee_shop", "cafe"],
    )
    result = search_places_with_intent(params, intent)

    # One request per type.
    types_used = sorted(b["includedType"] for b in call_bodies)
    assert types_used == ["cafe", "coffee_shop"]
    # Results merged from both.
    place_ids = sorted(p.place_id for p in result)
    assert place_ids == ["ChIJ_cafe", "ChIJ_coffee_shop"]


@respx.mock
def test_with_intent_filters_non_table_a_types() -> None:
    """Intent containing a Table B type (pizza_delivery) must not pass it
    as includedType server-side — only pizza_restaurant goes to the wire."""
    route = respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        return_value=httpx.Response(200, json={"places": []})
    )
    params = PlaceSearchParams(
        query="pizza",
        location=(37.54, -77.43),
        radius_m=5000,
        max_results=10,
    )
    intent = SearchIntent(
        query_kind="specific_food",
        included_types=["pizza_restaurant", "pizza_delivery"],  # B is Table B
    )
    search_places_with_intent(params, intent)

    # Only one request was issued — for pizza_restaurant.
    assert route.call_count == 1
    body = json.loads(route.calls[0].request.content)
    assert body["includedType"] == "pizza_restaurant"


# ── Existing integration test moved below the new ones ──


@respx.mock
def test_parse_handles_nested_review_and_generative_summaries() -> None:
    """End-to-end: raw Google JSON with nested LocalizedText parses correctly."""
    response = {
        "places": [
            {
                "id": "ChIJ_nested",
                "displayName": {"text": "Nested Place"},
                "location": {"latitude": 37.54, "longitude": -77.43},
                "primaryType": "restaurant",
                "editorialSummary": {"text": "Editorial blurb.", "languageCode": "en"},
                "reviewSummary": {
                    "text": {"text": "Review summary blurb.", "languageCode": "en"}
                },
                "generativeSummary": {
                    "overview": {"text": "Gen overview blurb.", "languageCode": "en"}
                },
            }
        ]
    }
    respx.post("https://places.googleapis.com/v1/places:searchText").mock(
        return_value=httpx.Response(200, json=response)
    )

    result = search_places(PARAMS)
    assert len(result) == 1
    place = result[0]
    assert place.primary_type == "restaurant"
    assert place.editorial_summary == "Editorial blurb."
    assert place.review_summary == "Review summary blurb."
    assert place.generative_summary == "Gen overview blurb."
