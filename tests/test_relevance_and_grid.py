"""Tests for relevance filtering and grid search."""

from unittest.mock import patch

from credrank.relevance import get_relevant_types, filter_by_relevance, ALL_TYPES
from credrank.grid_search import _generate_subcircles, grid_search
from credrank.models import PlaceResult


def _make_place(place_id: str, types: list[str], lat=37.54, lng=-77.43) -> PlaceResult:
    return PlaceResult(
        place_id=place_id, name=f"Place {place_id}",
        types=types, rating=4.0, user_rating_count=100,
        lat=lat, lng=lng,
    )


# ═══ Relevance Filtering ═══


class TestGetRelevantTypes:
    @patch("credrank.relevance.complete")
    def test_returns_parsed_types(self, mock_complete):
        mock_complete.return_value = '["pizza_restaurant", "italian_restaurant", "restaurant"]'
        result = get_relevant_types("best pizza")
        assert result is not None
        assert "pizza_restaurant" in result
        assert "restaurant" in result

    @patch("credrank.relevance.complete")
    def test_handles_markdown_fence(self, mock_complete):
        mock_complete.return_value = '```json\n["cafe", "coffee_shop"]\n```'
        result = get_relevant_types("coffee")
        assert result == ["cafe", "coffee_shop"]

    @patch("credrank.relevance.complete")
    def test_validates_against_taxonomy(self, mock_complete):
        mock_complete.return_value = '["pizza_restaurant", "fake_type_xyz", "restaurant"]'
        result = get_relevant_types("pizza")
        assert result is not None
        assert "pizza_restaurant" in result
        assert "restaurant" in result
        assert "fake_type_xyz" not in result

    @patch("credrank.relevance.complete")
    def test_returns_none_on_invalid_json(self, mock_complete):
        mock_complete.return_value = "I think pizza restaurants would be relevant"
        assert get_relevant_types("pizza") is None

    @patch("credrank.relevance.complete")
    def test_returns_none_on_exception(self, mock_complete):
        mock_complete.side_effect = Exception("API down")
        assert get_relevant_types("pizza") is None

    @patch("credrank.relevance.complete")
    def test_returns_none_on_empty_array(self, mock_complete):
        mock_complete.return_value = "[]"
        assert get_relevant_types("xyzzy") is None

    @patch("credrank.relevance.complete")
    def test_returns_none_on_all_invalid_types(self, mock_complete):
        mock_complete.return_value = '["not_a_type", "also_fake"]'
        assert get_relevant_types("pizza") is None

    @patch("credrank.relevance.complete")
    def test_non_food_types_work(self, mock_complete):
        mock_complete.return_value = '["hotel", "resort_hotel", "bed_and_breakfast"]'
        result = get_relevant_types("best hotel")
        assert result is not None
        assert "hotel" in result


class TestFilterByRelevance:
    def test_keeps_matching_places(self):
        places = [
            _make_place("1", ["pizza_restaurant", "restaurant"]),
            _make_place("2", ["grocery_store"]),
            _make_place("3", ["italian_restaurant"]),
        ]
        result = filter_by_relevance(places, ["pizza_restaurant", "italian_restaurant", "restaurant"])
        assert len(result) == 2
        assert result[0].place_id == "1"
        assert result[1].place_id == "3"

    def test_keeps_places_with_no_types(self):
        places = [_make_place("1", ["pizza_restaurant"]), _make_place("2", [])]
        result = filter_by_relevance(places, ["pizza_restaurant"])
        assert len(result) == 2

    def test_returns_all_when_no_relevant_types(self):
        places = [_make_place("1", ["grocery_store"]), _make_place("2", ["gas_station"])]
        assert len(filter_by_relevance(places, [])) == 2

    def test_partial_type_overlap_keeps_place(self):
        places = [_make_place("1", ["restaurant", "gas_station"])]
        assert len(filter_by_relevance(places, ["restaurant"])) == 1


class TestAllTypes:
    def test_taxonomy_is_not_empty(self):
        assert len(ALL_TYPES) > 100

    def test_common_types_present(self):
        for t in ["restaurant", "hotel", "park", "museum", "hospital", "school", "bar"]:
            assert t in ALL_TYPES, f"{t} missing from taxonomy"


# ═══ Grid Search ═══


class TestGenerateSubcircles:
    def test_small_radius_returns_single_circle(self):
        circles = _generate_subcircles(37.54, -77.43, 3000)
        assert len(circles) == 1
        assert circles[0] == (37.54, -77.43, 3000)

    def test_medium_radius_uses_hex_packing(self):
        circles = _generate_subcircles(37.54, -77.43, 8000)
        # Center (1) + at least one ring of 6
        assert len(circles) >= 7

    def test_large_radius_generates_many_subcircles(self):
        circles = _generate_subcircles(37.54, -77.43, 25000)
        # Should have multiple rings
        assert len(circles) >= 19

    def test_subcircles_have_small_radius(self):
        circles = _generate_subcircles(37.54, -77.43, 15000)
        for _, _, r in circles:
            assert r <= 2500  # MAX_SUB_RADIUS_M

    def test_subcircles_near_center(self):
        circles = _generate_subcircles(37.54, -77.43, 10000)
        for lat, lng, _ in circles:
            assert abs(lat - 37.54) < 0.2
            assert abs(lng - (-77.43)) < 0.2


class TestGridSearch:
    @patch("credrank.grid_search.search_places")
    def test_deduplicates_by_place_id(self, mock_search):
        a = _make_place("A", ["restaurant"])
        b = _make_place("B", ["restaurant"], lat=37.541, lng=-77.431)
        mock_search.side_effect = [[a], [a, b]] + [[] for _ in range(20)]
        results = grid_search("pizza", 37.54, -77.43, 8000)
        ids = [p.place_id for p in results]
        assert ids.count("A") == 1
        assert "B" in ids

    @patch("credrank.grid_search.search_places")
    def test_filters_to_original_radius(self, mock_search):
        inside = _make_place("IN", ["restaurant"], lat=37.54, lng=-77.43)
        outside = _make_place("OUT", ["restaurant"], lat=38.0, lng=-77.0)
        mock_search.return_value = [inside, outside]
        results = grid_search("pizza", 37.54, -77.43, 5000)
        assert any(p.place_id == "IN" for p in results)
        assert not any(p.place_id == "OUT" for p in results)

    @patch("credrank.grid_search.search_places")
    def test_handles_subcircle_failure_gracefully(self, mock_search):
        place = _make_place("A", ["restaurant"])
        mock_search.side_effect = [Exception("API error"), [place]] + [[] for _ in range(20)]
        results = grid_search("pizza", 37.54, -77.43, 8000)
        assert len(results) >= 1
        assert results[0].place_id == "A"

    @patch("credrank.grid_search.search_places")
    def test_respects_max_results(self, mock_search):
        places = [_make_place(str(i), ["restaurant"]) for i in range(20)]
        mock_search.return_value = places
        results = grid_search("pizza", 37.54, -77.43, 8000, max_results=5)
        assert len(results) <= 5
