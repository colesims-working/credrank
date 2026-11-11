"""Tests for intent extraction, intent-based filtering, and grid search."""

from unittest.mock import patch

from credrank.grid_search import (
    MIN_CELL_RADIUS_M,
    generate_subcircles,
    grid_search,
)
from credrank.models import (
    PlaceResult,
    PlacesSearchResult,
    SearchIntent,
)
from credrank.relevance import (
    ALL_TYPES,
    TABLE_A_FILTERABLE,
    build_intent,
    filter_by_intent,
)


def _make_place(
    place_id: str,
    types: list[str],
    lat: float = 37.54,
    lng: float = -77.43,
    name: str | None = None,
    primary_type: str | None = None,
) -> PlaceResult:
    return PlaceResult(
        place_id=place_id,
        name=name or f"Place {place_id}",
        types=types,
        primary_type=primary_type,
        rating=4.0,
        user_rating_count=100,
        lat=lat,
        lng=lng,
    )


# ═══ build_intent: LLM path ═══


class TestBuildIntentLLM:
    @patch("credrank.relevance.complete")
    def test_parses_valid_intent_json(self, mock_complete):
        mock_complete.return_value = (
            '{"query_kind": "specific_food", '
            '"included_types": ["pizza_restaurant"], '
            '"excluded_types": [], '
            '"required_terms": ["pizza"], '
            '"allow_generic_restaurant": true, '
            '"strictness": "broad"}'
        )
        intent = build_intent("pizza near me")
        assert intent.query_kind == "specific_food"
        assert "pizza_restaurant" in intent.included_types
        assert intent.required_terms == ["pizza"]
        assert intent.allow_generic_restaurant is True
        assert intent.strictness == "broad"

    @patch("credrank.relevance.complete")
    def test_strips_markdown_fences(self, mock_complete):
        mock_complete.return_value = (
            '```json\n'
            '{"query_kind":"specific_food","included_types":["cafe","coffee_shop"],'
            '"required_terms":["coffee"],"allow_generic_restaurant":false,'
            '"strictness":"strict"}\n'
            '```'
        )
        intent = build_intent("coffee")
        assert "cafe" in intent.included_types
        assert "coffee_shop" in intent.included_types
        assert intent.required_terms == ["coffee"]

    @patch("credrank.relevance.complete")
    def test_falls_back_on_malformed_json(self, mock_complete):
        mock_complete.return_value = "not json, just prose"
        intent = build_intent("pizza")
        # Deterministic pizza rule kicks in.
        assert "pizza_restaurant" in intent.included_types
        assert intent.query_kind == "specific_food"

    @patch("credrank.relevance.complete")
    def test_falls_back_on_exception(self, mock_complete):
        mock_complete.side_effect = RuntimeError("Gemini down")
        intent = build_intent("coffee")
        # Deterministic coffee rule kicks in.
        assert "coffee_shop" in intent.included_types
        assert intent.query_kind == "specific_food"

    @patch("credrank.relevance.complete")
    def test_falls_back_on_empty_response(self, mock_complete):
        mock_complete.return_value = ""
        intent = build_intent("sushi")
        assert "sushi_restaurant" in intent.included_types

    @patch("credrank.relevance.complete")
    def test_falls_back_when_all_types_invalid(self, mock_complete):
        mock_complete.return_value = (
            '{"query_kind": "specific_food", '
            '"included_types": ["fake_type_xyz", "another_fake"], '
            '"excluded_types": [], '
            '"required_terms": [], '
            '"allow_generic_restaurant": true, '
            '"strictness": "broad"}'
        )
        intent = build_intent("pizza")
        # Fallback fires because nothing useful survived cleanup.
        assert "pizza_restaurant" in intent.included_types

    @patch("credrank.relevance.complete")
    def test_normalizes_partial_valid_types(self, mock_complete):
        mock_complete.return_value = (
            '{"query_kind": "specific_food", '
            '"included_types": ["pizza_restaurant", "fake_type_xyz"], '
            '"required_terms": ["pizza"], '
            '"allow_generic_restaurant": true, '
            '"strictness": "broad"}'
        )
        intent = build_intent("pizza")
        assert "pizza_restaurant" in intent.included_types
        assert "fake_type_xyz" not in intent.included_types

    @patch("credrank.relevance.complete")
    def test_non_food_service_intent(self, mock_complete):
        mock_complete.return_value = (
            '{"query_kind":"service","included_types":["hotel","resort_hotel"],'
            '"required_terms":[],"allow_generic_restaurant":false,'
            '"strictness":"broad"}'
        )
        intent = build_intent("best hotel")
        assert "hotel" in intent.included_types
        assert intent.allow_generic_restaurant is False


# ═══ Deterministic fallback rules ═══


class TestDeterministicIntent:
    @patch("credrank.relevance.complete", side_effect=RuntimeError("llm down"))
    def test_pizza(self, _mock):
        intent = build_intent("best pizza in town")
        assert "pizza_restaurant" in intent.included_types
        assert "pizza" in intent.required_terms
        assert intent.allow_generic_restaurant is True

    @patch("credrank.relevance.complete", side_effect=RuntimeError("llm down"))
    def test_coffee(self, _mock):
        intent = build_intent("coffee near me")
        assert "coffee_shop" in intent.included_types
        assert "cafe" in intent.included_types
        assert intent.strictness == "strict"

    @patch("credrank.relevance.complete", side_effect=RuntimeError("llm down"))
    def test_breakfast(self, _mock):
        intent = build_intent("best breakfast near me")
        assert "breakfast_restaurant" in intent.included_types
        assert "brunch_restaurant" in intent.included_types
        assert "cafe" in intent.included_types
        # No required terms — filter auto-expands to accept `restaurant`.
        assert intent.required_terms == []
        assert intent.allow_generic_restaurant is True

    @patch("credrank.relevance.complete", side_effect=RuntimeError("llm down"))
    def test_brunch(self, _mock):
        intent = build_intent("brunch spots")
        assert "brunch_restaurant" in intent.included_types

    @patch("credrank.relevance.complete", side_effect=RuntimeError("llm down"))
    def test_dinner_is_general_passthrough(self, _mock):
        intent = build_intent("dinner")
        # Dinner is so broad we trust Google's relevance — no included
        # types means filter_by_intent short-circuits on the general
        # kind check and passes everything through.
        assert intent.query_kind == "general"
        assert intent.included_types == []

    @patch("credrank.relevance.complete", side_effect=RuntimeError("llm down"))
    def test_sushi(self, _mock):
        intent = build_intent("sushi")
        assert "sushi_restaurant" in intent.included_types

    @patch("credrank.relevance.complete", side_effect=RuntimeError("llm down"))
    def test_burger(self, _mock):
        intent = build_intent("cheeseburger")
        assert "hamburger_restaurant" in intent.included_types

    @patch("credrank.relevance.complete", side_effect=RuntimeError("llm down"))
    def test_hotel(self, _mock):
        intent = build_intent("hotel")
        assert "hotel" in intent.included_types
        assert intent.query_kind == "service"
        assert intent.allow_generic_restaurant is False

    @patch("credrank.relevance.complete", side_effect=RuntimeError("llm down"))
    def test_gym(self, _mock):
        intent = build_intent("gym nearby")
        assert "gym" in intent.included_types
        assert intent.strictness == "strict"

    @patch("credrank.relevance.complete", side_effect=RuntimeError("llm down"))
    def test_unrecognized_query_falls_through_to_general(self, _mock):
        intent = build_intent("xyzzy asdfghjkl")
        assert intent.query_kind == "general"
        assert intent.included_types == []


# ═══ filter_by_intent: pizza acceptance ═══


class TestFilterByIntentPizza:
    PIZZA_INTENT = SearchIntent(
        query_kind="specific_food",
        included_types=["pizza_restaurant", "pizza_delivery"],
        required_terms=["pizza", "pizzeria"],
        allow_generic_restaurant=True,
        strictness="broad",
    )

    def test_keeps_pizza_restaurant_by_primary_type(self):
        place = _make_place(
            "1",
            types=["pizza_restaurant", "restaurant"],
            primary_type="pizza_restaurant",
            name="Domino's",
        )
        kept = filter_by_intent([place], self.PIZZA_INTENT)
        assert len(kept) == 1

    def test_keeps_pizza_delivery_as_secondary_type(self):
        place = _make_place(
            "2",
            types=["pizza_restaurant", "pizza_delivery", "meal_delivery"],
            primary_type="pizza_restaurant",
            name="Pizza Hut",
        )
        kept = filter_by_intent([place], self.PIZZA_INTENT)
        assert len(kept) == 1

    def test_keeps_generic_restaurant_with_pizza_in_name(self):
        place = _make_place(
            "3",
            types=["restaurant"],
            name="Tony's Pizza Palace",
        )
        kept = filter_by_intent([place], self.PIZZA_INTENT)
        assert len(kept) == 1
        assert kept[0].place_id == "3"

    def test_drops_generic_restaurant_without_pizza_evidence(self):
        place = _make_place(
            "4",
            types=["restaurant"],
            name="Mel's Diner",
        )
        kept = filter_by_intent([place], self.PIZZA_INTENT)
        assert kept == []

    def test_drops_italian_restaurant_without_pizza_evidence(self):
        place = _make_place(
            "5",
            types=["italian_restaurant", "restaurant"],
            primary_type="italian_restaurant",
            name="Guido's Trattoria",
        )
        kept = filter_by_intent([place], self.PIZZA_INTENT)
        assert kept == []

    def test_keeps_italian_restaurant_with_pizza_in_name(self):
        place = _make_place(
            "6",
            types=["italian_restaurant", "restaurant"],
            primary_type="italian_restaurant",
            name="Tony's Pizza & Italian",
        )
        kept = filter_by_intent([place], self.PIZZA_INTENT)
        assert len(kept) == 1

    def test_drops_gas_station(self):
        place = _make_place(
            "7",
            types=["gas_station"],
            primary_type="gas_station",
            name="Shell",
        )
        kept = filter_by_intent([place], self.PIZZA_INTENT)
        assert kept == []

    def test_drops_grocery_store(self):
        place = _make_place(
            "8",
            types=["grocery_store", "supermarket"],
            primary_type="grocery_store",
            name="Kroger",
        )
        kept = filter_by_intent([place], self.PIZZA_INTENT)
        assert kept == []

    def test_pizza_end_to_end_acceptance(self):
        """Acceptance: pizza keeps pizza_restaurant, pizza_delivery, and
        generic restaurants with pizza evidence; drops unrelated generics,
        gas stations, and grocery stores."""
        places = [
            _make_place(
                "A",
                ["pizza_restaurant"],
                primary_type="pizza_restaurant",
                name="Domino's",
            ),
            _make_place(
                "B",
                ["pizza_restaurant", "pizza_delivery"],
                primary_type="pizza_restaurant",
                name="Pizza Hut",
            ),
            _make_place("C", ["restaurant"], name="Slice House Pizzeria"),
            _make_place("D", ["restaurant"], name="Mel's Diner"),
            _make_place("E", ["gas_station"], primary_type="gas_station", name="Shell"),
            _make_place(
                "F",
                ["grocery_store"],
                primary_type="grocery_store",
                name="Kroger",
            ),
            _make_place(
                "G",
                ["italian_restaurant"],
                primary_type="italian_restaurant",
                name="Guido's",
            ),
        ]
        kept_ids = {p.place_id for p in filter_by_intent(places, self.PIZZA_INTENT)}
        assert kept_ids == {"A", "B", "C"}


# ═══ filter_by_intent: coffee acceptance ═══


class TestFilterByIntentBroadFoodQueries:
    """Regression for the 'Mel's Aruba' bug.

    A broad food query ('breakfast', 'brunch') without required_terms
    must keep places Google labelled with just `primary_type="restaurant"`
    — these are often the real winners (diners, local breakfast spots)
    that Google didn't bother tagging with a specific food subtype. The
    filter auto-expands `included_types` with `restaurant` in that
    scenario. Food queries WITH required_terms (e.g. pizza) stay strict.
    """

    # Exactly the intent shape we observed in the production Aruba log:
    # LLM returned kind=general with a narrow food type list, no terms,
    # and allow_generic_restaurant unset (False).
    BREAKFAST_INTENT = SearchIntent(
        query_kind="general",
        included_types=["breakfast_restaurant", "brunch_restaurant", "cafe"],
        required_terms=[],
        allow_generic_restaurant=False,
        strictness="broad",
    )

    def test_keeps_generic_restaurant_with_no_terms(self):
        """The Mel's Aruba case: primary_type='restaurant', no cuisine
        subtype, and the breakfast intent has no required_terms to
        catch it via name evidence. Auto-expand must keep it."""
        place = _make_place(
            "mels",
            types=["restaurant", "food", "establishment"],
            primary_type="restaurant",
            name="Mel's Aruba",
        )
        kept = filter_by_intent([place], self.BREAKFAST_INTENT)
        assert len(kept) == 1
        assert kept[0].place_id == "mels"

    def test_keeps_specific_breakfast_restaurant(self):
        place = _make_place(
            "waffle",
            types=["breakfast_restaurant", "restaurant"],
            primary_type="breakfast_restaurant",
            name="Waffle House",
        )
        kept = filter_by_intent([place], self.BREAKFAST_INTENT)
        assert len(kept) == 1

    def test_keeps_cafe(self):
        place = _make_place(
            "joe",
            types=["cafe"],
            primary_type="cafe",
            name="Joe's Cafe",
        )
        kept = filter_by_intent([place], self.BREAKFAST_INTENT)
        assert len(kept) == 1

    def test_drops_gas_station(self):
        place = _make_place(
            "shell",
            types=["gas_station"],
            primary_type="gas_station",
            name="Shell",
        )
        kept = filter_by_intent([place], self.BREAKFAST_INTENT)
        assert kept == []

    def test_drops_grocery_store(self):
        place = _make_place(
            "kroger",
            types=["grocery_store", "supermarket"],
            primary_type="grocery_store",
            name="Kroger",
        )
        kept = filter_by_intent([place], self.BREAKFAST_INTENT)
        assert kept == []

    def test_food_query_with_terms_still_strict(self):
        """Regression guard: auto-expand must NOT fire when the intent
        has required_terms. Pizza queries need the narrow type check
        plus the term evidence to drop unrelated diners."""
        pizza_intent = SearchIntent(
            query_kind="specific_food",
            included_types=["pizza_restaurant", "pizza_delivery"],
            required_terms=["pizza", "pizzeria"],
            allow_generic_restaurant=True,
            strictness="broad",
        )
        diner = _make_place(
            "mels_diner",
            types=["restaurant"],
            primary_type="restaurant",
            name="Mel's Diner",
        )
        kept = filter_by_intent([diner], pizza_intent)
        # Generic diner with no pizza evidence must still be dropped.
        assert kept == []


class TestFilterByIntentCoffee:
    COFFEE_INTENT = SearchIntent(
        query_kind="specific_food",
        included_types=["coffee_shop", "cafe"],
        required_terms=["coffee", "espresso"],
        allow_generic_restaurant=False,
        strictness="strict",
    )

    def test_keeps_coffee_shop(self):
        place = _make_place(
            "1",
            types=["coffee_shop", "cafe"],
            primary_type="coffee_shop",
            name="Starbucks",
        )
        kept = filter_by_intent([place], self.COFFEE_INTENT)
        assert len(kept) == 1

    def test_keeps_cafe(self):
        place = _make_place(
            "2",
            types=["cafe"],
            primary_type="cafe",
            name="Joe's Cafe",
        )
        kept = filter_by_intent([place], self.COFFEE_INTENT)
        assert len(kept) == 1

    def test_drops_convenience_store(self):
        place = _make_place(
            "3",
            types=["convenience_store"],
            primary_type="convenience_store",
            name="7-Eleven",
        )
        kept = filter_by_intent([place], self.COFFEE_INTENT)
        assert kept == []

    def test_drops_gas_station(self):
        place = _make_place(
            "4",
            types=["gas_station"],
            primary_type="gas_station",
            name="Shell",
        )
        kept = filter_by_intent([place], self.COFFEE_INTENT)
        assert kept == []

    def test_allow_generic_false_drops_generic_restaurants(self):
        place = _make_place(
            "5",
            types=["restaurant"],
            name="The Diner",
        )
        kept = filter_by_intent([place], self.COFFEE_INTENT)
        assert kept == []

    def test_coffee_end_to_end_acceptance(self):
        """Acceptance: coffee keeps cafes/coffee shops, drops gas stations
        and convenience stores."""
        places = [
            _make_place(
                "A",
                ["coffee_shop"],
                primary_type="coffee_shop",
                name="Starbucks",
            ),
            _make_place("B", ["cafe"], primary_type="cafe", name="Joe's Cafe"),
            _make_place(
                "C",
                ["gas_station"],
                primary_type="gas_station",
                name="Shell",
            ),
            _make_place(
                "D",
                ["convenience_store"],
                primary_type="convenience_store",
                name="7-Eleven",
            ),
            _make_place("E", ["restaurant"], name="Bob's Diner"),
        ]
        kept_ids = {p.place_id for p in filter_by_intent(places, self.COFFEE_INTENT)}
        assert kept_ids == {"A", "B"}


# ═══ filter_by_intent: general/strict/broad edge cases ═══


class TestFilterByIntentEdgeCases:
    def test_general_intent_with_no_filters_is_noop(self):
        intent = SearchIntent(query_kind="general")
        places = [
            _make_place("1", ["restaurant"]),
            _make_place("2", ["gas_station"]),
            _make_place("3", []),
        ]
        kept = filter_by_intent(places, intent)
        assert len(kept) == 3

    def test_strict_mode_drops_no_type_place(self):
        intent = SearchIntent(
            query_kind="specific_food",
            included_types=["pizza_restaurant"],
            required_terms=["pizza"],
            strictness="strict",
            allow_generic_restaurant=False,
        )
        place = _make_place("1", types=[], primary_type=None, name="Unknown Place")
        kept = filter_by_intent([place], intent)
        assert kept == []

    def test_broad_mode_with_no_terms_keeps_no_type_place(self):
        intent = SearchIntent(
            query_kind="service",
            included_types=["hotel"],
            required_terms=[],
            strictness="broad",
            allow_generic_restaurant=False,
        )
        place = _make_place("1", types=[], primary_type=None, name="Unknown Lodging")
        kept = filter_by_intent([place], intent)
        assert len(kept) == 1

    def test_excluded_type_drops_place(self):
        intent = SearchIntent(
            query_kind="specific_food",
            included_types=["restaurant"],
            excluded_types=["gas_station"],
            required_terms=[],
            allow_generic_restaurant=False,
        )
        gas = _make_place(
            "1",
            types=["restaurant", "gas_station"],
            primary_type="gas_station",
            name="Gas + Grill",
        )
        kept = filter_by_intent([gas], intent)
        assert kept == []


# ═══ Graceful degradation ═══


class TestGracefulDegradation:
    @patch("credrank.relevance.complete", side_effect=RuntimeError("gemini down"))
    def test_search_continues_when_gemini_fails(self, _mock):
        """End-to-end: build_intent falls back to rules, then filter works."""
        intent = build_intent("pizza")
        assert intent is not None
        assert "pizza_restaurant" in intent.included_types

        places = [
            _make_place(
                "keep",
                ["pizza_restaurant"],
                primary_type="pizza_restaurant",
                name="Domino's",
            ),
            _make_place(
                "drop",
                ["gas_station"],
                primary_type="gas_station",
                name="Shell",
            ),
        ]
        kept = filter_by_intent(places, intent)
        assert len(kept) == 1
        assert kept[0].place_id == "keep"


# ═══ Taxonomy ═══


class TestTaxonomy:
    def test_all_types_is_not_empty(self):
        assert len(ALL_TYPES) > 100

    def test_common_types_present(self):
        for t in [
            "restaurant",
            "hotel",
            "park",
            "museum",
            "hospital",
            "school",
            "bar",
        ]:
            assert t in ALL_TYPES, f"{t} missing from taxonomy"

    def test_table_a_is_subset_of_all_types(self):
        assert TABLE_A_FILTERABLE.issubset(ALL_TYPES)

    def test_pizza_delivery_in_taxonomy_but_not_table_a(self):
        """pizza_delivery is valid response-only (Table B) — cannot be
        used as includedType server-side but can match post-filter."""
        assert "pizza_delivery" in ALL_TYPES
        assert "pizza_delivery" not in TABLE_A_FILTERABLE


# ═══ Grid Search ═══


class TestGenerateSubcircles:
    """Hex-packing math is tested with an explicit small `max_sub_radius`
    so results don't depend on the module's default (which is tuned for
    the adaptive grid and may change)."""

    def test_small_radius_returns_single_circle(self):
        circles = generate_subcircles(37.54, -77.43, 3000, max_sub_radius=2500)
        assert len(circles) == 1
        assert circles[0] == (37.54, -77.43, 3000)

    def test_medium_radius_uses_hex_packing(self):
        circles = generate_subcircles(37.54, -77.43, 8000, max_sub_radius=2500)
        # Center (1) + at least one ring of 6.
        assert len(circles) >= 7

    def test_large_radius_generates_many_subcircles(self):
        circles = generate_subcircles(37.54, -77.43, 25000, max_sub_radius=2500)
        assert len(circles) >= 19

    def test_subcircles_have_expected_cell_radius(self):
        circles = generate_subcircles(37.54, -77.43, 15000, max_sub_radius=2500)
        for _, _, r in circles:
            assert r <= 2500

    def test_subcircles_stay_near_center(self):
        circles = generate_subcircles(37.54, -77.43, 10000, max_sub_radius=2500)
        for lat, lng, _ in circles:
            assert abs(lat - 37.54) < 0.2
            assert abs(lng - (-77.43)) < 0.2

    def test_default_cell_radius_matches_initial_constant(self):
        """The module default for `max_sub_radius` is tuned for the
        adaptive grid. A 10-mile search should produce fewer than the
        old 91-cell fixed grid."""
        from credrank.grid_search import INITIAL_CELL_RADIUS_M

        circles = generate_subcircles(37.54, -77.43, 16000)
        # Cells all use the new default radius.
        for _, _, r in circles:
            assert r == INITIAL_CELL_RADIUS_M
        # Far fewer than the old 2500m-cell count of ~91.
        assert len(circles) < 50

    def test_num_rings_does_not_overprovision_small_radii(self):
        """Regression for the Aruba 7.5mi search bug.

        A 12131m radius (7.5 miles) with 6000m cells used to produce 19
        cells (3 rings) because the old formula `ceil(radius/step)`
        ignored that each cell's own `sub_radius` extends past its
        center. Ring 1 at 9000m already has outer edge at 15000m, which
        covers the whole 12131m disk — 7 cells suffice.
        """
        circles = generate_subcircles(37.54, -77.43, 12131, max_sub_radius=6000)
        # 1 center + 6 ring-1 = 7. Anything more means we spawned
        # ring-2 cells whose centers are well outside the user's disk.
        assert len(circles) == 7

    def test_num_rings_still_covers_large_radii(self):
        """The tighter num_rings formula must still add a ring when the
        user's radius genuinely reaches past ring 1's outer edge."""
        # Ring 1 outer edge = step + sub_radius = 9000 + 6000 = 15000m.
        # A 15001m radius needs ring 2.
        circles = generate_subcircles(37.54, -77.43, 15001, max_sub_radius=6000)
        assert len(circles) == 19  # 1 + 6 + 12

    def test_split_cell_covers_parent_at_worst_case_angle(self):
        """Regression for the SPLIT_RATIO=0.6 angular gap bug.

        After splitting, the worst uncovered point is on the parent's
        outer edge in the direction halfway between two adjacent ring
        children (30° from any ring child's angle). That point must be
        within some child cell's radius for the tile to be complete.
        """
        import math as _math

        from credrank.grid_search import (
            SPLIT_RATIO,
            SPLIT_RING_FACTOR,
            _split_cell,
        )

        parent_lat, parent_lng = 0.0, 0.0
        parent_radius = 6000
        children = _split_cell(parent_lat, parent_lng, parent_radius)

        # Worst point: on parent's outer edge, at 30° between two ring children.
        worst_angle = _math.radians(30)
        worst_x = parent_radius * _math.cos(worst_angle)
        worst_y = parent_radius * _math.sin(worst_angle)
        # Convert back into lat/lng offsets the way _split_cell does.
        worst_lat_off = worst_y / 111320
        worst_lng_off = worst_x / (111320 * _math.cos(_math.radians(parent_lat)))
        worst_lat = parent_lat + worst_lat_off
        worst_lng = parent_lng + worst_lng_off

        child_radius_m = int(parent_radius * SPLIT_RATIO)
        covered = False
        for child_lat, child_lng, _r in children:
            # Approximate meter distance from worst point to child center.
            dlat_m = (child_lat - worst_lat) * 111320
            dlng_m = (
                (child_lng - worst_lng)
                * 111320
                * _math.cos(_math.radians(parent_lat))
            )
            dist = _math.sqrt(dlat_m * dlat_m + dlng_m * dlng_m)
            if dist <= child_radius_m:
                covered = True
                break

        assert covered, (
            f"Split gap at 30° angle: SPLIT_RATIO={SPLIT_RATIO} "
            f"SPLIT_RING_FACTOR={SPLIT_RING_FACTOR} does not tile parent."
        )


class TestGridSearch:
    """Tests for adaptive grid search.

    These tests patch `credrank.grid_search.generate_subcircles` to
    control the initial cell set precisely (production behavior depends
    on hex-packing math that varies with input radius). Saturation
    behavior is tested separately by returning a `PlacesSearchResult`
    with `saturated=True`.
    """

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[(37.54, -77.43, 3000), (37.541, -77.431, 3000)],
    )
    @patch("credrank.grid_search.search_places")
    def test_deduplicates_by_place_id(self, mock_search, _mock_gen):
        a = _make_place("A", ["restaurant"])
        b = _make_place("B", ["restaurant"], lat=37.541, lng=-77.431)
        mock_search.side_effect = [[a], [a, b]]
        result = grid_search("pizza", 37.54, -77.43, 8000)
        ids = [p.place_id for p in result]
        assert ids.count("A") == 1
        assert "B" in ids

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[(37.54, -77.43, 3000)],
    )
    @patch("credrank.grid_search.search_places")
    def test_filters_to_original_radius(self, mock_search, _mock_gen):
        inside = _make_place("IN", ["restaurant"], lat=37.54, lng=-77.43)
        outside = _make_place("OUT", ["restaurant"], lat=38.0, lng=-77.0)
        mock_search.return_value = [inside, outside]
        result = grid_search("pizza", 37.54, -77.43, 5000)
        assert any(p.place_id == "IN" for p in result)
        assert not any(p.place_id == "OUT" for p in result)

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[(37.54, -77.43, 3000), (37.541, -77.431, 3000)],
    )
    @patch("credrank.grid_search.search_places")
    def test_handles_cell_failure_gracefully(self, mock_search, _mock_gen):
        place = _make_place("A", ["restaurant"])
        mock_search.side_effect = [Exception("API error"), [place]]
        result = grid_search("pizza", 37.54, -77.43, 8000)
        assert len(result) == 1
        assert result[0].place_id == "A"
        # A cell failure flips coverage to incomplete with a reason.
        assert result.coverage.complete is False
        assert result.coverage.cells_failed == 1
        assert "failed" in (result.coverage.reason or "")

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[(37.54, -77.43, 3000)],
    )
    @patch("credrank.grid_search.search_places")
    def test_passes_included_type_when_provided(self, mock_search, _mock_gen):
        mock_search.return_value = []
        grid_search("pizza", 37.54, -77.43, 3000, included_type="pizza_restaurant")
        assert mock_search.called
        params = mock_search.call_args[0][0]
        assert params.included_type == "pizza_restaurant"
        assert params.strict_type_filtering is True

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[(37.54, -77.43, 3000)],
    )
    @patch("credrank.grid_search.search_places")
    def test_omits_included_type_when_none(self, mock_search, _mock_gen):
        mock_search.return_value = []
        grid_search("best food", 37.54, -77.43, 3000)
        params = mock_search.call_args[0][0]
        assert params.included_type is None
        assert params.strict_type_filtering is False

    # ── No early stopping (regression-proof) ──

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[
            (37.54, -77.43, 3000),
            (37.55, -77.44, 3000),
        ],
    )
    @patch("credrank.grid_search.search_places")
    def test_no_early_stopping_later_cell_contributes_top_result(
        self, mock_search, _mock_gen
    ):
        """The first cell returning many mediocre places must not prevent
        a later cell's top place from appearing in the result set — i.e.
        grid search must NOT stop early on result count."""
        mediocre = [
            _make_place(
                f"med_{i}",
                ["restaurant"],
                lat=37.54,
                lng=-77.43,
            )
            for i in range(60)
        ]
        top = _make_place("TOP", ["restaurant"], lat=37.55, lng=-77.44)
        mock_search.side_effect = [mediocre, [top]]

        result = grid_search("pizza", 37.54, -77.43, 10000)

        assert result.coverage.cells_searched == 2
        assert any(p.place_id == "TOP" for p in result)
        assert len(result) == 61

    # ── Coverage metadata ──

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[(37.54, -77.43, 3000), (37.55, -77.44, 3000)],
    )
    @patch("credrank.grid_search.search_places")
    def test_coverage_complete_when_nothing_saturates(
        self, mock_search, _mock_gen
    ):
        mock_search.return_value = []
        result = grid_search("pizza", 37.54, -77.43, 6000)
        assert result.coverage.complete is True
        assert result.coverage.reason is None
        assert result.coverage.cells_searched == 2
        assert result.coverage.cells_saturated == 0
        assert result.coverage.cells_failed == 0

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[(37.54, -77.43, 3000)],
    )
    @patch("credrank.grid_search.search_places")
    def test_coverage_counts_raw_deduped_in_radius(
        self, mock_search, _mock_gen
    ):
        inside_a = _make_place("A", ["restaurant"], lat=37.54, lng=-77.43)
        inside_b = _make_place("B", ["restaurant"], lat=37.541, lng=-77.431)
        dup_a = _make_place("A", ["restaurant"], lat=37.54, lng=-77.43)
        out_of_radius = _make_place("OUT", ["restaurant"], lat=38.0, lng=-77.0)
        mock_search.return_value = [inside_a, inside_b, dup_a, out_of_radius]

        result = grid_search("pizza", 37.54, -77.43, 3000)

        # raw_count includes duplicates across pages
        assert result.coverage.raw_count == 4
        # deduped counts unique candidates before radius filter
        assert result.coverage.deduped_count == 3
        # in_radius drops places outside the user's requested circle
        assert result.coverage.in_radius_count == 2

    # ── Adaptive subdivision on saturation ──

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[(37.54, -77.43, 6000)],
    )
    @patch("credrank.grid_search.search_places")
    def test_adaptive_subdivision_splits_saturated_cells(
        self, mock_search, _mock_gen
    ):
        """A saturated parent cell above the min radius must be split
        into 6 hex-ring children (no concentric center child) and searched."""
        parent_result = PlacesSearchResult(
            [
                _make_place(f"P{i}", ["restaurant"])
                for i in range(60)
            ],
            saturated=True,
        )
        # Parent saturates; each of the 6 children returns empty.
        mock_search.side_effect = [parent_result] + [[] for _ in range(10)]

        result = grid_search("pizza", 37.54, -77.43, 6000)

        assert result.coverage.cells_searched == 1 + 6  # parent + 6 ring children
        assert result.coverage.cells_saturated == 1
        # The children were at >= min radius, so the run is still "complete".
        assert result.coverage.complete is True

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[(37.54, -77.43, 6000)],
    )
    @patch("credrank.grid_search.search_places")
    def test_split_has_no_concentric_center_child(
        self, mock_search, _mock_gen
    ):
        """Regression for the nested-saturation cascade: none of the 6
        children from a split should be at the exact parent center, so a
        dense parent can't trigger an infinite cascade of smaller and
        smaller circles at the same user point."""
        parent_lat, parent_lng = 37.54, -77.43
        parent_result = PlacesSearchResult(
            [_make_place(f"P{i}", ["restaurant"]) for i in range(60)],
            saturated=True,
        )
        mock_search.side_effect = [parent_result] + [[] for _ in range(10)]

        grid_search("pizza", parent_lat, parent_lng, 6000)

        # Collect the `lat, lng` args passed to search_places for every
        # call after the parent (i.e., the children).
        child_call_args = [
            call.args[0] for call in mock_search.call_args_list[1:]
        ]
        for params in child_call_args:
            lat, lng = params.location
            assert (lat, lng) != (parent_lat, parent_lng), (
                f"child at exact parent center ({parent_lat}, {parent_lng}) "
                "would reopen the nested-concentric cascade"
            )

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[(37.54, -77.43, MIN_CELL_RADIUS_M)],
    )
    @patch("credrank.grid_search.search_places")
    def test_saturation_at_minimum_radius_marks_incomplete(
        self, mock_search, _mock_gen
    ):
        """A cell already at MIN_CELL_RADIUS_M that saturates can't be
        subdivided — coverage must go to incomplete with a clear reason."""
        sat = PlacesSearchResult(
            [_make_place(f"P{i}", ["restaurant"]) for i in range(60)],
            saturated=True,
        )
        mock_search.return_value = sat

        result = grid_search("pizza", 37.54, -77.43, MIN_CELL_RADIUS_M)

        assert result.coverage.complete is False
        assert result.coverage.cells_saturated == 1
        assert "minimum radius" in (result.coverage.reason or "")

    @patch(
        "credrank.grid_search.generate_subcircles",
        return_value=[(37.54, -77.43, 1609)],  # 1 mile
    )
    @patch("credrank.grid_search.search_places")
    def test_one_mile_saturated_cell_subdivides(self, mock_search, _mock_gen):
        """Regression for the 'casual dinner' Aruba bug.

        A 1-mile (1609m) search that hits Google's 60-cap must subdivide
        and search children — previously MIN_CELL_RADIUS_M was 1500, so
        a 1609m parent's ~1045m children fell below the floor and the
        cell couldn't split at all. Dense 1-mile searches would return
        ~38 results and flag coverage incomplete without ever drilling in.
        """
        sat_parent = PlacesSearchResult(
            [_make_place(f"P{i}", ["restaurant"]) for i in range(60)],
            saturated=True,
        )
        # Parent saturates; children all return empty (simplest setup).
        mock_search.side_effect = [sat_parent] + [[] for _ in range(20)]

        result = grid_search("casual dinner", 37.54, -77.43, 1609)

        # Must subdivide into 6 ring children, so at least 7 cells total.
        assert result.coverage.cells_searched >= 7, (
            f"a 1-mile saturated cell must subdivide; got only "
            f"{result.coverage.cells_searched} cells (floor too high?)"
        )
        assert result.coverage.cells_saturated == 1

    def test_max_cell_cap_marks_incomplete(self, monkeypatch):
        """Hitting the global safety cap should short-circuit the run
        and mark coverage incomplete with a cap-specific reason."""
        monkeypatch.setattr("credrank.grid_search.MAX_CELL_COUNT", 2)
        monkeypatch.setattr(
            "credrank.grid_search.generate_subcircles",
            lambda *a, **k: [
                (37.54, -77.43, 3000),
                (37.55, -77.44, 3000),
                (37.56, -77.45, 3000),
            ],
        )
        monkeypatch.setattr(
            "credrank.grid_search.search_places",
            lambda _params: [],
        )

        result = grid_search("pizza", 37.54, -77.43, 8000)

        assert result.coverage.complete is False
        assert result.coverage.cells_searched == 2
        assert "safety cap" in (result.coverage.reason or "")
