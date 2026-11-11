"""Query-intent extraction and result filtering for CredRank.

Two responsibilities:

1. `build_intent(query)` — produces a structured `SearchIntent` from the
   user's free-text query. Primary path is a deterministic keyword-rule
   table that covers the overwhelmingly common query shapes for free and
   with zero latency. Queries no rule matches go to Gemini Flash (result
   cached per normalized query); on any LLM failure (network, timeout,
   bad JSON, unknown types, empty result) a permissive generic intent
   keeps the search working. `build_intent` never raises.

2. `filter_by_intent(places, intent)` — client-side post-filter that
   runs after Google has already narrowed results server-side via
   `includedType`. Catches false positives Google let through (e.g. a
   gas station under a "coffee" query) and keeps ambiguous matches
   Google labelled generic `restaurant` when the place name or a
   fetched summary carries the required term.

The taxonomy (`GOOGLE_PLACE_TYPES`, `ALL_TYPES`) and the filterable
subset (`TABLE_A_FILTERABLE`) live here so they're colocated with the
code that depends on them.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

from credrank.cache import MISS, intent_cache
from credrank.llm import complete
from credrank.models import PlaceResult, SearchIntent

log = logging.getLogger(__name__)


# ═══ Taxonomy ═══
#
# Complete set of Google Place types (Places API New, 2025), grouped by
# category so the LLM has semantic context in its prompt. The flat
# `ALL_TYPES` set validates LLM output; it's a superset of the types
# we can actually use as `includedType` in Text Search requests — that
# stricter subset is `TABLE_A_FILTERABLE` below.

GOOGLE_PLACE_TYPES: dict[str, list[str]] = {
    "food_and_drink": [
        "american_restaurant", "bakery", "bar", "barbecue_restaurant",
        "brazilian_restaurant", "breakfast_restaurant", "brunch_restaurant",
        "cafe", "chinese_restaurant", "coffee_shop", "deli",
        "fast_food_restaurant", "french_restaurant", "greek_restaurant",
        "hamburger_restaurant", "ice_cream_shop", "indian_restaurant",
        "indonesian_restaurant", "italian_restaurant", "japanese_restaurant",
        "korean_restaurant", "lebanese_restaurant", "meal_delivery",
        "meal_takeaway", "mediterranean_restaurant", "mexican_restaurant",
        "middle_eastern_restaurant", "pizza_delivery", "pizza_restaurant",
        "ramen_restaurant", "restaurant", "sandwich_shop", "seafood_restaurant",
        "spanish_restaurant", "steak_house", "sushi_restaurant",
        "thai_restaurant", "turkish_restaurant", "vegan_restaurant",
        "vegetarian_restaurant", "vietnamese_restaurant",
    ],
    "nightlife": [
        "bar", "night_club", "lounge", "pub", "wine_bar",
        "brewery", "distillery", "karaoke",
    ],
    "shopping": [
        "book_store", "cell_phone_store", "clothing_store",
        "convenience_store", "department_store", "discount_store",
        "electronics_store", "furniture_store", "gift_shop",
        "grocery_store", "hardware_store", "home_goods_store",
        "home_improvement_store", "jewelry_store", "liquor_store",
        "market", "pet_store", "shoe_store", "shopping_mall",
        "sporting_goods_store", "store", "supermarket",
        "wholesale_store",
    ],
    "health_and_wellness": [
        "dental_clinic", "dentist", "doctor", "drugstore",
        "hospital", "medical_lab", "pharmacy", "physiotherapist",
        "spa", "gym", "fitness_center", "yoga_studio",
    ],
    "services": [
        "accounting", "atm", "bank", "barber_shop", "beauty_salon",
        "car_dealer", "car_rental", "car_repair", "car_wash",
        "dry_cleaning", "electrician", "florist", "gas_station",
        "hair_care", "hair_salon", "insurance_agency", "laundry",
        "lawyer", "locksmith", "moving_company", "painter",
        "plumber", "real_estate_agency", "roofing_contractor",
        "storage", "tailor", "telecommunications_service_provider",
        "travel_agency", "veterinary_care",
    ],
    "entertainment_and_recreation": [
        "amusement_center", "amusement_park", "aquarium",
        "art_gallery", "bowling_alley", "casino", "community_center",
        "convention_center", "cultural_center", "dog_park",
        "event_venue", "hiking_area", "historical_landmark",
        "marina", "movie_rental", "movie_theater", "museum",
        "national_park", "park", "performing_arts_theater",
        "playground", "ski_resort", "stadium", "swimming_pool",
        "tourist_attraction", "visitor_center", "zoo",
    ],
    "lodging": [
        "bed_and_breakfast", "campground", "camping_cabin",
        "cottage", "extended_stay_hotel", "farmstay",
        "guest_house", "hostel", "hotel", "lodging",
        "motel", "private_guest_room", "resort_hotel",
        "rv_park",
    ],
    "education": [
        "library", "preschool", "primary_school", "school",
        "secondary_school", "university",
    ],
    "government_and_civic": [
        "city_hall", "courthouse", "embassy", "fire_station",
        "local_government_office", "police", "post_office",
    ],
    "transportation": [
        "airport", "bus_station", "bus_stop", "ferry_terminal",
        "heliport", "light_rail_station", "parking",
        "subway_station", "taxi_stand", "train_station",
        "transit_depot", "transit_station", "truck_stop",
    ],
    "religious": [
        "church", "hindu_temple", "mosque", "synagogue",
    ],
}

# Flat validation set — anything the LLM returns must be in here.
ALL_TYPES: set[str] = set()
for _types in GOOGLE_PLACE_TYPES.values():
    ALL_TYPES.update(_types)


# Specific food subtypes (everything in `food_and_drink` except the
# catch-all `restaurant`). When `filter_by_intent` sees any of these in
# the intent's `included_types` AND there are no `required_terms`, it
# auto-expands the accepted set to also include plain `restaurant` —
# Google often labels a diner/breakfast spot/brunch place with just
# `primary_type="restaurant"` even when it serves the specific food,
# and the narrow included_types list would otherwise drop real winners.
FOOD_SUBTYPES: frozenset[str] = frozenset({
    "american_restaurant", "bakery", "bar", "barbecue_restaurant",
    "brazilian_restaurant", "breakfast_restaurant", "brunch_restaurant",
    "cafe", "chinese_restaurant", "coffee_shop", "deli",
    "fast_food_restaurant", "french_restaurant", "greek_restaurant",
    "hamburger_restaurant", "ice_cream_shop", "indian_restaurant",
    "indonesian_restaurant", "italian_restaurant", "japanese_restaurant",
    "korean_restaurant", "lebanese_restaurant", "meal_delivery",
    "meal_takeaway", "mediterranean_restaurant", "mexican_restaurant",
    "middle_eastern_restaurant", "pizza_delivery", "pizza_restaurant",
    "ramen_restaurant", "sandwich_shop", "seafood_restaurant",
    "spanish_restaurant", "steak_house", "sushi_restaurant",
    "thai_restaurant", "turkish_restaurant", "vegan_restaurant",
    "vegetarian_restaurant", "vietnamese_restaurant",
})


# Subset of `ALL_TYPES` that is valid as `includedType` in a Text Search
# (New) request. Types in `ALL_TYPES` but NOT here (e.g. `pizza_delivery`)
# are Table B: they appear in responses and are useful for post-filter
# matching, but Google rejects them as request filters. `maps_client`
# imports this set to decide what to pass server-side.
TABLE_A_FILTERABLE: set[str] = {
    # Food & drink
    "american_restaurant", "bakery", "bar", "barbecue_restaurant",
    "brazilian_restaurant", "breakfast_restaurant", "brunch_restaurant",
    "cafe", "chinese_restaurant", "coffee_shop", "deli",
    "fast_food_restaurant", "french_restaurant", "greek_restaurant",
    "hamburger_restaurant", "ice_cream_shop", "indian_restaurant",
    "indonesian_restaurant", "italian_restaurant", "japanese_restaurant",
    "korean_restaurant", "lebanese_restaurant", "meal_delivery",
    "meal_takeaway", "mediterranean_restaurant", "mexican_restaurant",
    "middle_eastern_restaurant", "pizza_restaurant", "ramen_restaurant",
    "restaurant", "sandwich_shop", "seafood_restaurant",
    "spanish_restaurant", "steak_house", "sushi_restaurant",
    "thai_restaurant", "turkish_restaurant", "vegan_restaurant",
    "vegetarian_restaurant", "vietnamese_restaurant",
    # Lodging
    "bed_and_breakfast", "hostel", "hotel", "motel", "resort_hotel",
    # Everyday services
    "drugstore", "gas_station", "grocery_store", "gym", "hospital",
    "library", "museum", "park", "pharmacy", "supermarket",
    "convenience_store",
}


# ═══ Intent extraction: LLM primary path ═══

_VALID_QUERY_KINDS = {"specific_food", "cuisine", "service", "general"}
_VALID_STRICTNESS = {"strict", "broad"}

_INTENT_SYSTEM_PROMPT = """\
You extract a structured search intent from a free-text place search query.

Output EXACTLY one JSON object, nothing else. No prose, no markdown fences.
The keys are:

- "query_kind": one of "specific_food", "cuisine", "service", "general".
- "included_types": 1-3 Google Place types that best describe the query.
  Use the allowed list the user provides. Types outside the list are
  dropped on our side.
- "excluded_types": types to drop outright if they appear as primary_type.
  Usually empty.
- "required_terms": 0-4 lowercase substrings that an acceptable place's
  name or review summary would contain. For specific foods (pizza, ramen,
  sushi) include the food word. For general queries usually empty.
- "allow_generic_restaurant": true if a place Google labelled plainly
  "restaurant" (no cuisine subtype) could still satisfy the query —
  usually true for specific foods, false for non-food services.
- "strictness": "strict" or "broad". Use "strict" for non-food services
  (gyms, dentists, gas stations) and for queries where false positives
  are costly. Use "broad" for specific foods where Google may have
  labelled relevant places as generic.

Rules:
- Prefer 1 specific type over many broad ones. Use 2+ only when the
  query genuinely spans them (e.g. "coffee" -> coffee_shop + cafe).
- Do not include "restaurant" in included_types unless the query is
  genuinely generic ("dinner", "food near me"). For specific-food
  queries use allow_generic_restaurant=true instead.

Return the JSON object on a single line."""


def build_intent(query: str) -> SearchIntent:
    """Return a `SearchIntent` for the given free-text query.

    Resolution order:

    1. Deterministic keyword rules — free, instant, and hand-tuned for
       the query shapes people actually type ("pizza", "coffee",
       "sushi near me"). This used to be the fallback and the LLM the
       primary, which put a billed, blocking Gemini call with a large
       taxonomy prompt on *every* search for no accuracy gain on
       common queries.
    2. Gemini Flash for anything the rules don't recognize, cached per
       normalized query (intents are stable — "ethiopian food" means
       the same thing tomorrow).
    3. A permissive generic intent if the LLM fails.

    Never raises.
    """
    intent = _deterministic_intent(query)
    if intent is not None:
        log.info(
            "intent.rules.done query=%r kind=%s included=%s required=%s strict=%s",
            query,
            intent.query_kind,
            intent.included_types,
            intent.required_terms,
            intent.strictness,
        )
        return intent

    cache_key = query.strip().lower()
    cached = intent_cache.get(cache_key)
    if cached is not MISS:
        log.info("intent.cache_hit query=%r", query)
        return cached

    intent = _llm_intent(query)
    if intent is not None:
        intent_cache.set(cache_key, intent)
        return intent

    intent = _generic_intent()
    log.info("intent.fallback.generic query=%r", query)
    return intent


def _generic_intent() -> SearchIntent:
    """Permissive no-filter intent — lets Google's own relevance stand."""
    return SearchIntent(
        query_kind="general",
        included_types=[],
        required_terms=[],
        allow_generic_restaurant=True,
        strictness="broad",
    )


def _llm_intent(query: str) -> SearchIntent | None:
    """Call Gemini for a structured intent. Returns None on any failure."""
    type_display = "\n".join(
        f"  {cat}: {', '.join(sorted(types))}"
        for cat, types in GOOGLE_PLACE_TYPES.items()
    )
    allowed_display = ", ".join(sorted(ALL_TYPES))

    prompt = f"""Search query: "{query}"

Allowed Google Place types (use these only, anything else is dropped):
{allowed_display}

Full Google Place taxonomy grouped by category for semantic context:
{type_display}

Return the intent JSON object."""

    try:
        raw = complete(prompt, system=_INTENT_SYSTEM_PROMPT, max_tokens=400)
    except Exception as exc:
        log.warning("intent.llm.error query=%r error=%s", query, exc)
        return None

    if not raw:
        log.warning("intent.llm.empty query=%r", query)
        return None

    cleaned = _strip_fences(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning(
            "intent.llm.json_error query=%r error=%s raw=%r",
            query,
            exc,
            raw[:200],
        )
        return None

    intent = _parse_intent_payload(payload)
    if intent is None:
        return None

    log.info(
        "intent.llm.done query=%r kind=%s included=%s required=%s strict=%s",
        query,
        intent.query_kind,
        intent.included_types,
        intent.required_terms,
        intent.strictness,
    )
    return intent


def _strip_fences(raw: str) -> str:
    """Strip optional markdown fences from an LLM response."""
    return (
        raw.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )


def _parse_intent_payload(payload: Any) -> SearchIntent | None:
    """Validate and coerce a raw JSON payload into a `SearchIntent`.

    Returns None if the payload is structurally wrong or after cleanup
    has nothing useful to filter on. A partially-invalid payload (e.g.
    some types outside the taxonomy) is normalized, not rejected —
    unknown types are dropped but the rest stands.
    """
    if not isinstance(payload, dict):
        return None

    kind = payload.get("query_kind")
    if not isinstance(kind, str) or kind not in _VALID_QUERY_KINDS:
        kind = "general"

    included = [
        t for t in _string_list(payload.get("included_types"))
        if t in ALL_TYPES
    ]
    excluded = [
        t for t in _string_list(payload.get("excluded_types"))
        if t in ALL_TYPES
    ]

    required_terms = [
        t.strip().lower()
        for t in _string_list(payload.get("required_terms"))
        if t.strip()
    ]

    allow_generic = bool(payload.get("allow_generic_restaurant", False))

    strictness = payload.get("strictness")
    if strictness not in _VALID_STRICTNESS:
        strictness = "broad"

    # If nothing usable survived cleanup, signal failure so the caller
    # falls through to deterministic rules.
    if not included and not excluded and not required_terms:
        return None

    return SearchIntent(
        query_kind=kind,
        included_types=included,
        excluded_types=excluded,
        required_terms=required_terms,
        allow_generic_restaurant=allow_generic,
        strictness=strictness,
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str) and v]


# ═══ Intent extraction: deterministic rules (primary path) ═══


def _has_word(q: str, *words: str) -> bool:
    """True if any of `words` appears in `q` on word boundaries.

    Substring checks were fine when these rules were a rarely-hit
    fallback, but as the primary path they need word boundaries:
    "barber shop" must not match "bar", "phone store" must not match
    "pho".
    """
    return any(re.search(rf"\b{re.escape(w)}\b", q) for w in words)


def _deterministic_intent(query: str) -> SearchIntent | None:
    """Map common query shapes to known-good intents without the LLM.

    Order matters — first keyword match wins. Returns None when no rule
    matches, signalling the caller to consult the LLM.
    """
    q = query.lower()

    # ── Food ──
    if _has_word(q, "breakfast", "brunch"):
        # No required_terms — filter_by_intent's auto-expand adds
        # plain `restaurant` so generic-typed winners like a diner
        # still show up.
        return _food_intent(
            ["breakfast_restaurant", "brunch_restaurant", "cafe"],
            required=[],
            allow_generic=True,
        )
    if _has_word(q, "lunch", "dinner"):
        # Lunch/dinner covers essentially every restaurant — let
        # Google's relevance engine do the heavy lifting and don't
        # second-guess it client-side.
        return _generic_intent()
    if _has_word(q, "pizza", "pizzeria"):
        return _food_intent(
            ["pizza_restaurant", "pizza_delivery"],
            required=["pizza", "pizzeria"],
        )
    if _has_word(q, "coffee", "espresso", "latte"):
        return _food_intent(
            ["coffee_shop", "cafe"],
            required=["coffee", "espresso"],
            allow_generic=False,
            strictness="strict",
        )
    if _has_word(q, "sushi", "sashimi"):
        return _food_intent(
            ["sushi_restaurant", "japanese_restaurant"],
            required=["sushi"],
        )
    if _has_word(q, "ramen"):
        return _food_intent(
            ["ramen_restaurant", "japanese_restaurant"],
            required=["ramen"],
        )
    if _has_word(q, "burger", "burgers", "hamburger", "hamburgers", "cheeseburger", "cheeseburgers"):
        return _food_intent(
            ["hamburger_restaurant"],
            required=["burger"],
        )
    if _has_word(q, "taco", "tacos", "burrito", "burritos", "quesadilla", "quesadillas"):
        return _food_intent(
            ["mexican_restaurant"],
            required=["taco", "burrito"],
        )
    if _has_word(q, "bbq", "barbecue", "barbeque"):
        return _food_intent(
            ["barbecue_restaurant"],
            required=["bbq", "barbecue"],
        )
    if _has_word(q, "sandwich", "sandwiches", "hoagie", "hoagies", "deli"):
        return _food_intent(
            ["sandwich_shop", "deli"],
            required=["sandwich", "sub", "deli"],
        )
    if _has_word(q, "bakery", "pastry", "pastries", "croissant", "croissants"):
        return _food_intent(
            ["bakery"],
            required=["bakery", "pastry", "bread"],
        )
    if _has_word(q, "ice cream", "gelato"):
        return _food_intent(
            ["ice_cream_shop"],
            required=["ice cream", "gelato"],
            allow_generic=False,
            strictness="strict",
        )
    if _has_word(q, "thai"):
        return _food_intent(["thai_restaurant"])
    if _has_word(q, "mexican"):
        return _food_intent(["mexican_restaurant"])
    if _has_word(q, "chinese"):
        return _food_intent(["chinese_restaurant"])
    if _has_word(q, "indian"):
        return _food_intent(["indian_restaurant"])
    if _has_word(q, "italian"):
        return _food_intent(["italian_restaurant"])
    if _has_word(q, "japanese"):
        return _food_intent(["japanese_restaurant"])
    if _has_word(q, "korean"):
        return _food_intent(["korean_restaurant"])
    if _has_word(q, "vietnamese", "pho"):
        return _food_intent(["vietnamese_restaurant"])

    # ── Non-food services ──
    if _has_word(q, "bar", "bars", "pub", "pubs", "brewery", "breweries", "tavern", "taverns"):
        return SearchIntent(
            query_kind="service",
            included_types=["bar"],
            required_terms=["bar", "pub", "brewery"],
            allow_generic_restaurant=False,
            strictness="broad",
        )
    if _has_word(q, "hotel", "hotels", "motel", "motels", "inn", "resort", "resorts", "lodging"):
        return SearchIntent(
            query_kind="service",
            included_types=["hotel", "resort_hotel", "motel"],
            allow_generic_restaurant=False,
            strictness="broad",
        )
    if _has_word(q, "gas") and _has_word(q, "station", "stations"):
        return SearchIntent(
            query_kind="service",
            included_types=["gas_station"],
            allow_generic_restaurant=False,
            strictness="strict",
        )
    if _has_word(q, "grocery", "groceries", "supermarket", "supermarkets"):
        return SearchIntent(
            query_kind="service",
            included_types=["grocery_store", "supermarket"],
            allow_generic_restaurant=False,
            strictness="strict",
        )
    if _has_word(q, "pharmacy", "pharmacies", "drugstore", "drugstores"):
        return SearchIntent(
            query_kind="service",
            included_types=["pharmacy", "drugstore"],
            allow_generic_restaurant=False,
            strictness="strict",
        )
    if _has_word(q, "gym", "gyms", "fitness"):
        return SearchIntent(
            query_kind="service",
            included_types=["gym"],
            allow_generic_restaurant=False,
            strictness="strict",
        )
    if _has_word(q, "museum", "museums"):
        return SearchIntent(
            query_kind="service",
            included_types=["museum"],
            allow_generic_restaurant=False,
            strictness="broad",
        )
    if _has_word(q, "park", "parks"):
        return SearchIntent(
            query_kind="service",
            included_types=["park"],
            allow_generic_restaurant=False,
            strictness="broad",
        )

    # No rule matched — let the LLM interpret it.
    return None


def _food_intent(
    included: list[str],
    *,
    required: list[str] | None = None,
    allow_generic: bool = True,
    strictness: str = "broad",
) -> SearchIntent:
    return SearchIntent(
        query_kind="specific_food",
        included_types=list(included),
        required_terms=list(required or []),
        allow_generic_restaurant=allow_generic,
        strictness=strictness,
    )


# ═══ Post-filtering ═══


def filter_by_intent(
    places: Iterable[PlaceResult],
    intent: SearchIntent,
) -> list[PlaceResult]:
    """Filter a list of places against a `SearchIntent`.

    Decision order (first matching rule wins):

    1. `primary_type` or any type in `intent.excluded_types` → drop.
    2. `primary_type` or any type in `intent.included_types` → keep.
    3. Any `required_term` appears (case-insensitive) in the place name
       or any fetched summary (editorial, review, generative) → keep.
       This catches e.g. an italian_restaurant named "Tony's Pizza &
       Italian" under a pizza query.
    4. Place looks like a generic unspecified restaurant AND
       `allow_generic_restaurant` is true AND there are no required
       terms (otherwise rule 3 would have caught it) → keep.
    5. Place has no types at all → broad mode with no required terms
       keeps it; otherwise drop.
    6. Otherwise drop.

    A "general" intent with no filters is a no-op and passes through.
    """
    places_list = list(places)

    if (
        intent.query_kind == "general"
        and not intent.included_types
        and not intent.excluded_types
        and not intent.required_terms
    ):
        return places_list

    included = {t.lower() for t in intent.included_types}
    excluded = {t.lower() for t in intent.excluded_types}
    terms = [t.lower() for t in intent.required_terms]

    # Auto-expand: if the intent targets food specifically (any
    # *_restaurant, cafe, bakery, etc.) AND has no required_terms to
    # narrow results, also accept generic `restaurant`. Google often
    # labels real winners — diners, breakfast spots, brunch places —
    # with just `primary_type="restaurant"` rather than a specific food
    # subtype, and the strict list would otherwise drop them.
    #
    # Food queries WITH required_terms stay strict because the name/
    # summary term check (e.g. "pizza" in the name) already provides
    # the narrowing, so a generic restaurant without pizza evidence
    # still gets correctly dropped under a pizza query.
    if not terms and included & FOOD_SUBTYPES:
        included = included | {"restaurant"}

    kept: list[PlaceResult] = []
    for place in places_list:
        if _is_excluded_by_type(place, excluded):
            continue
        if _is_direct_type_match(place, included):
            kept.append(place)
            continue
        if terms and _place_matches_terms(place, terms):
            kept.append(place)
            continue
        if (
            intent.allow_generic_restaurant
            and _is_generic_restaurant(place)
            and not terms
        ):
            kept.append(place)
            continue
        if not (place.types or place.primary_type):
            if intent.strictness == "broad" and not terms:
                kept.append(place)
            continue
        # No match — drop.

    log.info(
        "intent.filter.done input=%d output=%d kind=%s included=%s",
        len(places_list),
        len(kept),
        intent.query_kind,
        sorted(included),
    )
    return kept


def _is_excluded_by_type(place: PlaceResult, excluded: set[str]) -> bool:
    if not excluded:
        return False
    primary = (place.primary_type or "").lower()
    if primary and primary in excluded:
        return True
    return bool(excluded.intersection(t.lower() for t in (place.types or [])))


def _is_direct_type_match(place: PlaceResult, included: set[str]) -> bool:
    if not included:
        return False
    primary = (place.primary_type or "").lower()
    if primary and primary in included:
        return True
    return bool(included.intersection(t.lower() for t in (place.types or [])))


def _place_matches_terms(place: PlaceResult, terms: list[str]) -> bool:
    """True if any `term` is a substring of the place name or any summary."""
    haystacks = [
        place.name or "",
        place.editorial_summary or "",
        place.review_summary or "",
        place.generative_summary or "",
    ]
    haystack = " ".join(h.lower() for h in haystacks if h)
    if not haystack:
        return False
    return any(term in haystack for term in terms)


_GENERIC_DISQUALIFIERS = {
    "cafe", "coffee_shop", "bakery", "bar", "pub", "fast_food_restaurant",
    "meal_delivery", "meal_takeaway",
}


def _is_generic_restaurant(place: PlaceResult) -> bool:
    """True if a place looks like an unspecified restaurant.

    Heuristic: primary_type is empty or literally "restaurant", AND the
    type list mentions "restaurant" somewhere, AND there is no more
    specific cuisine subtype (`*_restaurant` besides the bare
    "restaurant", or any of the `_GENERIC_DISQUALIFIERS`).
    """
    primary = (place.primary_type or "").lower()
    types = [t.lower() for t in (place.types or [])]
    if not types and primary != "restaurant":
        return False

    if primary and primary != "restaurant":
        return False

    if "restaurant" not in types and primary != "restaurant":
        return False

    for t in types:
        if t == "restaurant":
            continue
        if t.endswith("_restaurant"):
            return False  # e.g. italian_restaurant
        if t in _GENERIC_DISQUALIFIERS:
            return False

    return True
