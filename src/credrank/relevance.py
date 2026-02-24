"""LLM-powered relevance filtering for search results.

Uses a single Gemini call per search to determine which Google Place types
are relevant to the user's query, then filters results that don't match.
Covers ALL Google Place types — CredRank is not restaurant-only.
"""

from __future__ import annotations

import json
import logging

from credrank.llm import complete

log = logging.getLogger(__name__)

# Complete set of Google Place types (Places API New, 2025).
# Organized by category so the LLM has semantic context.
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
        "middle_eastern_restaurant", "pizza_restaurant", "ramen_restaurant",
        "restaurant", "sandwich_shop", "seafood_restaurant",
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

# Flat set for validation
ALL_TYPES: set[str] = set()
for _types in GOOGLE_PLACE_TYPES.values():
    ALL_TYPES.update(_types)

SYSTEM_PROMPT = """\
You determine which Google Place types are relevant to a search query.

Rules:
- Be INCLUSIVE. If a type COULD reasonably satisfy the query, include it.
- For food queries, always include 'restaurant' and 'meal_takeaway' as catch-alls.
- For ambiguous queries, err on the side of including more types.
- Only EXCLUDE types that are clearly unrelated.
- Respond ONLY with a JSON array of type strings. No explanation, no markdown."""


def get_relevant_types(query: str) -> list[str] | None:
    """Ask the LLM which Google Place types match this search query.

    Returns a list of relevant type strings, or None if the LLM call fails
    (in which case the caller should skip filtering — graceful degradation).
    """
    type_display = "\n".join(
        f"  {cat}: {', '.join(types)}"
        for cat, types in GOOGLE_PLACE_TYPES.items()
    )

    prompt = f"""Search query: "{query}"

Available Google Place types by category:
{type_display}

Which types could reasonably match what the user is looking for?
Return ONLY a JSON array of type strings."""

    try:
        raw = complete(prompt, system=SYSTEM_PROMPT, max_tokens=400)
        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        types = json.loads(cleaned)

        if isinstance(types, list) and len(types) > 0:
            validated = [t for t in types if isinstance(t, str) and t in ALL_TYPES]
            if validated:
                log.info(
                    "Relevance filter for '%s': %d types (from %d raw)",
                    query, len(validated), len(types),
                )
                return validated
            else:
                log.warning(
                    "LLM returned types not in taxonomy for '%s': %s",
                    query, types[:5],
                )
                return None

    except json.JSONDecodeError as e:
        log.warning("Relevance filter JSON parse error for '%s': %s", query, e)
    except Exception as e:
        log.warning("Relevance filter failed for '%s': %s", query, e)

    return None


def filter_by_relevance(places: list, relevant_types: list[str]) -> list:
    """Filter places to those with at least one matching type.

    Places with no types at all are kept (benefit of the doubt).
    """
    if not relevant_types:
        return places

    type_set = set(relevant_types)
    filtered = []
    for place in places:
        if not place.types:
            filtered.append(place)
        elif type_set.intersection(place.types):
            filtered.append(place)
    return filtered
