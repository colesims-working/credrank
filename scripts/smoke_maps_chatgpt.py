# scripts/smoke_maps_chatgpt.py
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_MAPS_API_KEY in .env")

URL = "https://places.googleapis.com/v1/places:searchText"


def search_text(
    query: str, lat: float, lng: float, radius_m: int, max_results: int = 20
):
    headers = {
        "X-Goog-Api-Key": API_KEY,
        # Field mask is REQUIRED on Places API (New)
        "X-Goog-FieldMask": (
            "places.id,"
            "places.displayName,"
            "places.formattedAddress,"
            "places.rating,"
            "places.userRatingCount,"
            "places.location"
        ),
    }

    body = {
        "textQuery": query,
        "maxResultCount": max_results,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": float(radius_m),
            }
        },
    }

    r = httpx.post(URL, headers=headers, json=body, timeout=20)
    r.raise_for_status()
    data = r.json()

    places = []
    for p in data.get("places", []):
        places.append(
            {
                "place_id": p.get("id"),
                "name": (p.get("displayName") or {}).get("text"),
                "rating": p.get("rating"),
                "ratings_count": p.get("userRatingCount"),
                "address": p.get("formattedAddress"),
            }
        )
    return places


if __name__ == "__main__":
    print("Starting smoke test for Places API (New)...")
    query = "coffee"
    lat, lng = 37.5407, -77.4360
    radius_m = 2500
    max_results = 20

    results = search_text(query, lat, lng, radius_m, max_results=max_results)
    for i, p in enumerate(results, start=1):
        print(
            f"{i:>2}. {p['name']} | {p['rating']} ({p['ratings_count']}) | {p['address']}"
        )
