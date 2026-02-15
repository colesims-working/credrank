"""Quick manual test for Google Places search."""

from credrank.maps_client import search_places
from credrank.models import PlaceSearchParams

# --- Config (tweak these and re-run) ---
QUERY = "best pizza"
LAT, LNG = 37.5407, -77.4360  # Richmond, VA
RADIUS_M = 5000
MAX_RESULTS = 10


def main() -> None:
    params = PlaceSearchParams(
        query=QUERY,
        location=(LAT, LNG),
        radius_m=RADIUS_M,
        max_results=MAX_RESULTS,
    )

    print(f'Searching for "{QUERY}" near ({LAT}, {LNG})...\n')
    results = search_places(params)

    if not results:
        print("No results found.")
        return

    for i, place in enumerate(results, 1):
        rating = (
            f"{place.rating} ({place.user_rating_count})" if place.rating else "N/A"
        )
        print(f"{i:>2}. {place.name:<30} | {rating:<10} | {place.address}")


if __name__ == "__main__":
    main()
