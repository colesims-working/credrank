"""Search + Bayesian ranking pipeline. The core CredRank workflow."""

from credrank.maps_client import search_places
from credrank.models import PlaceSearchParams
from credrank.ranking import rank_places

# --- Config (tweak these and re-run) ---
QUERY = "chain fast casual takeout dinner"
LAT, LNG = 37.603, -77.261  # Richmond, VA
RADIUS_M = 10000
MAX_RESULTS = 60


def main() -> None:
    params = PlaceSearchParams(
        query=QUERY,
        location=(LAT, LNG),
        radius_m=RADIUS_M,
        max_results=MAX_RESULTS,
    )

    print(f'Searching for "{QUERY}" near ({LAT}, {LNG})...')
    places = search_places(params)
    print(f"Found {len(places)} results. Ranking...\n")

    ranked = rank_places(places)

    print(f"{'#':<4} {'CredScore':<11} {'Name':<35} {'Rating':<14} {'Address'}")
    print("-" * 110)

    for entry in ranked:
        place = entry["place"]
        cs = entry["credscore"]

        if place.rating is not None:
            rating_str = f"{place.rating} ({place.user_rating_count:,})"
        else:
            rating_str = "N/A"

        print(
            f"#{entry['rank']:<3} {cs:>5.1f}      {place.name:<35} {rating_str:<14} {place.address}"
        )


if __name__ == "__main__":
    main()
