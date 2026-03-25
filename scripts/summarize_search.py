"""Full pipeline: Search → Rank → Summarize top results."""

from credrank.maps_client import search_places
from credrank.models import PlaceSearchParams
from credrank.ranking import rank_places
from credrank.summarizer import summarize_top_n

QUERY = "dry needling"
LAT, LNG = 37.603798, -77.260521  # Mechanicsville, VA
RADIUS_MI = 20
RADIUS_M = RADIUS_MI * 1609
MAX_RESULTS = 60
SUMMARIZE_TOP = 5


def main() -> None:
    params = PlaceSearchParams(
        query=QUERY,
        location=(LAT, LNG),
        radius_m=RADIUS_M,
        max_results=MAX_RESULTS,
    )

    print(f'Searching for "{QUERY}"...')
    places = search_places(params)
    print(
        f"Found {len(places)} results. Ranking and summarizing top {SUMMARIZE_TOP}...\n"
    )

    ranked = rank_places(places)
    ranked = summarize_top_n(ranked, n=SUMMARIZE_TOP)

    print(f"{'#':<4} {'Score':<7} {'Name':<35} {'Rating':<14} {'Summary'}")
    print("-" * 120)

    for entry in ranked:
        place = entry["place"]
        cs = entry["credscore"]
        summary = entry.get("summary", "")

        if place.rating is not None:
            rating_str = f"{place.rating} ({place.user_rating_count:,})"
        else:
            rating_str = "N/A"

        print(
            f"#{entry['rank']:<3} {cs:>5.1f}  {place.name:<35} {rating_str:<14} {summary}"
        )


if __name__ == "__main__":
    main()
