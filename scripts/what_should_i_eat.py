"""Given a place name, analyze reviews for item recommendations."""

from credrank.maps_client import search_places
from credrank.models import PlaceSearchParams
from credrank.recommender import recommend_items

# --- Change these to whatever place you're curious about ---
PLACE_NAME = "Mel's"
LAT, LNG = 12.517, -69.978  # Aruba
RADIUS_M = 10000


def main() -> None:
    params = PlaceSearchParams(
        query=PLACE_NAME,
        location=(LAT, LNG),
        radius_m=RADIUS_M,
        max_results=1,
    )

    print(f'Looking up "{PLACE_NAME}"...\n')
    places = search_places(params)

    if not places:
        print("Place not found.")
        return

    place = places[0]
    print(f"Found: {place.name}")
    print(f"Rating: {place.rating} ({place.user_rating_count} reviews)")
    print(f"Reviews available: {len(place.reviews)}")
    print(f"Has review summary: {'Yes' if place.review_summary else 'No'}")
    print(f"Has generative summary: {'Yes' if place.generative_summary else 'No'}")
    print()

    print("Analyzing reviews...\n")
    result = recommend_items(place)

    print(f"--- {result['place_name']} ---\n")
    print(f"Summary: {result['summary']}\n")

    if result["recommendations"]:
        print("What to order:")
        for i, rec in enumerate(result["recommendations"], 1):
            mentions = f" (mentioned {rec['mentions']}x)" if rec.get("mentions") else ""
            print(f"  {i}. {rec['item']}{mentions} -- {rec['why']}")

    if result.get("avoid"):
        print("\nMaybe skip:")
        for item in result["avoid"]:
            print(f"  - {item['item']} -- {item['why']}")


if __name__ == "__main__":
    main()
