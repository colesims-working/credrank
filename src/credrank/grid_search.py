"""Grid search: subdivide a large area for 60+ results.

Google caps at 60 results per text search query (20 per page × 3 pages).
To find more, we tile the search area with smaller overlapping circles,
search each one, and deduplicate by place_id.

Algorithm:
1. Pick a fixed MAX_SUB_RADIUS (2500m ≈ 1.5 mi) — small enough that Google
   returns most places in the area within 20 results.
2. Given the user's radius, compute hexagonal packing: a center circle
   plus concentric rings of 6, 12, 18... circles.
3. Search each sub-circle, deduplicate, filter to original radius.

Cost: each sub-circle = 1 Google Places API call ($0.032).
A 10-mile radius generates ~19 sub-circles = ~$0.61 per deep search.
"""

from __future__ import annotations

import logging
import math

from credrank.maps_client import _haversine_m, search_places
from credrank.models import PlaceResult, PlaceSearchParams

log = logging.getLogger(__name__)

# Maximum sub-circle radius. Smaller = more results per area but more API calls.
# 2500m (~1.5 mi) is the sweet spot: dense enough for urban areas,
# and Google typically returns all relevant places within this radius.
MAX_SUB_RADIUS_M = 2500

# Overlap factor: 0.75 means each circle overlaps neighbors by ~25%.
# This prevents gaps at boundaries where a place could be missed.
OVERLAP_FACTOR = 0.75


def _hex_ring_count(ring: int) -> int:
    """Number of circles in a hexagonal ring. Ring 0 = 1, ring 1 = 6, ring 2 = 12, etc."""
    return 1 if ring == 0 else 6 * ring


def _generate_subcircles(
    center_lat: float,
    center_lng: float,
    radius_m: int,
    max_sub_radius: int = MAX_SUB_RADIUS_M,
    overlap: float = OVERLAP_FACTOR,
) -> list[tuple[float, float, int]]:
    """Generate hexagonally-packed sub-circles covering the search area.

    Returns list of (lat, lng, sub_radius_m) tuples.
    """
    # If the area is already small, just search it directly
    if radius_m <= max_sub_radius * 1.5:
        return [(center_lat, center_lng, radius_m)]

    sub_radius = max_sub_radius
    # Step distance between circle centers
    step = sub_radius * 2 * overlap

    # How many rings do we need to cover the full radius?
    num_rings = max(1, math.ceil(radius_m / step))

    circles: list[tuple[float, float, int]] = []

    for ring in range(num_rings + 1):
        if ring == 0:
            circles.append((center_lat, center_lng, sub_radius))
            continue

        n_in_ring = _hex_ring_count(ring)
        ring_distance = step * ring

        for i in range(n_in_ring):
            angle_rad = (2 * math.pi * i) / n_in_ring
            d_lat = ring_distance * math.cos(angle_rad) / 111320
            d_lng = (
                ring_distance
                * math.sin(angle_rad)
                / (111320 * math.cos(math.radians(center_lat)))
            )
            circles.append(
                (
                    center_lat + d_lat,
                    center_lng + d_lng,
                    sub_radius,
                )
            )

    log.info(
        "Grid search: radius=%dm, sub_radius=%dm, rings=%d, subcircles=%d",
        radius_m,
        sub_radius,
        num_rings,
        len(circles),
    )
    return circles


def grid_search(
    query: str,
    center_lat: float,
    center_lng: float,
    radius_m: int,
    max_results: int = 200,
) -> list[PlaceResult]:
    """Search a large area by subdividing into overlapping grid cells.

    Deduplicates by place_id and filters to places within the original
    circle radius. Returns up to max_results unique places.
    """
    subcircles = _generate_subcircles(center_lat, center_lng, radius_m)

    seen_ids: set[str] = set()
    all_results: list[PlaceResult] = []

    for i, (lat, lng, sub_radius) in enumerate(subcircles):
        if len(all_results) >= max_results:
            log.info(
                "Grid search: hit max_results=%d after %d subcircles", max_results, i
            )
            break

        params = PlaceSearchParams(
            query=query,
            location=(lat, lng),
            radius_m=sub_radius,
            max_results=20,  # Single page per subcircle for speed
        )

        try:
            places = search_places(params)
        except Exception as e:
            log.warning("Grid search subcircle %d failed: %s", i, e)
            continue

        new_count = 0
        for place in places:
            if place.place_id in seen_ids:
                continue
            # Verify within the original circle
            if place.lat is not None and place.lng is not None:
                dist = _haversine_m(center_lat, center_lng, place.lat, place.lng)
                if dist > radius_m:
                    continue
            seen_ids.add(place.place_id)
            all_results.append(place)
            new_count += 1

        log.debug(
            "Subcircle %d/%d: %d results, %d new unique",
            i + 1,
            len(subcircles),
            len(places),
            new_count,
        )

    log.info("Grid search complete: %d unique results", len(all_results))
    return all_results[:max_results]
