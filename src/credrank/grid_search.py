"""Adaptive grid search across a large area.

Google's Text Search (New) caps any single query at ~60 results (3 pages
of 20). To find more, we tile the search area with overlapping circles,
search each one, deduplicate, and return the union.

This module used to generate a fixed grid of small cells and stop as
soon as `max_results` was hit — which meant later subcircles might hide
the top-ranked result. The current approach:

1. **Start with larger initial cells** (`INITIAL_CELL_RADIUS_M`) so a
   non-dense query is done in a handful of requests.
2. **Subdivide saturated cells adaptively**: when a cell comes back with
   `saturated=True` (i.e. Google had more results it didn't return), split
   it into 7 smaller children (1 center + a hex ring of 6) and search
   them. Keep going until cells either stop saturating or reach the
   `MIN_CELL_RADIUS_M` floor.
3. **Never stop early on result count**. Every planned and adaptively
   generated cell is attempted. Truncation for display happens upstream
   in `api.py` after ranking.
4. **Safety cap**. A hard `MAX_CELL_COUNT` prevents unbounded blowups
   in pathologically dense areas; hitting it sets `coverage.complete=False`.

Returns a `GridSearchResult` — a list subclass carrying a `GridCoverage`
attribute that reports exactly what happened (cells searched, failed,
saturated, raw/deduped counts, and whether the run was complete).
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor

from credrank.maps_client import haversine_m, search_places
from credrank.models import (
    GridCoverage,
    GridSearchResult,
    PlaceResult,
    PlaceSearchParams,
)

log = logging.getLogger(__name__)

# Initial cell radius. Larger = fewer upfront requests for non-dense queries.
# Subdivision kicks in automatically when a cell saturates.
INITIAL_CELL_RADIUS_M = 6000

# Floor: never subdivide below this. Low enough that a 1-mile (1609m)
# user search can actually subdivide — at MIN=1500 a 1609m parent's
# children would be ~1045m (below the floor) and `_can_subdivide`
# returned False, so dense 1-mile searches returned 60 results and
# never drilled in. 500m keeps enough coverage per cell to be useful
# while allowing small-radius searches to adaptively split.
MIN_CELL_RADIUS_M = 500

# Safety cap — this is also the **cost** cap: each cell is up to 3 billed
# Text Search requests (~$0.035 each), so 60 cells is a ~$6 deep search
# worst case. Tune per deployment with CREDRANK_MAX_CELLS. Hitting the
# cap marks coverage as incomplete so the user is warned.
MAX_CELL_COUNT = int(os.environ.get("CREDRANK_MAX_CELLS", "60"))

# Cells within a wave are independent, so search them concurrently.
# Bounded to stay well under Google's default QPS quota while cutting
# deep-search wall-clock time roughly by the worker count.
MAX_PARALLEL_CELLS = int(os.environ.get("CREDRANK_GRID_WORKERS", "6"))

# Each split replaces one cell with 7 cells. Child radius is SPLIT_RATIO
# of parent; the ring distance is half of parent. Together these ensure
# the 7 children overlap enough to cover the parent area without gaps.
#
# The worst-case point on the parent's outer edge at angle 30° (between
# two adjacent ring children) is at distance `R * sqrt((√3/2 − 1/2)² + 1/4)`
# ≈ 0.62 R from the nearest child center, so SPLIT_RATIO must be > 0.62
# or the parent isn't fully tiled. 0.65 adds a small safety margin.
SPLIT_RATIO = 0.65
SPLIT_RING_FACTOR = 0.5

# Overlap factor for the initial hex packing: 0.75 means each cell
# overlaps neighbors by ~25%, preventing boundary gaps.
OVERLAP_FACTOR = 0.75


def _hex_ring_count(ring: int) -> int:
    """Number of circles in a hexagonal ring. Ring 0 = 1, ring 1 = 6, ring 2 = 12."""
    return 1 if ring == 0 else 6 * ring


def generate_subcircles(
    center_lat: float,
    center_lng: float,
    radius_m: int,
    max_sub_radius: int = INITIAL_CELL_RADIUS_M,
    overlap: float = OVERLAP_FACTOR,
) -> list[tuple[float, float, int]]:
    """Generate hexagonally-packed sub-circles covering the search area.

    Returns a list of `(lat, lng, sub_radius_m)` tuples. When the total
    search area already fits within ~1.5x `max_sub_radius`, returns a
    single cell at the full radius so we don't pointlessly subdivide.
    """
    if radius_m <= max_sub_radius * 1.5:
        return [(center_lat, center_lng, radius_m)]

    sub_radius = max_sub_radius
    step = sub_radius * 2 * overlap
    # Each ring's outer edge reaches `ring * step + sub_radius` from the
    # user center, so we only need enough rings for that to cover
    # `radius_m`. The older `ceil(radius_m / step)` ignored the
    # `sub_radius` term and over-provisioned — e.g., 19 cells for a
    # 12km radius where 7 cells fully cover the user's disk.
    num_rings = max(1, math.ceil((radius_m - sub_radius) / step))

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
        "grid.subcircles radius=%dm cell_radius=%dm rings=%d cells=%d",
        radius_m,
        sub_radius,
        num_rings,
        len(circles),
    )
    return circles


def _split_cell(
    lat: float,
    lng: float,
    radius_m: int,
) -> list[tuple[float, float, int]]:
    """Split a saturated parent cell into 6 smaller child cells.

    Layout: 6 hex-ring children at distance `SPLIT_RING_FACTOR * radius`
    from the parent center. Each child has radius `SPLIT_RATIO * radius`.

    **No concentric "center" child.** Every ring child's own area already
    contains the parent center (the ring distance is `0.5 R` and the
    child radius is `0.65 R`, so the parent center sits `0.5 R` inside
    the ring child — well within its radius). Adding a concentric center
    child produced a nested-saturation cascade in dense areas: the first
    child of every split was at the exact same user center point, always
    re-hit the 60-result cap, and triggered yet another split at the
    same point with a smaller radius, ad infinitum.
    """
    child_radius = max(int(radius_m * SPLIT_RATIO), MIN_CELL_RADIUS_M)
    ring_distance = radius_m * SPLIT_RING_FACTOR

    children: list[tuple[float, float, int]] = []
    for i in range(6):
        angle = 2 * math.pi * i / 6
        d_lat = ring_distance * math.cos(angle) / 111320
        d_lng = (
            ring_distance
            * math.sin(angle)
            / (111320 * math.cos(math.radians(lat)))
        )
        children.append((lat + d_lat, lng + d_lng, child_radius))
    return children


def _can_subdivide(cell_radius: int) -> bool:
    """True iff splitting this cell would produce children above the floor."""
    if cell_radius <= MIN_CELL_RADIUS_M:
        return False
    return int(cell_radius * SPLIT_RATIO) >= MIN_CELL_RADIUS_M


def grid_search(
    query: str,
    center_lat: float,
    center_lng: float,
    radius_m: int,
    included_type: str | None = None,
) -> GridSearchResult:
    """Search a large area with adaptive subdivision.

    Starts with coarse initial cells; wherever Google tells us a cell
    was saturated (more results available than it returned), splits it
    into finer children and searches them too. Never stops early on
    result count — the caller is responsible for ranking and truncating
    for display.

    `included_type`, when set, is pushed to each cell's Text Search
    request as `includedType` + `strictTypeFiltering=True`. The caller
    is responsible for ensuring the value is a valid Table A type.

    Cells are searched in **waves**: everything currently in the
    frontier runs concurrently (bounded by `MAX_PARALLEL_CELLS`), then
    the saturated cells' children form the next wave. Cells within a
    wave never depend on each other, so this preserves the sequential
    version's behavior while cutting wall-clock time by roughly the
    worker count — deep searches used to take minutes purely from
    serialized round-trips.

    Returns a `GridSearchResult` containing unique places in-radius plus
    a `GridCoverage` attribute with detailed run metadata.
    """
    frontier: list[tuple[float, float, int]] = list(
        generate_subcircles(center_lat, center_lng, radius_m)
    )

    seen_ids: set[str] = set()
    candidates: list[PlaceResult] = []
    raw_count = 0
    cells_searched = 0
    cells_failed = 0
    cells_saturated = 0
    cells_saturated_at_min = 0
    hit_cell_cap = False

    def _search_cell(cell: tuple[float, float, int]):
        lat, lng, cell_radius = cell
        params = PlaceSearchParams(
            query=query,
            location=(lat, lng),
            radius_m=cell_radius,
            max_results=60,
            included_type=included_type,
            strict_type_filtering=bool(included_type),
        )
        try:
            return search_places(params)
        except Exception as exc:  # captured per-cell; one failure ≠ dead run
            return exc

    while frontier:
        budget = MAX_CELL_COUNT - cells_searched
        if budget <= 0:
            hit_cell_cap = True
            log.warning(
                "grid.cap query=%r cells_searched=%d pending=%d",
                query,
                cells_searched,
                len(frontier),
            )
            break

        wave, frontier = frontier[:budget], frontier[budget:]
        cells_searched += len(wave)

        if len(wave) == 1:
            outcomes = [_search_cell(wave[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=min(MAX_PARALLEL_CELLS, len(wave))
            ) as pool:
                outcomes = list(pool.map(_search_cell, wave))

        # Process outcomes in wave order so dedup stays deterministic.
        for (lat, lng, cell_radius), outcome in zip(wave, outcomes):
            if isinstance(outcome, Exception):
                cells_failed += 1
                log.warning(
                    "grid.cell.error radius=%d error=%s", cell_radius, outcome
                )
                continue

            sub_result = outcome
            raw_count += len(sub_result)

            new_count = 0
            for place in sub_result:
                if not place.place_id or place.place_id in seen_ids:
                    continue
                seen_ids.add(place.place_id)
                candidates.append(place)
                new_count += 1

            sub_saturated = bool(getattr(sub_result, "saturated", False))
            log.debug(
                "grid.cell radius=%d new=%d unique=%d saturated=%s",
                cell_radius,
                new_count,
                len(candidates),
                sub_saturated,
            )

            if sub_saturated:
                cells_saturated += 1
                if _can_subdivide(cell_radius):
                    frontier.extend(_split_cell(lat, lng, cell_radius))
                else:
                    cells_saturated_at_min += 1

    # Filter to the user's original radius.
    in_radius: list[PlaceResult] = []
    for p in candidates:
        if p.lat is not None and p.lng is not None:
            dist = haversine_m(center_lat, center_lng, p.lat, p.lng)
            if dist <= radius_m:
                in_radius.append(p)
        else:
            in_radius.append(p)

    coverage = _build_coverage(
        cells_searched=cells_searched,
        cells_failed=cells_failed,
        cells_saturated=cells_saturated,
        cells_saturated_at_min=cells_saturated_at_min,
        hit_cell_cap=hit_cell_cap,
        raw_count=raw_count,
        deduped_count=len(candidates),
        in_radius_count=len(in_radius),
    )

    log.info(
        "grid.done query=%r cells=%d failed=%d saturated=%d unique=%d "
        "in_radius=%d complete=%s",
        query,
        cells_searched,
        cells_failed,
        cells_saturated,
        len(candidates),
        len(in_radius),
        coverage.complete,
    )
    return GridSearchResult(in_radius, coverage=coverage)


def _build_coverage(
    *,
    cells_searched: int,
    cells_failed: int,
    cells_saturated: int,
    cells_saturated_at_min: int,
    hit_cell_cap: bool,
    raw_count: int,
    deduped_count: int,
    in_radius_count: int,
) -> GridCoverage:
    """Assemble a `GridCoverage` from run counters, computing `complete`."""
    reasons: list[str] = []
    if cells_failed > 0:
        reasons.append(f"{cells_failed} cell(s) failed")
    if cells_saturated_at_min > 0:
        reasons.append(
            f"{cells_saturated_at_min} saturated cell(s) reached minimum radius"
        )
    if hit_cell_cap:
        reasons.append(f"hit safety cap of {MAX_CELL_COUNT} cells")

    return GridCoverage(
        complete=not reasons,
        reason="; ".join(reasons) if reasons else None,
        cells_searched=cells_searched,
        cells_failed=cells_failed,
        cells_saturated=cells_saturated,
        raw_count=raw_count,
        deduped_count=deduped_count,
        in_radius_count=in_radius_count,
    )
