from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Review:
    """A single user review from Google Places."""

    rating: int
    text: str
    author: str = ""


@dataclass
class PlaceSearchParams:
    """Input parameters for a Google Places text search."""

    query: str
    location: tuple[float, float]  # (lat, lng)
    radius_m: int
    max_results: int = 60
    # Optional Google Places (New) server-side type filter. When set, Google
    # biases results toward this Table A type. `strict_type_filtering=True`
    # asks Google to only return results matching this type.
    included_type: str | None = None
    strict_type_filtering: bool = False


@dataclass
class SearchIntent:
    """Structured interpretation of a user's search query.

    Produced by `relevance.build_intent()` — either via an LLM call or a
    deterministic fallback. Used for two things:

    1. Google-side filtering (`included_types` feed `includedType` +
       `strictTypeFiltering` in Text Search requests).
    2. Client-side post-filtering (`filter_by_intent` uses the whole
       intent to decide which ambiguous results to keep).
    """

    # "specific_food" | "cuisine" | "service" | "general"
    query_kind: str = "general"
    # Google Places Table A types we can filter on server-side.
    included_types: list[str] = field(default_factory=list)
    # Types whose presence as primary_type should drop a place.
    excluded_types: list[str] = field(default_factory=list)
    # Substrings we look for in the place name / details summaries to
    # accept ambiguous matches (e.g. a generic "restaurant" whose name
    # includes "pizza").
    required_terms: list[str] = field(default_factory=list)
    # If True, generic `restaurant`-typed results are eligible. Usually
    # True for food queries (because Google often labels real matches
    # as generic) and False for non-food queries (where we want hard
    # type filtering).
    allow_generic_restaurant: bool = False
    # "strict" drops no-type places and unmatched types outright.
    # "broad" gives ambiguous places the benefit of the doubt.
    strictness: str = "broad"


@dataclass
class PlaceResult:
    """Normalized result from the Google Places API."""

    place_id: str
    name: str
    rating: float | None = None
    user_rating_count: int | None = None
    address: str | None = None
    lat: float | None = None
    lng: float | None = None
    types: list[str] = field(default_factory=list)
    primary_type: str | None = None
    reviews: list[Review] = field(default_factory=list)
    editorial_summary: str | None = None
    review_summary: str | None = None
    generative_summary: str | None = None


class PlacesSearchResult(list):
    """List of PlaceResult with pagination metadata attached.

    Subclasses list so existing call sites that iterate, index, or `len()`
    the result keep working unchanged. Two extra attributes surface the
    state of the underlying pagination:

    - `saturated`: True if more results were available when the loop
      stopped — either we hit the configured page cap while Google was
      still returning a nextPageToken, or we reached `max_results` while
      a nextPageToken was still pending. Equivalent to "the user could
      see more by widening the search or bumping the page cap."
    - `page_count`: number of Text Search HTTP requests actually made.

    `__repr__` is overridden for readable logs/debugging.
    """

    def __init__(
        self,
        places: Iterable[PlaceResult] | None = None,
        *,
        saturated: bool = False,
        page_count: int = 0,
    ) -> None:
        super().__init__(places or [])
        self.saturated = saturated
        self.page_count = page_count

    def __repr__(self) -> str:
        return (
            f"PlacesSearchResult(n={len(self)}, "
            f"saturated={self.saturated}, page_count={self.page_count})"
        )


@dataclass
class GridCoverage:
    """Coverage metadata for a grid search run.

    `complete=False` signals that the result set may be missing places.
    `reason` is a human-readable explanation (cell failures, saturated
    cells that hit the minimum radius, safety cap hit).

    Frontend code reads this to decide whether to warn the user that
    "broader coverage wasn't possible within API limits."
    """

    complete: bool = True
    reason: str | None = None
    cells_searched: int = 0
    cells_failed: int = 0
    cells_saturated: int = 0
    raw_count: int = 0
    deduped_count: int = 0
    in_radius_count: int = 0


class GridSearchResult(list):
    """List of PlaceResult with `coverage: GridCoverage` attached.

    Subclasses list so api.py, tests, and any other consumer can keep
    iterating / indexing / len()-ing the result as before while still
    being able to read `.coverage` for metadata. Filtering or copying
    this through normal list operations will strip the coverage — grab
    it off the original result before any transformations.
    """

    def __init__(
        self,
        places: Iterable[PlaceResult] | None = None,
        *,
        coverage: GridCoverage | None = None,
    ) -> None:
        super().__init__(places or [])
        self.coverage = coverage if coverage is not None else GridCoverage()

    def __repr__(self) -> str:
        return (
            f"GridSearchResult(n={len(self)}, complete={self.coverage.complete}, "
            f"cells_searched={self.coverage.cells_searched})"
        )
