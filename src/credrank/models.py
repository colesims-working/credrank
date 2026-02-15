from dataclasses import dataclass, field


@dataclass
class PlaceSearchParams:
    """Input parameters for a Google Places search."""

    query: str
    location: tuple[float, float]  # (lat, lng)
    radius_m: int
    max_results: int = 20


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
