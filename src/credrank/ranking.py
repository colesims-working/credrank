"""Bayesian ranking engine for place results."""

import statistics

from credrank.models import PlaceResult


def compute_prior(places: list[PlaceResult]) -> tuple[float, float]:
    """Compute the prior mean (m) and confidence constant (C) from the result set.

    m = mean rating across all rated places.
    C = 25th percentile of review counts (Q1), computed using Python's
        standard 'exclusive' interpolation method. This sets the credibility
        threshold: places with fewer reviews than C get pulled toward the
        population mean, while places with more reviews are trusted at
        close to their own rating.

    Returns:
        (m, C) tuple. Returns (0.0, 0.0) if no rated places exist.
    """
    rated = [
        p for p in places if p.rating is not None and p.user_rating_count is not None
    ]

    if not rated:
        return 0.0, 0.0

    m = statistics.mean(p.rating for p in rated)

    review_counts = [p.user_rating_count for p in rated]

    if len(review_counts) >= 2:
        c = statistics.quantiles(review_counts, n=4)[0]  # Q1, 25th percentile
    else:
        c = float(review_counts[0])

    return m, c


def bayesian_average(
    rating: float,
    review_count: int,
    prior_mean: float,
    confidence: float,
) -> float:
    """Compute the Bayesian average for a single place.

    Formula: (C * m + n * R) / (C + n)

    Where:
        C = confidence constant (pseudo-review count)
        m = prior mean rating (population average)
        n = actual review count for this place
        R = actual average rating for this place

    Interpretation: we start with C imaginary reviews at rating m,
    then let the place's real reviews pull the score away from that anchor.
    A place needs roughly C real reviews before its own average dominates.
    """
    if confidence + review_count == 0:
        return 0.0

    return (confidence * prior_mean + review_count * rating) / (
        confidence + review_count
    )


def bayesian_score_to_credscore(
    bayesian_avg: float, floor: float = 1.0, ceiling: float = 5.0
) -> float:
    """Map a Bayesian average (1-5 scale) to a 0-100 CredScore.

    Linear mapping where floor -> 0 and ceiling -> 100.
    Clamped to [0, 100].
    """
    if ceiling == floor:
        return 50.0

    score = (bayesian_avg - floor) / (ceiling - floor) * 100
    return max(0.0, min(100.0, score))


def rank_places(places: list[PlaceResult]) -> list[dict]:
    """Score and rank a list of PlaceResults using Bayesian averaging.

    Returns a list of dicts sorted by CredScore (descending), each containing:
        - place: the original PlaceResult
        - bayesian_avg: raw Bayesian average (1-5 scale)
        - credscore: normalized 0-100 score
        - rank: 1-indexed position

    Unrated places are appended at the end with a CredScore of 0.
    """
    if not places:
        return []

    m, c = compute_prior(places)

    scored = []
    unrated = []

    for place in places:
        if place.rating is None or place.user_rating_count is None:
            unrated.append(place)
            continue

        ba = bayesian_average(place.rating, place.user_rating_count, m, c)
        scored.append({"place": place, "bayesian_avg": ba})

    # Sort by Bayesian average descending, then review count descending
    scored.sort(
        key=lambda x: (x["bayesian_avg"], x["place"].user_rating_count or 0),
        reverse=True,
    )

    # Compute CredScore using the range of Bayesian averages in this result set
    if scored:
        ba_values = [s["bayesian_avg"] for s in scored]
        ba_min = min(ba_values)
        ba_max = max(ba_values)

        # Use result-set range if there's meaningful spread, otherwise use 1-5
        if ba_max - ba_min > 0.1:
            floor, ceiling = ba_min, ba_max
        else:
            floor, ceiling = 1.0, 5.0

        for entry in scored:
            entry["credscore"] = round(
                bayesian_score_to_credscore(entry["bayesian_avg"], floor, ceiling), 1
            )

    # Build final ranked list
    results = []
    for i, entry in enumerate(scored, 1):
        entry["rank"] = i
        results.append(entry)

    for place in unrated:
        results.append(
            {
                "place": place,
                "bayesian_avg": 0.0,
                "credscore": 0.0,
                "rank": len(results) + 1,
            }
        )

    return results
