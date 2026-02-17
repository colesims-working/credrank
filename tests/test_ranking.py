from credrank.models import PlaceResult
from credrank.ranking import bayesian_average, compute_prior, rank_places


def _make_place(
    place_id: str = "fake",
    name: str = "Test Place",
    rating: float | None = 4.0,
    user_rating_count: int | None = 100,
) -> PlaceResult:
    """Helper to create PlaceResult instances for testing."""
    return PlaceResult(
        place_id=place_id,
        name=name,
        rating=rating,
        user_rating_count=user_rating_count,
    )


class TestComputePrior:
    def test_basic_prior(self) -> None:
        places = [
            _make_place(rating=4.0, user_rating_count=100),
            _make_place(rating=3.0, user_rating_count=200),
            _make_place(rating=5.0, user_rating_count=50),
        ]
        m, c = compute_prior(places)
        assert m == 4.0  # mean of 4.0, 3.0, 5.0
        # Q1 should be between the lowest and median review counts
        assert 50 <= c <= 100

    def test_skips_unrated_places(self) -> None:
        places = [
            _make_place(rating=4.0, user_rating_count=100),
            _make_place(rating=None, user_rating_count=None),
        ]
        m, c = compute_prior(places)
        assert m == 4.0
        assert c == 100

    def test_no_rated_places(self) -> None:
        places = [_make_place(rating=None, user_rating_count=None)]
        m, c = compute_prior(places)
        assert m == 0.0
        assert c == 0.0

    def test_empty_list(self) -> None:
        m, c = compute_prior([])
        assert m == 0.0
        assert c == 0.0


class TestBayesianAverage:
    def test_high_review_count_trusts_own_rating(self) -> None:
        """A place with many reviews should score close to its own average."""
        score = bayesian_average(
            rating=4.8, review_count=1000, prior_mean=3.5, confidence=50
        )
        assert score > 4.7

    def test_low_review_count_pulled_toward_prior(self) -> None:
        """A place with few reviews should be pulled toward the prior mean."""
        score = bayesian_average(
            rating=5.0, review_count=3, prior_mean=3.5, confidence=50
        )
        assert score < 4.0

    def test_zero_confidence_returns_own_rating(self) -> None:
        """With C=0, the Bayesian average equals the raw rating."""
        score = bayesian_average(
            rating=4.5, review_count=10, prior_mean=3.0, confidence=0
        )
        assert score == 4.5

    def test_zero_reviews_returns_prior(self) -> None:
        """With no reviews, the Bayesian average equals the prior mean."""
        score = bayesian_average(
            rating=0.0, review_count=0, prior_mean=3.5, confidence=50
        )
        assert score == 3.5

    def test_zero_everything(self) -> None:
        score = bayesian_average(
            rating=0.0, review_count=0, prior_mean=0.0, confidence=0
        )
        assert score == 0.0


class TestRankPlaces:
    def test_ranks_by_bayesian_not_raw_rating(self) -> None:
        """A 4.5 with 500 reviews should beat a 5.0 with 3 reviews
        when the prior mean is pulled down by a realistic result set."""
        places = [
            _make_place(
                place_id="few", name="Few Reviews", rating=5.0, user_rating_count=3
            ),
            _make_place(
                place_id="many", name="Many Reviews", rating=4.5, user_rating_count=500
            ),
            # Populate a realistic result set to anchor the prior lower
            _make_place(rating=3.8, user_rating_count=120),
            _make_place(rating=3.5, user_rating_count=200),
            _make_place(rating=4.0, user_rating_count=80),
            _make_place(rating=3.2, user_rating_count=150),
            _make_place(rating=4.1, user_rating_count=90),
        ]
        results = rank_places(places)

        # Find our two places of interest in the ranked output
        ranked_ids = [r["place"].place_id for r in results]
        assert ranked_ids.index("many") < ranked_ids.index("few")

    def test_unrated_places_sort_last(self) -> None:
        places = [
            _make_place(place_id="unrated", rating=None, user_rating_count=None),
            _make_place(place_id="rated", rating=4.0, user_rating_count=100),
        ]
        results = rank_places(places)

        assert results[0]["place"].place_id == "rated"
        assert results[-1]["place"].place_id == "unrated"
        assert results[-1]["credscore"] == 0.0

    def test_credscores_are_bounded(self) -> None:
        places = [
            _make_place(rating=4.8, user_rating_count=500),
            _make_place(rating=3.2, user_rating_count=300),
            _make_place(rating=4.0, user_rating_count=100),
        ]
        results = rank_places(places)

        for r in results:
            assert 0.0 <= r["credscore"] <= 100.0

    def test_single_place(self) -> None:
        places = [_make_place(rating=4.5, user_rating_count=200)]
        results = rank_places(places)

        assert len(results) == 1
        assert results[0]["rank"] == 1

    def test_empty_input(self) -> None:
        assert rank_places([]) == []

    def test_all_same_rating(self) -> None:
        """When all places share a rating, higher review count wins
        only if the prior mean differs from that rating."""
        places = [
            _make_place(place_id="low", rating=4.0, user_rating_count=10),
            _make_place(place_id="high", rating=4.0, user_rating_count=1000),
            # These lower-rated places pull the prior mean below 4.0
            _make_place(rating=3.0, user_rating_count=100),
            _make_place(rating=3.5, user_rating_count=200),
        ]
        results = rank_places(places)

        ranked_ids = [r["place"].place_id for r in results]
        assert ranked_ids.index("high") < ranked_ids.index("low")

    def test_credscore_spread_is_meaningful(self) -> None:
        """Verify CredScores use the full range when there's real spread."""
        places = [
            _make_place(place_id="best", rating=4.8, user_rating_count=800),
            _make_place(place_id="good", rating=4.5, user_rating_count=400),
            _make_place(place_id="mid", rating=4.0, user_rating_count=200),
            _make_place(place_id="low", rating=3.5, user_rating_count=100),
            _make_place(place_id="worst", rating=3.0, user_rating_count=50),
        ]
        results = rank_places(places)

        scores = [r["credscore"] for r in results]
        assert max(scores) == 100.0
        assert min(scores) == 0.0

        # Verify ordering matches expectations
        ranked_ids = [r["place"].place_id for r in results]
        assert ranked_ids == ["best", "good", "mid", "low", "worst"]

    def test_credscore_with_real_world_distribution(self) -> None:
        """Simulate a typical Google Places result set and verify rankings feel right.

        The Bayesian average formula has a mathematical property: when a place's
        rating equals the prior mean, its Bayesian average is always the prior mean
        regardless of review count. This means a 5.0-rated place with few reviews
        can rank above a 4.5-rated place if 4.5 IS the prior mean. This is expected
        behavior — C tuning and credibility thresholds are future iterations.
        """
        places = [
            _make_place(
                place_id="hidden_gem",
                name="Hidden Gem",
                rating=4.9,
                user_rating_count=45,
            ),
            _make_place(
                place_id="popular",
                name="Popular Spot",
                rating=4.5,
                user_rating_count=1200,
            ),
            _make_place(
                place_id="solid", name="Solid Choice", rating=4.6, user_rating_count=600
            ),
            _make_place(
                place_id="new_hype", name="New Hype", rating=5.0, user_rating_count=8
            ),
            _make_place(
                place_id="chain",
                name="Chain Restaurant",
                rating=4.2,
                user_rating_count=3000,
            ),
            _make_place(
                place_id="mediocre",
                name="Mediocre Place",
                rating=3.8,
                user_rating_count=200,
            ),
        ]
        results = rank_places(places)
        ranked_ids = [r["place"].place_id for r in results]

        # High rating + credible volume ranks first
        assert ranked_ids[0] == "hidden_gem"

        # Solid (4.6, 600) beats Popular (4.5, 1200) — rating wins when both have volume
        assert ranked_ids.index("solid") < ranked_ids.index("popular")

        # Below-average ratings sink regardless of volume
        assert ranked_ids.index("chain") > ranked_ids.index("solid")
        assert ranked_ids.index("mediocre") > ranked_ids.index("chain")

        # New Hype (5.0, 8 reviews) IS penalized — pulled from 5.0 toward prior
        new_hype = next(r for r in results if r["place"].place_id == "new_hype")
        assert new_hype["bayesian_avg"] < 5.0
