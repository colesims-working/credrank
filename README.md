# CredRank

A restaurant discovery engine that ranks places using **Bayesian statistics** instead of raw Google ratings. It answers two questions: *"Where should I eat?"* and *"What should I eat?"*

The problem: Google's star ratings treat a 4.8 with 12 reviews the same as a 4.8 with 2,000 reviews. CredRank doesn't — it uses a Bayesian average that weights ratings by review volume, so places with genuine credibility rise to the top.

## How It Works

1. **Search** — Queries the Google Places API (New) for places matching a text query within a radius
2. **Filter** — Uses Gemini Flash to determine which Google Place types are relevant, then removes noise (e.g., a gas station result when you searched for "pizza")
3. **Rank** — Computes a Bayesian average using Q1 review count as the credibility threshold, then maps it to a 0–100 CredScore
4. **Summarize** *(optional)* — Generates one-line LLM summaries from reviews and Google's own review/editorial summaries
5. **Recommend** *(optional)* — Analyzes reviews to extract specific dishes/items worth ordering (or avoiding)

### The Ranking Math

$$\text{CredScore} = \frac{C \cdot m + n \cdot R}{C + n}$$

Where $C$ = 25th percentile of review counts (credibility threshold), $m$ = population mean rating, $n$ = this place's review count, $R$ = this place's average rating. Places with fewer reviews than $C$ get pulled toward the population mean; places with more reviews are trusted at close to their own rating.

## Architecture

```
frontend/index.html    → Single-page app (vanilla HTML/JS/CSS, Leaflet maps)
src/credrank/
  api.py               → FastAPI backend, all endpoints
  maps_client.py       → Google Places API (New) integration
  models.py            → PlaceResult, PlaceSearchParams dataclasses
  ranking.py           → Bayesian ranking engine (Q1 credibility threshold)
  llm.py               → Gemini Flash client wrapper
  summarizer.py        → "Where should I eat?" one-line summaries
  recommender.py       → "What should I eat?" item recommendations
  relevance.py         → LLM-powered type filtering per search
  grid_search.py       → Hexagonal grid subdivision for 60+ results
  config.py            → Environment variable loading
tests/                 → pytest, all mocked (no live API calls)
```

## Setup

1. Copy `.env.example` to `.env` and add your API keys:
   ```
   GOOGLE_MAPS_API_KEY="your_key"
   GOOGLE_GEMINI_API_KEY="your_key"
   ```
2. Install dependencies:
   ```bash
   uv sync
   ```

## Running

```bash
# Dev server (hot-reload)
uv run python main.py

# Or with Docker Compose
docker compose up
```

The app serves at `http://localhost:8000`.

## Tests

```bash
uv run pytest -v
```

All tests are fully mocked — no live API calls to Google or Gemini.

## Tech Stack

- **Backend:** Python 3.12+, FastAPI, httpx, Pydantic
- **LLM:** Google Gemini Flash (relevance filtering, summaries, recommendations)
- **Frontend:** Vanilla HTML/JS/CSS, Leaflet.js maps
- **Testing:** pytest, respx (HTTP mocking), unittest.mock
- **Packaging:** uv, hatchling
- **Infrastructure:** Docker multi-stage build, Docker Compose for dev
