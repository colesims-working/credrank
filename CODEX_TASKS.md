# Codex Task: Security, Testing, and Infrastructure Hardening

> Read AGENTS.md first for project context and conventions.

## Context
CredRank is a restaurant discovery engine. The codebase has been reviewed and needs
the following fixes before public deployment. Focus on correctness and minimal changes.

## Task 1 (P0): Add HTTP Timeouts to All External Calls

**Files:** `src/credrank/maps_client.py`, `src/credrank/llm.py`

- In `maps_client.py`, add `timeout=10` to the `httpx.Client()` constructor in both
  `search_places` and `get_place_details`.
- In `llm.py`, ensure the Gemini client call has a reasonable timeout. If using
  `google.genai`, add a timeout parameter to the `generate_content` call if supported,
  or wrap the call in a try/except that catches `TimeoutError` and returns an empty string.
- Do NOT change any function signatures or return types.

## Task 2 (P0): Fully Mock LLM Calls in test_api.py

**File:** `tests/test_api.py`

The `/api/search` endpoint now calls `get_relevant_types()` which calls the Gemini API.
The `/api/summarize` and `/api/recommend` endpoints call `summarize_place` and
`recommend_items` which also call Gemini.

- Add `@patch("credrank.api.get_relevant_types", return_value=None)` to all search tests.
  Returning `None` means "skip filtering" which is the graceful degradation path.
- Add `@patch("credrank.api.summarize_place", return_value="Great place")` to summarize tests.
- Add `@patch("credrank.api.recommend_items", return_value={"place_name": "Test", "place_id": "123", "summary": "Good", "recommendations": [], "avoid": []})` to recommend tests.
- Verify all tests pass with `uv run pytest -v` and NO network calls are made.

## Task 3 (P1): Add Structured Logging

**Files:** `src/credrank/api.py`, `src/credrank/maps_client.py`, `src/credrank/llm.py`,
`src/credrank/recommender.py`

- Add `import logging` and `log = logging.getLogger(__name__)` to every module that
  doesn't already have it.
- Log at INFO level: search queries, result counts, filter counts, LLM call durations.
- Log at WARNING level: LLM parse failures, API errors, timeout events.
- Log at DEBUG level: individual place results, subcircle details.
- Do NOT add print statements. Use the `logging` module exclusively.
- Add a basic logging config in api.py:
  ```python
  import logging
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
  ```

## Task 4 (P1): Harden Recommender JSON Parsing

**File:** `src/credrank/recommender.py`

The `recommend_items` function parses LLM output as JSON. If the LLM returns a non-dict
(e.g., a string or list), it crashes.

- Wrap the JSON parsing in a try/except.
- If parsing fails or the result is not a dict, return a fallback:
  ```python
  {"place_name": place.name, "place_id": place.place_id,
   "summary": "Could not analyze reviews.", "recommendations": [], "avoid": []}
  ```
- Also handle the case where `recommendations` or `avoid` keys are missing from the dict.
- Use `.get()` with defaults instead of direct key access.

## Task 5 (P1): Move Geocoding Server-Side

**New file:** Create a new endpoint in `src/credrank/api.py`:
```python
@app.get("/api/geocode")
def geocode(address: str):
    """Server-side geocoding to avoid exposing Maps API key to browser."""
    import httpx
    from credrank.config import get_api_key
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    resp = httpx.get(url, params={"address": address, "key": get_api_key()}, timeout=5)
    data = resp.json()
    if data.get("results"):
        loc = data["results"][0]["geometry"]["location"]
        return {"lat": loc["lat"], "lng": loc["lng"], "name": data["results"][0]["formatted_address"]}
    return {"lat": None, "lng": None, "name": None}
```

Also add a reverse geocode endpoint:
```python
@app.get("/api/reverse-geocode")
def reverse_geocode(lat: float, lng: float):
    """Reverse geocode coordinates to a human-readable name."""
    import httpx
    from credrank.config import get_api_key
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    resp = httpx.get(url, params={"latlng": f"{lat},{lng}", "key": get_api_key()}, timeout=5)
    data = resp.json()
    if data.get("results"):
        short = next((r for r in data["results"]
                       if any(t in r.get("types", []) for t in ["locality", "neighborhood", "sublocality"])),
                      data["results"][0])
        return {"name": short["formatted_address"]}
    return {"name": f"{lat:.3f}, {lng:.3f}"}
```

**Then update `frontend/index.html`:**
- Replace all calls to `https://maps.googleapis.com/maps/api/geocode/json` with calls to
  `/api/geocode?address=...` and `/api/reverse-geocode?lat=...&lng=...`
- Remove the `/api/config/maps-key` fetch from the geocoding function.
- Remove the `maps_key()` endpoint from `api.py` (or keep it behind an auth check for later).

## Task 6 (P2): Add Rate Limiting Middleware

**File:** `src/credrank/api.py`

Add `slowapi` for rate limiting:
```bash
uv add slowapi
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/search")
@limiter.limit("30/minute")
def search(req: SearchRequest, request: Request): ...

@app.post("/api/summarize")
@limiter.limit("10/minute")
def summarize(req: PlaceIdRequest, request: Request): ...

@app.post("/api/recommend")
@limiter.limit("10/minute")
def recommend(req: PlaceIdRequest, request: Request): ...
```

## Validation
After all changes:
1. `uv run pytest -v` — all tests pass, no network calls
2. `uv run uvicorn credrank.api:app --reload --port 8000` — server starts
3. Search works in browser at http://localhost:8000
4. No API keys visible in browser Network tab (after Task 5)
5. Check logs show structured output with timestamps
