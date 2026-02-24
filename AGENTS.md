# AGENTS.md — CredRank Project Guide for AI Assistants

> Read this file FIRST before making any changes to this codebase.

## What is CredRank?

CredRank is a restaurant discovery engine that ranks places using Bayesian statistics instead of raw Google ratings. It answers two questions: "Where should I eat?" and "What should I eat?" The product differentiator is transparent, math-based rankings with LLM-powered review analysis.

**Target user:** Someone standing on a street corner, hungry, wanting an honest answer fast — on their phone.

**Business model:** Freemium SaaS. Free tier: 5 searches/day. Pro tier ($9.99/month): unlimited searches, deep search (grid), LLM features (summarize, recommendations), drive/walk time radius, corridor search.

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
tests/                 → pytest, all mocked (NO live API calls in tests)
scripts/               → Manual smoke tests and utilities
```

## Critical Conventions

### Security
- **NEVER** expose API keys to the frontend without referrer restrictions in production
- **ALWAYS** escape user input before HTML rendering (use `escapeHtml()` in frontend)
- **ALWAYS** validate and sanitize on the API boundary (Pydantic models)
- **CORS** must be tightened before production deploy (not `allow_origins=["*"]`)
- Rate limiting is required on all LLM-consuming endpoints before public launch

### Testing
- **ALL tests must be fully mocked.** No live API calls to Google, Gemini, or any external service.
- Mock at the function boundary: `search_places`, `get_place_details`, `complete`, `summarize_place`, `recommend_items`, `get_relevant_types`
- Test properties and ranges, not exact float equality on statistical functions
- Run: `uv run pytest -v`

### API Cost Awareness
- Google Places: $0.032 per text search, $0.02 per Place Details (Enterprise SKU)
- Google Atmosphere SKU (reviews): Only fetched on-demand via summarize/recommend, never on search
- Gemini Flash: ~$0.001 per relevance filter call, ~$0.002 per summarize, ~$0.005 per recommend
- Grid search multiplies Places cost by number of subcircles (7-19x)
- **Every new feature must document its API cost impact**

### Code Style
- Python 3.12+, type hints everywhere, dataclasses for models
- FastAPI + Pydantic for API boundary validation
- `httpx` for HTTP (not `requests`)
- Single-file frontend for now (will migrate to React when complexity demands it)
- Graceful degradation: if an LLM call fails, skip that feature, don't crash

### Ranking Algorithm
The Bayesian average uses the 25th percentile (Q1) of review counts as the credibility threshold `C`, and the population mean as the prior `m`. This means:
- Places with ~30-50+ reviews have their own rating dominate
- Places with very few reviews are pulled toward the population mean
- **Do NOT change the ranking algorithm without explicit approval** — it's the core product

### Frontend
- Mobile-first, dark theme, editorial design language (Instrument Serif + DM Sans)
- Colorblind-safe: scores use both color AND shape indicators (▲ ● ◆ ▽ ✕)
- WCAG 2.1 AA compliance: aria labels, semantic HTML, keyboard navigation, prefers-reduced-motion
- Leaflet.js for maps (free, no API key needed)
- Search bar is a unified pill: query | location | 📍 | radius | Search

## Deployment Target
- Railway (container hosting)
- PostgreSQL on Railway (users, saved searches, feedback)
- Clerk for auth (Google sign-in)
- Stripe for payments ($0 beta → $9.99 Pro)
- Cloudflare DNS for credrank.org

## Current Sprint Goals
1. Fix grid search to return meaningfully more results (smaller sub-circles, dynamic packing)
2. LLM relevance filtering covering all Google Place types (not just food)
3. Accessibility: Lighthouse 90+, colorblind safe, keyboard navigable
4. Security: XSS prevention, API key protection, CORS lockdown, timeouts
5. Test isolation: all tests fully mocked, CI-safe
6. Dockerize and deploy to Railway

## Open Design Decisions
- CredRank is NOT restaurant-only. Any place with reviews should be rankable (hotels, bars, shops, etc.)
- Geocoding should move server-side to protect the Maps API key
- Anonymous access for beta; auth required before monetization
- Corridor search ("between A and B") and isochrone search ("within X min drive") are V2 features

## File Locations
- Environment: `.env` (gitignored), `.env.example` for reference
- Required env vars: `GOOGLE_MAPS_API_KEY`, `GOOGLE_GEMINI_API_KEY`
- Future env vars: `CLERK_SECRET_KEY`, `STRIPE_SECRET_KEY`, `DATABASE_URL`

## Commands
```bash
uv run uvicorn credrank.api:app --reload --port 8000 --host 0.0.0.0  # Dev server
uv run pytest -v                                                       # Tests
uv run python scripts/smoke_maps.py                                    # Manual smoke test
```
