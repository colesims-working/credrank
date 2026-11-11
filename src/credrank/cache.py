"""In-memory TTL + LRU caches for billable upstream calls.

Every Google Places Text Search page, Place Details fetch, and Geocoding
call is individually billed. Users repeat searches constantly (same
query, same neighborhood, tweaked radius), so a modest in-process cache
eliminates a large fraction of spend at zero quality cost — ratings and
review counts move on the scale of days, not minutes.

Four caches, all configurable via env (seconds):

- ``search_cache``   / ``CREDRANK_SEARCH_CACHE_TTL_S``   (default 24h)
- ``details_cache``  / ``CREDRANK_DETAILS_CACHE_TTL_S``  (default 24h)
- ``geocode_cache``  / ``CREDRANK_GEOCODE_CACHE_TTL_S``  (default 30d)
- ``intent_cache``   / ``CREDRANK_INTENT_CACHE_TTL_S``   (default 7d)

Google's Places API policy allows caching most place fields for up to
30 days (place IDs indefinitely) — the defaults stay well inside that.

Thread-safe: FastAPI runs sync endpoints in a thread pool and grid
search fans out to worker threads, so every operation takes the lock.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from time import monotonic
from typing import Any, Hashable

# Sentinel distinguishing "cached None" (e.g. place not found, geocode
# miss — worth caching too!) from "not in cache".
MISS = object()


def _ttl_from_env(var: str, default_s: int) -> float:
    raw = os.environ.get(var, "").strip()
    if not raw:
        return float(default_s)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(default_s)


class TTLCache:
    """Bounded LRU cache where entries expire after ``ttl_s`` seconds."""

    def __init__(self, maxsize: int, ttl_s: float) -> None:
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Any:
        """Return the cached value, or the ``MISS`` sentinel."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return MISS
            expires_at, value = entry
            if monotonic() >= expires_at:
                del self._data[key]
                self.misses += 1
                return MISS
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: Hashable, value: Any) -> None:
        if self.ttl_s <= 0:  # TTL of 0 disables the cache entirely
            return
        with self._lock:
            self._data[key] = (monotonic() + self.ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "size": len(self._data),
                "maxsize": self.maxsize,
                "ttl_s": self.ttl_s,
                "hits": self.hits,
                "misses": self.misses,
            }


search_cache = TTLCache(
    maxsize=1024, ttl_s=_ttl_from_env("CREDRANK_SEARCH_CACHE_TTL_S", 24 * 3600)
)
details_cache = TTLCache(
    maxsize=2048, ttl_s=_ttl_from_env("CREDRANK_DETAILS_CACHE_TTL_S", 24 * 3600)
)
geocode_cache = TTLCache(
    maxsize=2048, ttl_s=_ttl_from_env("CREDRANK_GEOCODE_CACHE_TTL_S", 30 * 24 * 3600)
)
intent_cache = TTLCache(
    maxsize=1024, ttl_s=_ttl_from_env("CREDRANK_INTENT_CACHE_TTL_S", 7 * 24 * 3600)
)

_ALL = (search_cache, details_cache, geocode_cache, intent_cache)


def clear_all() -> None:
    """Empty every cache — used by tests and available for ops."""
    for cache in _ALL:
        cache.clear()


def all_stats() -> dict[str, dict[str, int | float]]:
    return {
        "search": search_cache.stats(),
        "details": details_cache.stats(),
        "geocode": geocode_cache.stats(),
        "intent": intent_cache.stats(),
    }
