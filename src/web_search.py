"""Web search via DuckDuckGo with a small in-memory TTL cache."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from ddgs import DDGS

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.Lock()
_SEARCH_CACHE: dict[tuple[str, int], tuple[float, list[dict[str, str]]]] = {}
_MAX_CACHE_ENTRIES = 64
_DEFAULT_CACHE_TTL_SEC = 90


def _cache_key(query: str, max_results: int) -> tuple[str, int]:
    normalized = " ".join(str(query).strip().lower().split())
    return normalized, int(max_results)


def clear_search_cache() -> None:
    with _CACHE_LOCK:
        _SEARCH_CACHE.clear()


def _evict_stale_entries(now_mono: float, ttl_sec: int) -> None:
    stale_keys = [
        key
        for key, (ts, _value) in _SEARCH_CACHE.items()
        if ttl_sec <= 0 or (now_mono - ts) > ttl_sec
    ]
    for key in stale_keys:
        _SEARCH_CACHE.pop(key, None)

    if len(_SEARCH_CACHE) <= _MAX_CACHE_ENTRIES:
        return
    oldest_first = sorted(_SEARCH_CACHE.items(), key=lambda item: item[1][0])
    for key, _value in oldest_first[: len(_SEARCH_CACHE) - _MAX_CACHE_ENTRIES]:
        _SEARCH_CACHE.pop(key, None)


def _copy_results(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "snippet": str(item.get("snippet", "")),
        }
        for item in results
    ]


def search(query: str, max_results: int = 3, *, cache_ttl_sec: int = _DEFAULT_CACHE_TTL_SEC) -> list[dict]:
    """Search the web and return top results."""
    ttl_sec = max(0, int(cache_ttl_sec))
    cache_key = _cache_key(query, max_results)
    now_mono = time.monotonic()
    if ttl_sec > 0:
        with _CACHE_LOCK:
            _evict_stale_entries(now_mono, ttl_sec)
            cached = _SEARCH_CACHE.get(cache_key)
            if cached is not None:
                logger.info("Search cache hit: query='%s' results=%d", query, len(cached[1]))
                return _copy_results(cached[1])

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        logger.info("Search '%s' -> %d results", query, len(results))
        normalized_results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
        if ttl_sec > 0:
            with _CACHE_LOCK:
                _evict_stale_entries(time.monotonic(), ttl_sec)
                _SEARCH_CACHE[cache_key] = (time.monotonic(), _copy_results(normalized_results))
        return normalized_results
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        return []


def format_results(results: list[dict]) -> str:
    """Format search results into a readable string for the AI."""
    if not results:
        return "No results found."
    parts = []
    for r in results:
        parts.append(f"- {r['title']}\n  {r['snippet']}\n  {r['url']}")
    return "\n\n".join(parts)
