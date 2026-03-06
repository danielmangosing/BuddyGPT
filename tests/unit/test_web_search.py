"""Tests for web-search caching behavior."""

from __future__ import annotations

from src import web_search


class _FakeDDGS:
    def __init__(self, calls: dict[str, int]):
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def text(self, query: str, max_results: int = 3):
        self._calls["n"] += 1
        return [
            {
                "title": query,
                "href": "https://example.com",
                "body": f"results={max_results}",
            }
        ]


def test_search_cache_reuses_result_within_ttl(monkeypatch):
    calls = {"n": 0}
    web_search.clear_search_cache()
    monkeypatch.setattr("src.web_search.DDGS", lambda: _FakeDDGS(calls))
    monkeypatch.setattr("src.web_search.time.monotonic", lambda: 100.0)

    first = web_search.search("BuddyGPT", cache_ttl_sec=90)
    second = web_search.search("BuddyGPT", cache_ttl_sec=90)

    assert calls["n"] == 1
    assert first == second


def test_search_cache_expires_after_ttl(monkeypatch):
    calls = {"n": 0}
    now = {"t": 100.0}
    web_search.clear_search_cache()
    monkeypatch.setattr("src.web_search.DDGS", lambda: _FakeDDGS(calls))
    monkeypatch.setattr("src.web_search.time.monotonic", lambda: now["t"])

    web_search.search("BuddyGPT", cache_ttl_sec=30)
    now["t"] = 200.0
    web_search.search("BuddyGPT", cache_ttl_sec=30)

    assert calls["n"] == 2
