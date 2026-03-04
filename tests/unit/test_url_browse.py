"""Unit tests for direct URL browsing utilities."""

from __future__ import annotations

from src.url_browse import (
    FetchedPage,
    build_browse_context,
    extract_urls,
    fetch_public_page,
    is_private_or_local_url,
)


class _FakeHeaders:
    def __init__(self, content_type: str):
        self._ct = content_type

    def get(self, key: str, default=None):
        if key.lower() == "content-type":
            return self._ct
        return default


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str, status: int = 200):
        self._body = body
        self.headers = _FakeHeaders(content_type)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _n: int) -> bytes:
        return self._body


def test_extract_urls_dedup_and_order():
    text = (
        "See https://example.com/a and https://example.com/b, "
        "then https://example.com/a again."
    )
    urls = extract_urls(text)
    assert urls == ["https://example.com/a", "https://example.com/b"]


def test_fetch_public_page_html_success(monkeypatch):
    html = b"<html><head><title>Hello</title></head><body><p>World page</p></body></html>"

    def _fake_urlopen(_req, timeout):
        return _FakeResponse(html, "text/html; charset=utf-8")

    monkeypatch.setattr("src.url_browse.urlopen", _fake_urlopen)
    page = fetch_public_page("https://example.com", timeout_sec=2, max_bytes=500_000)
    assert page.ok is True
    assert page.title == "Hello"
    assert "World page" in page.text


def test_fetch_public_page_unsupported_content_type(monkeypatch):
    def _fake_urlopen(_req, timeout):
        return _FakeResponse(b"%PDF-1.7 ...", "application/pdf")

    monkeypatch.setattr("src.url_browse.urlopen", _fake_urlopen)
    page = fetch_public_page("https://example.com/file.pdf", timeout_sec=2, max_bytes=500_000)
    assert page.ok is False
    assert page.error == "unsupported_content_type"


def test_fetch_public_page_timeout(monkeypatch):
    def _fake_urlopen(_req, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("src.url_browse.urlopen", _fake_urlopen)
    page = fetch_public_page("https://example.com", timeout_sec=0.2, max_bytes=500_000)
    assert page.ok is False
    assert page.error == "timeout"


def test_private_or_local_url_detection():
    assert is_private_or_local_url("http://localhost:8080/a") is True
    assert is_private_or_local_url("http://127.0.0.1/x") is True
    assert is_private_or_local_url("http://192.168.1.8/x") is True
    assert is_private_or_local_url("https://example.com/x") is False


def test_fetch_public_page_blocks_private_when_disabled(monkeypatch):
    called = {"v": False}

    def _fake_urlopen(_req, timeout):
        called["v"] = True
        return _FakeResponse(b"hello", "text/plain; charset=utf-8")

    monkeypatch.setattr("src.url_browse.urlopen", _fake_urlopen)
    page = fetch_public_page(
        "http://127.0.0.1/internal",
        timeout_sec=2,
        max_bytes=500_000,
        allow_private=False,
    )
    assert page.ok is False
    assert page.error == "private_url_blocked"
    assert called["v"] is False


def test_build_browse_context_respects_limits():
    pages = [
        FetchedPage(url="https://a", ok=True, title="A", text="x" * 2000),
        FetchedPage(url="https://b", ok=True, title="B", text="y" * 2000),
    ]
    context = build_browse_context(pages, max_chars_per_url=500, max_total_chars=900)
    assert "URL: https://a" in context
    assert len(context) <= 900
