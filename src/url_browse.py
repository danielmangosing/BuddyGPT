"""Direct URL browsing utilities for public web pages."""

from __future__ import annotations

import ipaddress
import logging
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

# Limits chosen by product spec.
DEFAULT_PER_URL_TIMEOUT = 6.0
DEFAULT_GLOBAL_TIMEOUT = 20.0
DEFAULT_MAX_BYTES = 1_500_000
DEFAULT_MAX_CHARS_PER_URL = 6000
DEFAULT_MAX_TOTAL_CHARS = 18000

_URL_PATTERN = re.compile(r"https?://[^\s<>()\"']+")


@dataclass(slots=True)
class FetchedPage:
    """Result of one public URL fetch attempt."""

    url: str
    ok: bool
    title: str = ""
    text: str = ""
    content_type: str = ""
    error: str = ""
    status_code: int | None = None
    duration_ms: int = 0


class _MainTextHTMLParser(HTMLParser):
    """Minimal HTML text extractor with script/style suppression."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_depth = 0
        self._parts: list[str] = []
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        tag_l = tag.lower()
        if tag_l in {"script", "style", "noscript"}:
            self._ignore_depth += 1
        if tag_l == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l in {"script", "style", "noscript"} and self._ignore_depth > 0:
            self._ignore_depth -= 1
        if tag_l == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignore_depth > 0:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text
        self._parts.append(text)

    @property
    def text(self) -> str:
        return " ".join(self._parts).strip()


def extract_urls(text: str) -> list[str]:
    """Extract and dedupe http/https URLs while preserving order."""
    raw = _URL_PATTERN.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for url in raw:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        clean = url.rstrip(".,;:!?)]}")
        if clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def _decode_body(data: bytes, content_type: str) -> str:
    charset = "utf-8"
    m = re.search(r"charset=([a-zA-Z0-9_-]+)", content_type)
    if m:
        charset = m.group(1)
    try:
        return data.decode(charset, errors="replace")
    except LookupError:
        return data.decode("utf-8", errors="replace")


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_supported_content_type(content_type: str) -> bool:
    c = content_type.lower()
    return c.startswith("text/") or "json" in c or "xml" in c


def _is_private_or_local_host(host: str) -> bool:
    host_l = host.strip().lower()
    if not host_l:
        return False
    if host_l in {"localhost"}:
        return True
    if host_l.endswith(".local") or host_l.endswith(".internal"):
        return True
    try:
        ip = ipaddress.ip_address(host_l)
    except ValueError:
        return False
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
    )


def is_private_or_local_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    return _is_private_or_local_host(host)


def fetch_public_page(
    url: str,
    timeout_sec: float,
    max_bytes: int,
    allow_private: bool = True,
) -> FetchedPage:
    """Fetch one public page and return cleaned text payload."""
    start = time.monotonic()

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return FetchedPage(url=url, ok=False, error="unsupported_url_scheme")
    if is_private_or_local_url(url):
        if not allow_private:
            return FetchedPage(url=url, ok=False, error="private_url_blocked")
        logger.warning(
            "event=URL_PRIVATE_ALLOWED flow=url_browse url=%s result=allowed reason=private_or_local_host",
            url,
        )

    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            status = getattr(resp, "status", None)
            content_type = (resp.headers.get("Content-Type") or "").strip()
            if not _is_supported_content_type(content_type):
                return FetchedPage(
                    url=url,
                    ok=False,
                    error="unsupported_content_type",
                    content_type=content_type,
                    status_code=status,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )

            body = resp.read(max_bytes + 1)
            if len(body) > max_bytes:
                return FetchedPage(
                    url=url,
                    ok=False,
                    error="payload_too_large",
                    content_type=content_type,
                    status_code=status,
                    duration_ms=int((time.monotonic() - start) * 1000),
                )
    except HTTPError as exc:
        return FetchedPage(
            url=url,
            ok=False,
            error=f"http_error_{exc.code}",
            status_code=exc.code,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except URLError as exc:
        return FetchedPage(
            url=url,
            ok=False,
            error=f"url_error_{exc.reason}",
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except TimeoutError:
        return FetchedPage(
            url=url,
            ok=False,
            error="timeout",
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception:
        logger.exception("URL fetch failed: %s", url)
        return FetchedPage(
            url=url,
            ok=False,
            error="fetch_exception",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    decoded = _decode_body(body, content_type)
    title = ""
    text = ""
    if "html" in content_type.lower():
        parser = _MainTextHTMLParser()
        parser.feed(decoded)
        title = parser.title
        text = parser.text
    else:
        text = decoded

    normalized = _normalize_text(text)
    return FetchedPage(
        url=url,
        ok=True,
        title=title,
        text=normalized,
        content_type=content_type,
        status_code=status,
        duration_ms=int((time.monotonic() - start) * 1000),
    )


def build_browse_context(
    pages: Iterable[FetchedPage],
    max_chars_per_url: int = DEFAULT_MAX_CHARS_PER_URL,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> str:
    """Build bounded text context from fetched pages."""
    parts: list[str] = []
    separator = "\n\n---\n\n"
    total = 0
    for page in pages:
        if not page.ok:
            continue
        body = page.text[:max_chars_per_url]
        chunk = (
            f"URL: {page.url}\n"
            f"Title: {page.title or '(no title)'}\n"
            f"Content: {body}"
        )
        prefix_len = len(separator) if parts else 0
        if total + prefix_len + len(chunk) > max_total_chars:
            remain = max_total_chars - total - prefix_len
            if remain <= 0:
                break
            chunk = chunk[:remain]
        parts.append(chunk)
        total += prefix_len + len(chunk)
        if total >= max_total_chars:
            break
    return separator.join(parts)
