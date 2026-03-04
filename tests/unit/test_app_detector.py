"""Unit tests for browser URL hint extraction."""

from src.app_detector import _extract_url_hint


def test_extract_url_hint_three_part_browser_title_uses_site_name():
    title = "How to write tests - Stack Overflow - Google Chrome"
    assert _extract_url_hint(title) == "Stack Overflow"


def test_extract_url_hint_two_part_browser_title_uses_page_title():
    title = "Pricing - Microsoft Edge"
    assert _extract_url_hint(title) == "Pricing"


def test_extract_url_hint_single_part_title_returns_title():
    title = "localhost:3000"
    assert _extract_url_hint(title) == "localhost:3000"
